from transformer import TransformerEncoder 
from prototype_ot import PrototypeOT
from vae_loss import VAEBridge, VAEDecoder, compute_total_loss
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import csv

# 配置参数
class Config:
    seq_len = 20
    stride = 1
    input_dim = 25
    model_dim = 512
    mlp_dim = 256
    num_heads = 8
    num_layers = 3
    proto_dim = 256
    num_global = 10
    num_local = 2
    epochs = 150
    batch_size = 16
    lr = 2e-5
    rho1_init = 0.01
    rho2_init = 0.01
    rho3_init = 0.01
    warmup_epochs = 20

class SafeDataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def load_dataset(self):
        train_df = pd.read_csv("Dataset/sockshop/train.csv", index_col=None)
        test_df = pd.read_csv("Dataset/sockshop/test.csv", index_col=None)

        train_features = train_df.drop(columns=["timestamp"], errors="ignore")
        test_features = test_df.drop(columns=["timestamp"], errors="ignore")

        train_data = self.scaler.fit_transform(np.nan_to_num(train_features.values, nan=0.0))
        test_data = self.scaler.transform(np.nan_to_num(test_features.values, nan=0.0))

        def create_sequences(data):
            return np.array([
                data[i:i + self.config.seq_len]
                for i in range(0, len(data) - self.config.seq_len + 1, self.config.stride)
            ])

        return create_sequences(train_data), create_sequences(test_data)

    def get_dataloader(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        return dataset.shuffle(1024).batch(self.config.batch_size).prefetch(2)

class PUADModel(tf.keras.Model):
    def __init__(self, config, train_local_only=False):
        super().__init__()
        self.config = config
        self.train_local_only = train_local_only
        self.encoder = TransformerEncoder(
            seq_len=config.seq_len,
            input_dim=config.input_dim,
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            mlp_dim=config.mlp_dim
        )
        self.prototype_ot = PrototypeOT(
            num_global=config.num_global,
            num_local=config.num_local,
            proto_dim=config.proto_dim,
            epsilon=0.5,
            sinkhorn_iter=5,
            train_local_only=train_local_only,
            use_sinkhorn=False
        )
        self.vae_bridge = VAEBridge(input_dim=config.mlp_dim)
        self.decoder = VAEDecoder(
            latent_dim=config.model_dim,
            seq_len=config.seq_len,
            output_dim=config.input_dim
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.encoder.build(input_shape)
        self.prototype_ot.build((None, self.config.mlp_dim))
        self.vae_bridge.build((None, self.config.mlp_dim))
        self.decoder.build((None, self.config.model_dim))

    def call(self, x, training=False):
        theta_0 = self.encoder(x, training=training)
        theta_0_prime, ot_loss = self.prototype_ot(theta_0)
        theta_1, mu_1, logvar_1 = self.vae_bridge(theta_0_prime)
        x_recon, mu_2, logvar_2 = self.decoder(theta_1)
        return x_recon, mu_1, logvar_1, theta_0_prime, ot_loss

def safe_train():
    cfg = Config()
    loader = SafeDataLoader(cfg)

    train_data, _ = loader.load_dataset()

    if len(train_data) < cfg.batch_size:
        new_batch_size = max(1, len(train_data) // 2)
        print(f"自动调整batch_size: {cfg.batch_size} -> {new_batch_size}")
        cfg.batch_size = new_batch_size

    train_loader = loader.get_dataloader(train_data)

    model = PUADModel(cfg)
    dummy_input = tf.keras.Input(shape=(cfg.seq_len, cfg.input_dim))
    model(dummy_input)

    optimizer = optimizers.Adam(cfg.lr, clipnorm=1.0)

    global_loss_history = []
    global_loss_std = []
    local_loss_history = []
    local_loss_std = []

    for epoch in range(cfg.epochs):
        use_local = epoch >= cfg.epochs // 2
        model.prototype_ot.train_local_only = use_local

        rho1 = cfg.rho1_init * min((epoch + 1) / cfg.warmup_epochs, 1.0)
        rho2 = cfg.rho2_init * min((epoch + 1) / cfg.warmup_epochs, 1.0)
        rho3 = cfg.rho3_init * min((epoch + 1) / cfg.warmup_epochs, 1.0)

        epoch_losses = []

        for step, batch in enumerate(train_loader):
            with tf.GradientTape() as tape:
                x_recon, mu_1, logvar_1, theta_0_prime, ot_loss = model(batch, training=True)
                loss = compute_total_loss(
                    x=batch,
                    x_recon=x_recon,
                    mu_1=mu_1,
                    logvar_1=logvar_1,
                    mu_0=theta_0_prime,
                    logvar_0=tf.zeros_like(theta_0_prime),
                    ot_loss=ot_loss,
                    rho1=rho1, rho2=rho2, rho3=rho3
                )

            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 5.0) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_losses.append(loss.numpy())

            if step % 10 == 0:
                print(f"Epoch {epoch + 1:3d} Step {step:4d} | Loss: {loss.numpy():.3f} | {'Local' if use_local else 'Global'}")

        mean_loss = np.mean(epoch_losses)
        std_loss = np.std(epoch_losses)

        if use_local:
            local_loss_history.append(mean_loss)
            local_loss_std.append(std_loss)
        else:
            global_loss_history.append(mean_loss)
            global_loss_std.append(std_loss)

    # === 保存模型 ===
    model.save_weights("puad_model.ckpt")

    # === 保存 loss 到 CSV ===
    all_epochs = list(range(1, cfg.epochs + 1))
    loss_means = global_loss_history + local_loss_history
    loss_stds = global_loss_std + local_loss_std
    phases = ['global'] * len(global_loss_history) + ['local'] * len(local_loss_history)

    with open("loss_metrics.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "phase", "mean_loss", "std_loss"])
        for epoch, phase, mean, std in zip(all_epochs, phases, loss_means, loss_stds):
            writer.writerow([epoch, phase, mean, std])

    # === 绘制连续的 loss 曲线，分阶段上色 ===
    plt.figure(figsize=(10, 5))

    all_epochs = list(range(1, cfg.epochs + 1))
    loss_means = global_loss_history + local_loss_history
    loss_stds = global_loss_std + local_loss_std

    switch_epoch = len(global_loss_history)  # 切换点

    # 全部作为一条连续曲线数据
    y_all = np.array(loss_means)
    std_all = np.array(loss_stds)

    # 先画全局部分
    plt.plot(all_epochs[:switch_epoch], y_all[:switch_epoch], label="Global Phase", color='blue')
    plt.fill_between(all_epochs[:switch_epoch],
                     y_all[:switch_epoch] - std_all[:switch_epoch],
                     y_all[:switch_epoch] + std_all[:switch_epoch],
                     color='blue', alpha=0.2)

    # 再画局部部分（从切换点继续，不断线）
    plt.plot(all_epochs[switch_epoch - 1:], y_all[switch_epoch - 1:], label="Local Phase", color='green')
    plt.fill_between(all_epochs[switch_epoch - 1:],
                     y_all[switch_epoch - 1:] - std_all[switch_epoch - 1:],
                     y_all[switch_epoch - 1:] + std_all[switch_epoch - 1:],
                     color='green', alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PUAD Training Loss Curve (Connected, Colored by Phase)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_combined.png")
    plt.close()


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    safe_train()