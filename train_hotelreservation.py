import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
import csv

from transformer import TransformerEncoder 
from prototype_ot import PrototypeOT
from vae_loss import VAEBridge, VAEDecoder, compute_total_loss
from sklearn.preprocessing import StandardScaler


class Config:
    seq_len = 20
    stride = 1
    input_dim = 25
    model_dim = 512
    mlp_dim = 256
    num_heads = 8
    num_layers = 3
    proto_dim = 256
    num_global_epochs = 50
    num_local_epochs = 150
    batch_size = 16
    lr = 2e-5
    rho1_init = 0.01
    rho2_init = 0.01
    rho3_init = 0.01
    warmup_epochs = 20
    save_dir = "res_hr"
    # 这里必须是不带后缀的ckpt路径
    model_weight_path = "puad_model2.ckpt"  


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
            num_global=10,
            num_local=2,
            proto_dim=256,
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
    os.makedirs(cfg.save_dir, exist_ok=True)  # 确保loss保存目录存在
    loader = SafeDataLoader(cfg)
    train_data, _ = loader.load_dataset()

    if len(train_data) < cfg.batch_size:
        new_batch_size = max(1, len(train_data) // 2)
        print(f"自动调整batch_size: {cfg.batch_size} -> {new_batch_size}")
        cfg.batch_size = new_batch_size

    train_loader = loader.get_dataloader(train_data)
    model = PUADModel(cfg)
    dummy_input = tf.keras.Input(shape=(cfg.seq_len, cfg.input_dim))
    model(dummy_input)  # 触发build

    optimizer = optimizers.Adam(cfg.lr, clipnorm=1.0)

    loss_csv_path = os.path.join(cfg.save_dir, "loss_metrics.csv")
    resume_epoch = 0

    # 检查loss记录文件
    if os.path.exists(loss_csv_path):
        with open(loss_csv_path, mode='r') as file:
            reader = list(csv.reader(file))
            resume_epoch = len(reader) - 1  # 减去表头
            print(f"已存在loss记录文件，共 {resume_epoch} 轮训练记录")
    else:
        # 写入loss文件表头
        with open(loss_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "phase", "mean_loss", "std_loss"])

    # 判断是否继续训练
    if resume_epoch <= 300:
        if os.path.exists(cfg.model_weight_path + ".index"):
            print(f"继续训练：加载已有权重 {cfg.model_weight_path}，从第 {resume_epoch + 1} 轮开始")
            model.load_weights(cfg.model_weight_path)
        else:
            print("未找到权重文件，重新开始训练")
    else:
        print(f"训练已完成 {resume_epoch} 轮，大于300，跳过训练")
        return

    total_epochs = cfg.num_global_epochs + cfg.num_local_epochs
    for epoch in range(resume_epoch, total_epochs):
        use_local = epoch >= cfg.num_global_epochs
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
        phase_str = "local" if use_local else "global"
        print(f"Epoch {epoch + 1} {phase_str} phase: mean_loss={mean_loss:.4f}, std_loss={std_loss:.4f}")

        # 追加写入 loss 记录
        with open(loss_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, phase_str, mean_loss, std_loss])

        # 保存当前模型权重
        model.save_weights(cfg.model_weight_path)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    safe_train()
