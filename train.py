from transformer import TransformerEncoder
from  prototype_ot import PrototypeOT
from vae_loss import VAEBridge,VAEDecoder,compute_total_loss
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

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
    epochs = 100
    batch_size = 32
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
        train_df = pd.read_csv("Dataset/PSM/train.csv")
        test_df = pd.read_csv("Dataset/PSM/test.csv")
        
        train_data = self.scaler.fit_transform(np.nan_to_num(train_df.values[:, 1:], nan=0.0))
        test_data = self.scaler.transform(np.nan_to_num(test_df.values[:, 1:], nan=0.0))
        
        def create_sequences(data):
            return np.array([
                data[i:i+self.config.seq_len] 
                for i in range(0, len(data)-self.config.seq_len+1, self.config.stride)
            ])
            
        return create_sequences(train_data), create_sequences(test_data)

    def get_dataloader(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        return dataset.shuffle(1024).batch(self.config.batch_size).prefetch(2)

class PUADModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 关键修复：保存config参数
        
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
            epsilon=0.5,                # 显式传递参数
            sinkhorn_iter=5,             # 显式传递参数
            train_local_only=False,
            use_sinkhorn=False
        )
        self.vae_bridge = VAEBridge(input_dim=config.mlp_dim)
        self.decoder = VAEDecoder(
            latent_dim=config.model_dim,
            seq_len=config.seq_len,
            output_dim=config.input_dim  # 传递正确维度
        )

    def build(self, input_shape):
        super().build(input_shape)
        # 显式构建所有子组件
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
    
    # 数据加载
    train_data, test_data = loader.load_dataset()
    train_loader = loader.get_dataloader(train_data)
    
    # 模型初始化
    model = PUADModel(cfg)
    
    # 显式构建（两种方式任选其一）
    # 方式1：使用虚拟输入
    dummy_input = tf.keras.Input(shape=(cfg.seq_len, cfg.input_dim))
    model(dummy_input)
    
    # 方式2：使用真实数据
    # for first_batch in train_loader.take(1):
    #     model(first_batch)
    #     break

    # 参数检查
    print("\n=== 可训练参数清单 ===")
    for i, var in enumerate(model.trainable_variables):
        print(f"#{i+1} {var.name:60} | Shape: {var.shape}")
    print(f"总参数数量: {len(model.trainable_variables)}\n")
    
    # 训练循环
    optimizer = optimizers.Adam(cfg.lr, clipnorm=1.0)
    
    for epoch in range(cfg.epochs):
        rho1 = cfg.rho1_init * min((epoch+1)/cfg.warmup_epochs, 1.0)
        rho2 = cfg.rho2_init * min((epoch+1)/cfg.warmup_epochs, 1.0)
        rho3 = cfg.rho3_init * min((epoch+1)/cfg.warmup_epochs, 1.0)
        
        for step, batch in enumerate(train_loader):
            with tf.GradientTape() as tape:
                x_recon, mu_1, logvar_1, theta_0_prime, ot_loss = model(batch, training=True)
                
                # 数值检查
                tf.debugging.assert_all_finite(batch, "输入数据异常")
                tf.debugging.assert_all_finite(x_recon, "重建数据异常")
                
                loss = compute_total_loss(
                    x=batch,
                    x_recon=x_recon,
                    mu_1=mu_1,
                    logvar_1=logvar_1,
                    mu_0=theta_0_prime,
                    logvar_0=tf.zeros_like(theta_0_prime),
                    ot_loss=ot_loss,
                    rho1=rho1,
                    rho2=rho2,
                    rho3=rho3
                )
                tf.debugging.check_numerics(loss, "无效损失值")
                
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 5.0) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            if step % 10 == 0:
                grad_norms = [tf.norm(g).numpy() for g in grads]
                print(
                    f"Epoch {epoch+1:3d} Step {step:4d} | "
                    f"Loss: {loss.numpy():.3f} | "
                    f"OT Loss: {ot_loss.numpy():.3f} | "
                    f"Grad Norm: {np.mean(grad_norms):.2f}±{np.std(grad_norms):.2f}"
                )
    
    model.save_weights("puad_model.ckpt")

if __name__ == "__main__":
    # 配置GPU内存
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    safe_train()