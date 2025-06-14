import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, dim_model):
        super().__init__()
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis].numpy()
        i = tf.range(dim_model, dtype=tf.float32)[tf.newaxis, :].numpy()
        angle_rates = 1 / (10000.0 ** (2 * (i // 2) / dim_model))
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.convert_to_tensor(angle_rads[tf.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class TransformerEncoder(tf.keras.Model):
    def __init__(self, seq_len=20, input_dim=25, model_dim=512, num_heads=8, num_layers=3, mlp_dim=256, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.input_proj = layers.Dense(model_dim)
        self.pos_enc = PositionalEncoding(seq_len, model_dim)

        self.attention_layers = []
        self.ffn_layers = []
        self.norm1_layers = []
        self.norm2_layers = []

        for _ in range(num_layers):
            self.attention_layers.append(
                layers.MultiHeadAttention(
                    num_heads=num_heads, 
                    key_dim=model_dim // num_heads, 
                    dropout=dropout_rate
                )
            )
            self.ffn_layers.append(
                keras.Sequential([
                    layers.Dense(model_dim * 4, activation='relu'),
                    layers.Dropout(dropout_rate),
                    layers.Dense(model_dim),
                ])
            )
            self.norm1_layers.append(layers.LayerNormalization(epsilon=1e-6))
            self.norm2_layers.append(layers.LayerNormalization(epsilon=1e-6))

        self.output_mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(mlp_dim)
        ])

    def call(self, x, training=False, return_sequence=False):
        x = self.input_proj(x)
        x = self.pos_enc(x)

        for i in range(self.num_layers):
            # 添加残差连接保护
            attn_output = self.attention_layers[i](
                self.norm1_layers[i](x, training=training), 
                self.norm1_layers[i](x, training=training),
                training=training
            )
            x = x + attn_output
            x = tf.clip_by_value(x, -1e3, 1e3)  # 添加激活值截断

            ffn_output = self.ffn_layers[i](
                self.norm2_layers[i](x, training=training), 
                training=training
            )
            x = x + ffn_output
            x = tf.clip_by_value(x, -1e3, 1e3)

        x = self.output_mlp(x)
        return x if return_sequence else tf.reduce_mean(x, axis=1)
    
    # 新添加的模型结构打印功能
    def print_structure(self):
        print("⚙️ Transformer Encoder Architecture Details ⚙️")
        print(f"• Positional Encoding: Sequence Length={self.seq_len}, Model Dim={self.model_dim}")
        print(f"• Input Projection: {self.input_proj.__class__.__name__} (input_dim → {self.model_dim})")
        print(f"• Transformer Layers: {self.num_layers} × [")
        print(f"    ↳ {self.attention_layers[0].__class__.__name__} ({self.num_heads} heads, key_dim={self.model_dim//self.num_heads})")
        print(f"    ↳ LayerNormalization (epsilon=1e-6)")
        print(f"    ↳ FFN: Dense({self.model_dim}→{self.model_dim*4})→ReLU→Dense({self.model_dim*4}→{self.model_dim})")
        print("  ]")
        print(f"• Output Projection: {self.output_mlp.layers[0].__class__.__name__} ({self.model_dim}→{self.mlp_dim})")
        
        # 计算参数总量
        total_params = sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        print(f"\n📊 Model Parameters Summary:")
        print(f"• Total Parameters: {total_params:,}")
        print(f"• Per Layer Distribution:")
        layer_names = [f"attention_layer_{i}" for i in range(self.num_layers)] + \
                     [f"ffn_layer_{i}" for i in range(self.num_layers)] + \
                     [f"norm1_layer_{i}" for i in range(self.num_layers)] + \
                     [f"norm2_layer_{i}" for i in range(self.num_layers)]
        
        layers_list = self.attention_layers + self.ffn_layers + \
                     self.norm1_layers + self.norm2_layers
                     
        for name, layer in zip(layer_names, layers_list):
            params = sum([np.prod(v.get_shape().as_list()) for v in layer.trainable_variables])
            print(f"    - {name}: {params:,} params")
        
        print("\n🎯 Key Implementation Notes:")
        print("- PositionalEncoding uses sin/cos functions per paper Eq.1")
        print(f"- Residual connections in all {self.num_layers} Transformer blocks")
        print("- Dual output modes: return_sequence=True (full sequence) or False (mean pooling)")
        print(f"- Output dimension: {self.mlp_dim} dim features")
        print("- Gradient protection: tf.clip_by_value(-1e3, 1e3) applied")

# 创建模型示例并打印结构
if __name__ == "__main__":
    transformer = TransformerEncoder(
        seq_len=20,
        input_dim=25,
        model_dim=512,
        num_heads=8,
        num_layers=3,
        mlp_dim=256
    )
    
    # 构建模型（输入形状：batch_size × sequence_length × input_dim）
    transformer.build(input_shape=(None, 20, 25))
    
    # 打印模型结构
    transformer.print_structure()