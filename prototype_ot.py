import tensorflow as tf
from tensorflow import keras
from keras import layers


class PrototypeOT(tf.keras.layers.Layer):
    def __init__(self, num_global=10, num_local=2, proto_dim=256, 
                 epsilon=0.1, train_local_only=False, 
                 use_sinkhorn=False, sinkhorn_iter=20, **kwargs):  # 显式声明所有参数
        super().__init__(**kwargs)
        self.num_global = num_global
        self.num_local = num_local
        self.proto_dim = proto_dim
        self.epsilon = epsilon
        self.train_local_only = train_local_only
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iter = sinkhorn_iter

    def build(self, input_shape):
        # 原型初始化
        self.global_prototypes = self.add_weight(
            shape=(self.num_global, self.proto_dim),
            initializer='glorot_uniform',
            trainable=not self.train_local_only,
            name='global_prototypes'
        )
        self.local_prototypes = self.add_weight(
            shape=(self.num_local, self.proto_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='local_prototypes'
        )
        super().build(input_shape)
        
    def sinkhorn(self, cost_matrix):
        B = tf.shape(cost_matrix)[0]
        N = tf.shape(cost_matrix)[1]
        
        # 数值稳定性处理
        cost_matrix = tf.clip_by_value(cost_matrix, -10.0, 10.0)
        cost_matrix = tf.math.divide_no_nan(cost_matrix, tf.reduce_max(tf.abs(cost_matrix)) + 1e-8)

        a = tf.ones([B, N], dtype=tf.float32) / tf.cast(N, tf.float32)
        b = tf.ones([B, N], dtype=tf.float32) / tf.cast(N, tf.float32)

        K = tf.exp(-cost_matrix / self.epsilon) + 1e-8

        u = tf.ones_like(a)
        v = tf.ones_like(b)

        # 改用tf.while_loop保证可微分性
        def body(i, u, v):
            K_v = tf.matmul(K, v[..., None])[:, :, 0] + 1e-8
            new_u = a / K_v
            new_u = tf.clip_by_value(new_u, 1e-8, 1e8)
            
            K_u = tf.matmul(new_u[..., None], K, transpose_a=True)[:, :, 0] + 1e-8
            new_v = b / K_u
            new_v = tf.clip_by_value(new_v, 1e-8, 1e8)
            return i+1, new_u, new_v

        _, u, v = tf.while_loop(
            cond=lambda i, *_: i < self.sinkhorn_iter,
            body=body,
            loop_vars=(0, u, v),
            maximum_iterations=self.sinkhorn_iter
        )

        T = u[:, :, None] * K * v[:, None, :]
        return tf.reduce_sum(T, axis=2)

    def call(self, theta_0, return_transport=False):
        beta = tf.concat([self.global_prototypes, self.local_prototypes], axis=0)
        
        # 确保可微分计算
        with tf.GradientTape() as proto_tape:
            proto_tape.watch(beta)
            diff = tf.expand_dims(theta_0, 1) - tf.expand_dims(beta, 0)
            cost_matrix = tf.reduce_sum(diff ** 2, axis=-1)

        if self.use_sinkhorn:
            T = self.sinkhorn(cost_matrix)
        else:
            T = tf.nn.softmax(-cost_matrix / self.epsilon, axis=-1)
            T = tf.clip_by_value(T, 1e-8, 1.0)

        theta_0_prime = tf.matmul(T, beta)
        ot_loss = tf.reduce_mean(tf.reduce_sum(T * cost_matrix, axis=1))

        if return_transport:
            return theta_0_prime, ot_loss, T
        return theta_0_prime, ot_loss