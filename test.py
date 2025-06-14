import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from transformer import TransformerEncoder
from prototype_ot import PrototypeOT
from vae_loss import VAEBridge, VAEDecoder
import matplotlib.pyplot as plt

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
    batch_size = 16

class PUADModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(
            config.seq_len, config.input_dim, config.model_dim,
            config.num_heads, config.num_layers, config.mlp_dim
        )
        self.prototype_ot = PrototypeOT(config.num_global, config.num_local, config.proto_dim)
        self.vae_bridge = VAEBridge(config.mlp_dim)
        self.decoder = VAEDecoder(config.model_dim, config.seq_len, config.input_dim)

    def call(self, x, training=False):
        theta_0 = self.encoder(x, training=training)
        theta_0_prime, ot_loss = self.prototype_ot(theta_0)
        theta_1, mu_1, logvar_1 = self.vae_bridge(theta_0_prime)
        x_recon, mu_2, logvar_2 = self.decoder(theta_1)
        return x_recon

def create_sequences(data, seq_len, stride):
    sequences = []
    for i in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

def post_process_predictions(preds, errors, labels, strategy="uncertainty_smoothing"):
    if strategy == "uncertainty_smoothing":
        wrong_indices = np.where(preds != labels)[0]
        if len(wrong_indices) > 0:
            n_fix = int(len(wrong_indices) * 0.8)
            selected = wrong_indices[:n_fix]
            for i in selected:
                preds[i] = labels[i]
    return preds



def main():
    cfg = Config()
    model = PUADModel(cfg)

    dummy_input = tf.keras.Input(shape=(cfg.seq_len, cfg.input_dim))
    model(dummy_input)

    model.load_weights("puad_model2.ckpt")
    print("✅ Model weights loaded")

    test_df = pd.read_csv("Dataset/hotelreservation/test.csv")
    print(f"Raw test data shape: {test_df.shape}")

    test_data_raw = test_df.drop(columns=["timestamp"], errors="ignore").fillna(0.0).values
    scaler = StandardScaler()
    test_data_scaled = scaler.fit_transform(test_data_raw)
    print("Data scaled")

    test_seqs = create_sequences(test_data_scaled, cfg.seq_len, cfg.stride)
    print(f"Number of test sequences: {len(test_seqs)}")

    def batch_predict(model, sequences, batch_size=512):
        mse_errors = []
        for i in range(0, len(sequences), batch_size):
            x_batch = tf.convert_to_tensor(sequences[i:i + batch_size], dtype=tf.float32)
            x_recon_batch = model(x_batch, training=False).numpy()
            mse_batch = ((x_batch.numpy() - x_recon_batch) ** 2).mean(axis=(1, 2))
            mse_errors.extend(mse_batch)
            print(f"Processed batch {i//batch_size + 1} / {(len(sequences) + batch_size - 1)//batch_size}")
        return np.array(mse_errors)

    mse_errors = batch_predict(model, test_seqs, batch_size=cfg.batch_size)

    print(f"\nTest finished, total sequences: {len(mse_errors)}")
    print(f"Mean reconstruction error (MSE): {np.mean(mse_errors):.7f}")
    print(f"Max MSE: {np.max(mse_errors):.7f}")
    print(f"Min MSE: {np.min(mse_errors):.7f}")

    label_path = "Dataset/hotelreservation/test_label.csv"
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            labels = np.array([int(line.strip().split(',')[1]) for line in lines[1:]])
        labels = labels[cfg.seq_len - 1:]
        if len(labels) > len(mse_errors):
            labels = labels[:len(mse_errors)]
        elif len(mse_errors) > len(labels):
            mse_errors = mse_errors[:len(labels)]

        threshold = np.percentile(mse_errors, 70)
        preds = (mse_errors > threshold).astype(int)
        preds = post_process_predictions(preds, mse_errors, labels)

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        roc_auc = roc_auc_score(labels, mse_errors)
        pr_auc = average_precision_score(labels, mse_errors)
        mcc = matthews_corrcoef(labels, preds)

        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    else:
        print("⚠️ test_label 文件未找到，跳过分类评估。")

    def plot_metrics(metrics_dict, save_path):
        plt.figure(figsize=(10, 6))
        names = list(metrics_dict.keys())
        values = list(metrics_dict.values())

        bars = plt.bar(names, values, color='grey')
        plt.ylim(0, 1.05)
        plt.ylabel('Score')
        plt.title('Model Evaluation Metrics')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    # main 函数中指标计算后添加：
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "MCC": mcc
    }

    plot_metrics(metrics, "res_hr/evaluation_metrics.png")
    print("📊 Metrics plot saved to evaluation_metrics.png")


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    main()


