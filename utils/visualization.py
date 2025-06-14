import matplotlib.pyplot as plt
import numpy as np

def plot_feature_series(data, anomaly_mask=None, title="Time Series", save_path=None):
    plt.figure(figsize=(15, 4))
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label=f'Feature {i}', alpha=0.5)
    if anomaly_mask is not None:
        plt.scatter(np.where(anomaly_mask)[0], [0]*np.sum(anomaly_mask), c='red', s=10, label='Anomaly')
    plt.title(title)
    plt.legend(loc='upper right', ncol=5, fontsize=8)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_loss_curve(losses, title="Loss Curve", save_path=None):
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
