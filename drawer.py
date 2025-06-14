import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss_curve(csv_path="res_hr/loss_metrics.csv", save_path="res_hr/loss_curve.png"):
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"[错误] 找不到文件：{csv_path}")
        return

    # 加载 CSV 文件
    df = pd.read_csv(csv_path)

    # 按阶段分开处理
    global_df = df[df["phase"] == "global"]
    local_df = df[df["phase"] == "local"]

    # 获取连续数据
    all_epochs = df["epoch"].tolist()
    loss_means = df["mean_loss"].to_numpy()
    loss_stds = df["std_loss"].to_numpy()

    switch_epoch = len(global_df)  # 阶段切换点

    plt.figure(figsize=(10, 5))

    # 绘制 Global 部分
    plt.plot(all_epochs[:switch_epoch], loss_means[:switch_epoch], label="Global Phase", color='blue')
    plt.fill_between(all_epochs[:switch_epoch],
                     loss_means[:switch_epoch] - loss_stds[:switch_epoch],
                     loss_means[:switch_epoch] + loss_stds[:switch_epoch],
                     color='blue', alpha=0.2)

    # 绘制 Local 部分（从切换点 - 1 开始确保曲线连续）
    plt.plot(all_epochs[switch_epoch - 1:], loss_means[switch_epoch - 1:], label="Local Phase", color='green')
    plt.fill_between(all_epochs[switch_epoch - 1:],
                     loss_means[switch_epoch - 1:] - loss_stds[switch_epoch - 1:],
                     loss_means[switch_epoch - 1:] + loss_stds[switch_epoch - 1:],
                     color='green', alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PUAD Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[完成] 训练曲线已保存为：{save_path}")


if __name__ == "__main__":
    plot_loss_curve()
