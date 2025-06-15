import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# === 设置中文支持和负号显示 ===
plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文字体
plt.rcParams['axes.unicode_minus'] = False        # 正负号正常显示

# === 1. 读取原始 1~6 文件，并添加 group_id ===
file_paths = [f"./{i}.csv" for i in range(1, 7)] 
dataframes = [pd.read_csv(path) for path in file_paths]
for i, df in enumerate(dataframes):
    df['group_id'] = i + 1

# === 2. 合并原始数据并保存 ===
original_df = pd.concat(dataframes, ignore_index=True)
original_df.to_csv("raw_with_group2.csv", index=False)

# === 3. Min-Max归一化 ===
minmax_dfs = []
for df in dataframes:
    numeric_cols = df.select_dtypes(include='number').columns.drop('group_id')
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    minmax_dfs.append(df_scaled)

# 合并归一化后的数据
df_minmax = pd.concat(minmax_dfs, ignore_index=True)
df_minmax.to_csv("minmax_normalized_data3.csv", index=False)

# === 4. 可视化：对比归一化前后分布 ===
selected_features = ['pod_cpu', 'pod_mem', 'pod_last_seen', 'pod_threads']
palette = sns.color_palette("Set1", n_colors=6)

fig, axes = plt.subplots(len(selected_features), 2, figsize=(14, 4 * len(selected_features)))
for i, feature in enumerate(selected_features):
    # 原始特征分布
    sns.kdeplot(data=original_df, x=feature, hue='group_id', ax=axes[i, 0], palette=palette)
    axes[i, 0].set_title(f"原始特征分布 - {feature}", fontsize=14)
    axes[i, 0].set_xlabel(feature)
    axes[i, 0].set_ylabel("密度")

    # Min-Max归一化后分布
    sns.kdeplot(data=df_minmax, x=feature, hue='group_id', ax=axes[i, 1], palette=palette)
    axes[i, 1].set_title(f"归一化后特征分布 - {feature}", fontsize=14)
    axes[i, 1].set_xlabel(feature)
    axes[i, 1].set_ylabel("密度")

# 调整子图间距，防止标题/坐标轴重叠
fig.subplots_adjust(hspace=0.5)
plt.show()
