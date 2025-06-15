import requests
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce

# === 配置项 ===
PROM_URL = "http://127.0.0.1:24368"
STEP = "15s"

# 设置时间范围（确保覆盖故障）
start_time = datetime.fromisoformat("2025-06-03T03:53:45")
end_time = datetime.fromisoformat("2025-06-03T04:08:45")

# 故障注入时间（UTC）
FAULT_START = datetime.fromisoformat("2025-06-03T03:59:45")
FAULT_END = datetime.fromisoformat("2025-06-03T04:05:45")

# Prometheus 查询指标 + 别名（共25个）
METRICS = {
    'container_cpu_usage_seconds_total{pod="catalogue-55b6f8c75f-s2svb"}': "pod_cpu",
    'container_memory_usage_bytes{pod="catalogue-55b6f8c75f-s2svb"}': "pod_mem",
    'container_network_receive_bytes_total{pod="catalogue-55b6f8c75f-s2svb"}': "pod_net_in",
    'container_network_transmit_bytes_total{pod="catalogue-55b6f8c75f-s2svb"}': "pod_net_out",
    'container_fs_usage_bytes{pod="catalogue-55b6f8c75f-s2svb"}': "pod_fs_usage",
    'kube_pod_container_status_restarts_total{pod="catalogue-55b6f8c75f-s2svb"}': "pod_restart",
    'kube_pod_container_status_running{container="catalogue"}': "pod_running",
    'node_cpu_seconds_total{instance="minikube",mode="user"}': "node_cpu_user",
    'node_cpu_seconds_total{instance="minikube",mode="system"}': "node_cpu_sys",
    'node_cpu_seconds_total{instance="minikube",mode="idle"}': "node_cpu_idle",
    'node_load1{instance="minikube"}': "node_load1",
    'node_memory_MemAvailable_bytes{instance="minikube"}': "node_mem_avail",
    'node_memory_MemTotal_bytes{instance="minikube"}': "node_mem_total",
    'node_filesystem_avail_bytes{instance="minikube",mountpoint="/"}': "node_disk_avail",
    'up{pod="catalogue-55b6f8c75f-s2svb"}': "pod_up",
    'process_start_time_seconds{pod="catalogue-55b6f8c75f-s2svb"}': "pod_uptime",
    'container_start_time_seconds{pod="catalogue-55b6f8c75f-s2svb"}': "pod_container_start",
    'container_last_seen{pod="catalogue-55b6f8c75f-s2svb"}': "pod_last_seen",
    'container_spec_memory_limit_bytes{pod="catalogue-55b6f8c75f-s2svb"}': "pod_mem_limit",
    'container_spec_cpu_quota{pod="catalogue-55b6f8c75f-s2svb"}': "pod_cpu_quota",
    'container_spec_cpu_period{pod="catalogue-55b6f8c75f-s2svb"}': "pod_cpu_period",
    'container_spec_memory_reservation_limit_bytes{pod="catalogue-55b6f8c75f-s2svb"}': "pod_mem_reserve",
    'container_threads{pod="catalogue-55b6f8c75f-s2svb"}': "pod_threads",
    'container_tasks_state{pod="catalogue-55b6f8c75f-s2svb",state="running"}': "pod_tasks_running",
    'container_tasks_state{pod="catalogue-55b6f8c75f-s2svb",state="blocked"}': "pod_tasks_blocked"
}

# === 查询函数 ===
def query_range(metric, start, end):
    url = f"{PROM_URL}/api/v1/query_range"
    params = {
        "query": metric,
        "start": start.isoformat() + "Z",
        "end": end.isoformat() + "Z",
        "step": STEP
    }
    print(f"🔍 正在查询: {metric}")
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        result = []
        for series in r.json()["data"]["result"]:
            for ts, val in series["values"]:
                dt = datetime.utcfromtimestamp(float(ts))
                result.append([dt, float(val)])
        return pd.DataFrame(result, columns=["timestamp", METRICS[metric]])
    except Exception as e:
        print(f"⚠️ 查询失败: {metric}，错误: {e}")
        return pd.DataFrame(columns=["timestamp", METRICS[metric]])

# === 主函数 ===
def main():
    dfs = [query_range(metric, start_time, end_time) for metric in METRICS]
    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        print("❌ 没有可用数据")
        return

    df_all = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dfs)
    df_all.sort_values("timestamp", inplace=True)
    df_all.fillna(method="ffill", inplace=True)
    df_all.dropna(inplace=True)

    # 添加标签列
    df_all["label"] = df_all["timestamp"].apply(
        lambda x: 1 if FAULT_START <= x <= FAULT_END else 0
    )

    # 格式化时间戳便于 Excel 显示
    df_all["timestamp"] = df_all["timestamp"].dt.strftime("%Y/%m/%d %H:%M:%S")

    # 导出为 CSV
    output_path = "C:/Users/Lenovo/Desktop/Software Testing and Maintenance/lastlab/6.csv"
    df_all.to_csv(output_path, index=False)
    print(f"✅ 已保存至：{output_path}")

if __name__ == "__main__":
    main()
