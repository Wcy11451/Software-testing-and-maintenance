### READ
### 📌 1. **数据采集与实验环境**  
#### **数据采集方式**  
- **数据源**：基于Kubernetes的微服务系统`sock-shop`的实时监控指标  
- **监控维度**：25维指标，包括：  
  - CPU使用率、内存占用  
  - 线程数、网络流量  
  - 磁盘I/O、服务状态  
- **采集工具**：  
  - `Prometheus`（端口`24368`），每15秒采集一次（`STEP="15s"`）  
  - Python脚本通过Prometheus API导出CSV文件  
- **故障注入**：  
  - **资源压力故障**（StressChaos）：CPU过载（例如2个工作线程持续3-6分钟）  
  - **节点故障**（PodChaos）：模拟网络中断（例如节点暂停2-5分钟）  
  - **目标服务**：`catalogue`和`payment`等关键微服务  

#### **实验环境**  
- **平台**：  
  - Kubernetes集群 + Chaos Mesh故障注入系统  
  - GPU加速训练（NVIDIA RTX 3090）  
- **数据集**：  
  - `Dataset/sockshop/train.csv`/`test.csv`：标准化处理的时序数据  
  - `test_label.csv`：异常标签（若有）  
- **文件示例**：  
  - `data_1.5.6.csv`：资源压力故障数据  
  - `data_2.3.4.csv`：节点故障数据  

---

### 🤖 2. **PUAD模型简介**  
**原型导向的无监督异常检测（PUAD）** 是一种多变量时序异常检测模型：  
- **核心思想**：  
  - 通过全局/局部原型学习正常模式多样性  
  - 基于重构误差识别异常  
- **架构组成**：  
  1. **Transformer编码器**：捕捉长期时序依赖  
  2. **PrototypeOT模块**：基于最优传输(OT)的原型学习  
  3. **VAE解码器**：概率生成式重构  
- **损失函数**：  
  ```数学公式  
  Loss = 重构误差 + KL散度 + OT距离  
  ```  
- **输出**：重构误差作为异常得分，得分越低越异常  

---

### 🗂 3. **代码文件功能说明**  
| 文件 | 功能描述 |  
|------|-------------|  
| **`vae_loss.py`** | VAE解码器/桥梁层 + KL散度/重构损失计算 |  
| **`prototype_ot.py`** | 全局/局部原型学习（支持Sinkhorn最优传输算法） |  
| **`transformer.py`** | Transformer编码器（含位置编码） |  
| **`train_safe.py`** | 主训练脚本（全局+局部两阶段训练），保存模型权重和损失记录 |  
| **`test.py`** | 模型评估脚本，计算F1/AUC等指标并生成可视化结果 |  
| **`plot_loss.py`** | 从`loss_metrics.csv`生成训练损失曲线 |  
| **`*.yaml`** | Chaos Mesh故障注入配置文件（如`cpu-stress-test.yaml`） |  
| **`export_metrics.py`** | Prometheus指标导出脚本（采集故障期间数据） |  

---

### 🔬 4. **实验流程**  
#### ▶️ 步骤1：故障注入  
1. 部署`sock-shop`微服务系统  
2. 注入故障：  
   ```bash 
   # CPU压力故障（StressChaos） 
   kubectl apply -f cpu-stress-test.yaml 

   # 节点暂停故障（PodChaos） 
   kubectl apply -f pause-catalogue.yaml  
   ```  
3. 记录故障时间窗口（用于数据对齐）  

#### ▶️ 步骤2：数据采集  
运行数据采集脚本：  
```python 
python export_metrics.py \ 
    --start "2025-06-03T03:00:00" \ 
    --end "2025-06-03T03:15:00" 
```  
生成数据文件：  
- `data_1.5.6.csv`：CPU压力故障数据  
- `data_2.3.4.csv`：节点暂停故障数据  

#### ▶️ 步骤3：模型训练  
1. 数据预处理：  
   ```python 
   from utils import create_sequences 
   seqs = create_sequences(data, seq_len=20, stride=1)  # 生成滑动窗口序列 
   ```  
2. 两阶段训练：  
   ```bash 
   python train_safe.py  # 保存模型至puad_model2.ckpt 
   ```  
   - **全局阶段**：学习跨服务的共享原型  
   - **局部阶段**：适配新服务的特有模式  

#### ▶️ 步骤4：评估与可视化  
```bash 
# 评估测试集性能 
python test.py  # 在res_hr/生成指标图表 

# 绘制训练损失曲线 
python plot_loss.py 
```  
输出结果：  
- `evaluation_metrics.png`：F1/精确率/召回率等指标  
- `loss_curve.png`：全局/局部训练阶段损失变化  

#### ▶️ 步骤5：结果分析  
- **定量分析**：对比不同故障类型的F1/AUC得分  
- **定性分析**：  
  - 检查异常得分与故障时间窗口的对齐情况  
  - 分析重构误差的敏感度（如文档8中的异常得分曲线）  

---

### ⚠️ 注意事项  
1. **数据敏感性**：不同故障类型的绝对值差异较大，建议组内分析相对变化  
2. **依赖环境**：  
   - Python 3.8+ / TensorFlow 2.x / scikit-learn  
   - Chaos Mesh命令行工具  
3. **可复现性**：  
   - 固定随机种子  
   - 使用`Config`类统一超参数