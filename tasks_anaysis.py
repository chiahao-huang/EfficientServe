import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# --- 核心：解决中文乱码问题的配置 ---
#
# 1. 设置字体
# 根据你系统上的字体列表，我们选择 'Droid Sans Fallback'
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback'] 
# 
# 2. 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False 
# ------------------------------------
# --- 1. 数据加载与处理 ---
# 定义文件名
file_path = '/root/hjh/llminference/vllm_inert/vllm-0.5.0.post1/plugin/EfficientServe/sample_tasks_from_datasets/sampled_experiment_datasets/sampled_alpaca-qwen2-5-14b.json'

# 读取 JSON 文件
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。请确保文件路径正确。")
    # 创建一些示例数据以便演示
    print("正在使用示例数据生成图表...")
    data = [[i, "instruction", 10 + (i % 50) * 2, 50 + (i % 100) * 3, "alpaca"] for i in range(1000)]
except json.JSONDecodeError:
    print(f"错误：文件 '{file_path}' 不是有效的 JSON 格式。")
    data = []

if data:
    # 将数据转换为 Pandas DataFrame
    df = pd.DataFrame(data, columns=['id', 'instruction', 'input_length', 'output_length', 'type'])

    # --- 2. 绘制分布对比图 ---
    # 设置 Seaborn 的绘图风格
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7)) # 设置图表尺寸

    # 绘制输入长度的分布图 (直方图 + KDE曲线)
    sns.histplot(df['input_length'], color="skyblue", kde=True, label='Input Length', bins=50)

    # 绘制输出长度的分布图 (直方图 + KDE曲线)
    sns.histplot(df['output_length'], color="salmon", kde=True, label='Output Length', bins=50)

    # --- 3. 添加图表元素，使其更清晰 ---
    plt.title('Alpaca数据集的输入输出词元长度分布', fontsize=16, fontweight='bold')
    plt.xlabel('长度 (词元个数)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.legend() # 显示图例

    # 优化 x 轴范围，避免极端值影响可读性 (可根据实际数据调整)
    # 例如，如果99%的数据都小于1000，可以这样设置：
    # combined_lengths = pd.concat([df['input_length'], df['output_length']])
    # plt.xlim(0, combined_lengths.quantile(0.99))
    
    plt.tight_layout()
    plt.savefig("alpaca_distribution.png", dpi=200)
    plt.show()
