import pandas as pd

# 1. 加载数据
df = pd.read_csv("data/drug.csv")

# 2. 检查列是否存在
if 'pACT' not in df.columns:
    raise ValueError("列 'pACT' 不存在，请确认你的CSV文件格式为：smiles,pACT")

# 3. 去除空值
pact_values = df['pACT'].dropna()

# 4. 计算最大值和最小值
max_pact_hcv = pact_values.max()
min_pact_hcv = pact_values.min()

# 5. 打印结果
print(f"max_pact_hcv = {max_pact_hcv}")
print(f"min_pact_hcv = {min_pact_hcv}")
