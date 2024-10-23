import pandas as pd

# 读取 CSV 文件
evs_data = pd.read_csv('test_demo/EVs_50.csv', header=None)

# 获取行数
num_rows = evs_data.shape[0]

# 打印行数
print(f"EVs_50.csv 的行数是: {num_rows}")