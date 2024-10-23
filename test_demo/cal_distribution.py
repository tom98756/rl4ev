import pandas as pd
import numpy as np
from scipy.stats import shapiro

# 读取 CSV 文件
file_path = './EVs_50.csv'
data = pd.read_csv(file_path, header=None)

# 提取第 7 列数据（索引从 0 开始，所以第 7 列是索引 6）
column_7_data = data.iloc[:, 6]

# 计算均值和标准差
mean = np.mean(column_7_data)
std_dev = np.std(column_7_data)

# 打印均值和标准差
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")

# 检查数据是否为正态分布
stat, p_value = shapiro(column_7_data)
print(f"Shapiro-Wilk Test Statistic: {stat}")
print(f"P-Value: {p_value}")

# 判断是否为正态分布
alpha = 0.05
if p_value > alpha:
    print("The data is normally distributed (fail to reject H0)")
else:
    print("The data is not normally distributed (reject H0)")