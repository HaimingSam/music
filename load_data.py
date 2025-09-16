# 导入库
from datasets import load_dataset

# 直接加载数据集
# 这里使用仓库名称，而不是直接的文件URL
dataset = load_dataset("loubb/aria-midi", data_files="aria-midi-v1-pruned-ext.tar.gz")

# 查看数据集结构
print(dataset)

# 访问训练集数据
print(dataset["train"][0])  # 查看第一个样本
