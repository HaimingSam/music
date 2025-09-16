# music
## 数据下载
- huggingface

  """""
    export HF_ENDPOINT=https://hf-mirror.com
    python load_data.py
  """""
  
## 数据处理
- 移调增强
  
  """"
    python augmentation.py <词表文件> <输入JSONL> <输出目录>
  """"
