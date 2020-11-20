# AutomatedTesting2020

- 选题方向：AI数据扩增

- 姓名：张程昱
- 学号：181250183

## 1 目录结构
│  README.md  \n
│  struct.png\n
├─Data\n
│  │\n
│  ├─cifar-10-batches-py
│  ├─cifar-100-python
│  ├─cifar100_result_data
│  ├─cifar10_result_data
│  └─operation_data_png_cifar100
│      ├─feature
│      ├─oringin
│      ├─rotate
│      ├─sample
│      ├─shift
│      └─zca
├─Demo
├─model
│  ├─cifar10
│  ├─cifar100
│  ├─my_model_cifar10
│  │  ├─composite
│  │  ├─oringin
│  │  └─zca
│  └─my_model_cifar100
│      ├─composite
│      ├─oringin
│      └─zca
│
├─Project
│  │  cifar10.html
│  │  cifar100.html
│  │  demo.py
│  │  eval.py
│  │  load.py
│  │  methods.py
│  │  social_network.py
│  │  train_model.py
│  │  visialize.py
│  │
│  └─__pycache__
└─Report
        Report.pdf
        Report.md



- Data：下是起始数据集和扩增后的数据集
- Model：目录下存原模型和我（张程昱）训练的模型、
- Project：代码部分
  - demo.py 是主入口，调用其它函数的各个方法。
  - eval.py 使用模型评估数据（验证测试正确率）
  - load.py 加载文件
  - methods.py 是不同的图像处理方法
  - visialize.py 可视化的相关代码
  - train_model.py 训练代码

## 2 第三方库
tensorflow:2.3.1

keras:2.4.3

numpy:1.18.5

python:3.7.0



