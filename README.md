## 数据集下载与解压
访问百度[网盘链接](https://pan.baidu.com/s/1WwHYtdskxHhUGqCbWwqrEQ?pwd=n0jk) 
下载数据集至当前文件夹。
解压：
``` unzip dataset.zip  ```

## 环境准备
建议python版本为3.8，安装依赖：
``` pip install -r requirements.txt```

## 训练
``` python train.py ```
训练完在runs/train/expxx目录下可以查看训练结果，该目录下的wights/best.pt 为训练好的模型权重。如图：
![alt text](img/image.png)

## 测试
``` python test.py  --weights "runs/expxx/weights/best.pt"``` 其中expxx是实际实验中产生的具体文件夹。

## 测试结果查看
runs/test/expxx下是具体的模型测试结果，主要是混淆矩阵confusion_matrix.png 如图。
![alt text](img/confusion_matrix.png)
