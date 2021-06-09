
##### 任务介绍:
轴分析任务:
定位图表上的刻度点，同时将刻度点与旁边的文字内容匹配连接起来。

输入内容:
图片的类别，图片中所有定位出的文字框
输出内容:
1. 检测绿色矩形框
2. 定位图片上的刻度点，确定刻度点在x轴/y轴，将刻度点与文字内容对应匹配连接起来

#### 运行环境:
>pytor1.0 应该都没问题
其它包缺啥装啥

#### 文件说明:
train.py 训练代码
demo.py, detect.py 都是测试代码，都一样. 推荐使用demo.py

#### 数据集:
- 2020_UB_PMC

- 2020_One_Drive
需要修改一下train.txt test.txt, 文件名

#### 比赛提交的结果使用的模型:
合成数据集上:
debug0_onedrive_data(利用划分的训练集和测试集)

指标结果:
    batch_size 16,  res18, [256,128,128,64,32]
ave recall: 99.70; ave prec: 99.93; ave F: 99.82
ave recall: 99.73; ave prec: 99.93; ave F: 99.83
            99.66;           99.83;        99.74

后处理: prepare_output2.py

---

真实数据集上:
debug0_real_data:
模型找不到了... 因为后期直接训练全部数据的模型，好像把这个文件夹的模型删了

thresh_48_0.4 ave recall:78.70; ave prec: 83.55; ave F: 81.05
                     79.01;           83.72;        81.30(修改tick radius 和try time)

后处理: prepare_output4.py


#### 评价代码
OneDrive文件夹对应自己划分的1067张图片
real文件夹对应自己划分的297张图片


#### 目前想到的对比实验:
- 该模型结构，先前的模型基于down-up 的结构，上采样的时候没有考虑结合一下前面的层，应该会有提升
- 改loss function，实验中使用的loss 损失为MSELoss, 可以参考一下这篇论文中的loss试试 [SelfText Beyond Polygon: Unconstrained Text Detection with Box Supervision and Dynamic Self-Training](https://arxiv.org/abs/2011.13307v2)
Text skeleton loss (modify the cross-entropy loss into a “soft” form.)
- ...