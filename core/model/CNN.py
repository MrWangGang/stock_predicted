from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Conv2d 会在图像像素矩阵上滑动，3X3的卷积核代表模型的感受视野,在卷积核内的矩阵会进行点积运算得到新的特征图
        #padding 填充0
        #kernel_size 卷积核大小
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        #激活函数通常会将输入信号转换为输出信号。#引入非线性变换
        # 它可以看作是一个“门”，决定了当前神经元的输出是否要传递给下一个神经元。
        #例如，ReLU 函数会将负值置为零，只有正值才会被传递。
        self.relu1 = nn.ReLU()
        #MaxPool2d 不会进行点积运算，他会选取卷积核里最大值
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #kernel_size卷积核大小 stride是滑动步长
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)#定义输入数据是 28*28的图像，经过2次maxpool 会变成 28/2/2*28/2/2 = 7*7
        self.relu_fc1 = nn.ReLU() # 第一个全连接层后的 ReLU 激活
        self.sigmoid_fc1 = nn.Sigmoid() # 在 ReLU 之后额外添加 Sigmoid 激活层
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10) # 输出层
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7) # 展平，-1代表自动计算批次，批次指的是每次模型训练的数据量，全量数据送入模型,算力不够
        x = self.relu_fc1(self.fc1(x)) # 经过 ReLU 激活
        x = self.sigmoid_fc1(x)       # 紧接着经过 Sigmoid 激活
        x = self.dropout(x)
        x = self.fc2(x)
        return x