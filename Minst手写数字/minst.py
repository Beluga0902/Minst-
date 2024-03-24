from keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape, test_images.shape)
# print(train_images[0])
# print(train_labels[0])
# plt.imshow(train_images[0])
# plt.show()

# 将数据铺开成一维向量
train_images = train_images.reshape((60000, 28*28)).astype('float')
test_images = test_images.reshape((10000, 28*28)).astype('float')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# to_categorical one hot编码 1 [0 1 0 0 0 0 0...] 把矩阵转化为向量
# print(train_labels[0])测试一下向量
# 以上步骤为数据集的处理keras框架

# 训练集
network = models.Sequential()
# 隐藏层有三条神经元，分别为128，64，32，不使用sigmoid和tation，因为梯度弥散，曲线是s形，数值很大时趋近平缓，梯度会消失，正则化kernel_regularizer=regularizers.l1
network.add(layers.Dense(units=128, activation='relu', input_shape=(28*28, ), kernel_regularizer=regularizers.l1(0.0001)))
# Dropout有0.01的概率让神经元丧失功能
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
# 输出层有十个神经元，每个样本属于0到9的概率
network.add(layers.Dense(units=10, activation='softmax'))

# 编译步骤
network.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
network.fit(train_images, train_labels, epochs=20, batch_size=128, verbose=2)

# print(network.summary())

# 测试集
# y_pre = network.predict(test_images[:5])
# print(y_pre, test_labels[:5])
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
