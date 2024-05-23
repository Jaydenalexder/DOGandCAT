# utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras.layers
import os

from tensorflow import keras
import sys
import cv2


print(tf.__version__)
print(sys.version_info)

for module in mpl, pd, np, keras:
    print(module.__name__, module.__version__)

#路径
train_dir = './train'
valid_dir = './validation'
test_dir = './test'

#参数
height = 128
width = 128
channel = 3
batch_size = 64
valid_batch_size = 64
num_classes = 5
epochs = 100


# 训练模型
'''
该函数用于训练模型
参数model为要训练的模型
参数train_generator和valid_generator是数据输入生成器 用于生成训练集和验证集数据
参数callbacks是回调函数 用于在训练中执行特定操作
'''

def trainModel(model, train_generator, valid_generator,callbacks):
    # 使用 fit 函数对模型进行训练   # train_generator是用于训练的生成器
# epochs为训练次数  # validation_data为用于验证的生成器  # callbacks为回调函数
    history = model.fit (
        train_generator,
        epochs=epochs,
        validation_data = valid_generator,
        callbacks = callbacks
    )
    return history



# 训练过程中损失和准确度的变化
def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


# 使用模型完成训练和分类 把图片分类到两个不同的文件夹
def predictModel(model, output_model_file):

    # 加载模型的权重
    model.load_weights(output_model_file)
    #创建文件夹
    os.makedirs('./save', exist_ok=True)
    os.makedirs('./save/cat', exist_ok=True)
    os.makedirs('./save/dog', exist_ok=True)
    #要预测的图像的文件夹
    test_dir = './test/'  # 1-12500.jpg
    for i in range(1, 12500):
        #获取图像并resize
        img_name = test_dir + '{}.jpg'.format(i)
        img = cv2.imread(img_name)
        img = cv2.resize(img, (width, height))
        img_arr = img / 255.0
        img_arr = img_arr.reshape((1, width, height, 3))
        # 对图像进行预测并保存
        pre = model.predict(img_arr)
        if pre[0][0] > pre[0][1]:
            cv2.imwrite('./save/cat/' + '{}.jpg'.format(i), img)
            print(img_name, ' is classified as Cat.')
        else:
            cv2.imwrite('./save/dog/' + '{}.jpg'.format(i), img)
            print(img_name, ' is classified as Dog.')


if __name__ ==  '__main__':

    print('开始导入数据…')

    # 导入数据并做一个扩充
    
    '''
    创建一个 ImageDataGenerator 对象用于生成图像数据的增强版
    ImageDataGenerator 是 Keras 的一个图像处理工具，
    用于在训练期间进行数据增强，以增加模型的稳健性和泛化能力
    '''
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255, #表示将图像像素值重新缩放到 [0,1] 范围内，以便更好地进行训练。
        rotation_range=40, #表示随机旋转图像的范围为 ±40°。
        width_shift_range=0.2, #表示随机水平和垂直移动图像的范围为 0.2。
        height_shift_range=0.2, #表示随机剪切图像的范围为0.2。
        shear_range=0.2,#表示随机剪切图像的强度为 0.2
        zoom_range=0.2, #表示随机缩放图像的范围为 [0.8,1.2]。
        horizontal_flip=True,#表示随机水平反转图像
        fill_mode='nearest',#表示填充新像素时使用最近的颜色值。
    )
    
    #从路径为 train_dir 的文件夹中实时生成增强版的图像数据。
    train_generator = train_datagen.flow_from_directory(
        train_dir, #路径
        target_size=(width, height), #将图像调整为指定的大小 (width, height)
        batch_size=batch_size, #每次从文件夹中取出的样本数
        seed=7, #随机种子
        shuffle=True, #每个 epoch 后打乱数据集的顺序
        class_mode='categorical' #one-hot 编码的标签
    )
    #图像数据生成器
    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,#将图像像素值缩小到0-1范围内，即归一化到0-1范围内
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,  #图像路径
        target_size=(width, height), #图像尺寸
        batch_size=valid_batch_size, #batch_size
        seed=7, #随机种子 
        shuffle=False, #不随机打乱数据
        class_mode="categorical" #one-hotel分类标签
    )


    train_num = train_generator.samples #samples 属性可以获取它们所生成的图像样本的数量
    valid_num = valid_generator.samples
    # print(train_num, valid_num)

    print('正在构建模型...')

    # 建立模型
    #网络模型搭建 Conv-Conv-pooling-Conv-Conv-pooling-Conv-Conv-pooling-Flatten-ReLU-FN
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3,
                            padding='same', activation='relu',
                            input_shape=[width, height, channel]),
        keras.layers.Conv2D(filters=32, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),

        keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),

        keras.layers.Conv2D(filters=128, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.Conv2D(filters=128, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),

        keras.layers.Flatten(),#打平为一维数组 方便全连接 
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax') #FN分类
    ])
    
    #categorical_crossentropy作为损失函数，其适用于多类分类问题
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy']) #Adam优化器(自适应梯度下降)
    model.summary()

    # 设置保存模型的路径
    logdir = './graph_def_and_weights'
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    output_model_file = os.path.join(logdir,
                                     "catDog_weights.h5")
    print('Start training ...')
    
    
    # 开始训练
    mode = input('选择模型: 1.训练\n  2.预测(只有当已经完成模型的训练并保存它时，才运行此模式)\n请输入: ')
    if mode == '1':
        #回调函数
        callbacks = [
            keras.callbacks.TensorBoard(logdir),
            keras.callbacks.ModelCheckpoint(output_model_file,
                                            save_best_only=True,
                                            save_weights_only=True),
            keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
        ]
        #训练
        history = trainModel(model, train_generator, valid_generator, callbacks)
        #记录训练历史记录，可视化准确率和损失 画出曲线变化曲线
        plot_learning_curves(history, 'accuracy', epochs, 0, 1)
        plot_learning_curves(history, 'loss', epochs, 0, 5)
    elif mode == '2':
        #  只有当你已经完成模型的训练并保存它时，才运行此模式
        predictModel(model, output_model_file)
    else:
        print('请输入正确的数字!!!')

    print('完成! 退出!.')
