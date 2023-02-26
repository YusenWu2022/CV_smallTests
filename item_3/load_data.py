#代码中残留了一些numpy的操作，取消注释后可以实现npy格式的存储再从中读取成tensor数据利用
#由于机子上的vscode不能识别相对路径，于是使用了绝对路径qwq
#使用了RGB通道的Image库配合tqdm库对图片进行了读取，也可以使用cv2
#random.shuffle使得图片的batch是随机组合的
#学会了transform和transform_back的基本操作

import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from cv2 import cv2 as cv2
from tqdm import tqdm
import glob
import random
''' 这里可以 import 更多库 '''


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        dir_path = 'C:\\Users\\冰山战役\\Desktop\\dataloader_project (2)\\data'#在这里修改读取路径
        testScale = -1
        i = 0
        label2idx = {}
        data = []
        for (root, dirs, files) in os.walk(dir_path):
            for Ufile in tqdm(files):
                # Ufile是文件名
                img_path = os.path.join(root, Ufile)  # 文件的所在路径
                File = root.split('/')[-1]  # 文件所在文件夹的名字, 也就是label
                # 读取image和label数据
                # img_data = cv2.imread(img_path)
                img_data = Image.open(img_path)
                # data_res = cv2.resize(img_data, (256, 256),
                # interpolation = cv2.INTER_CUBIC)
                data_res = img_data.resize((256, 256), Image.ANTIALIAS)

                if File not in label2idx:
                    label2idx[File] = i
                    i = i + 1
                    # 返回的是字典类型
                label2idx, i = label2idx, i
                label = label2idx[File]
                # 存储image和label数据
                data.append([np.array(data_res), label])
        random.shuffle(data)  # 随机打乱,直接打乱data
    # 训练集和测试集的划分
        testNum = int(len(data)*testScale)
        train_data = data[:-1*testNum]  # 训练集
        # test_data = data[-1*testNum:]  # 测试集
    # 测试集的输入输出和训练集的输入输出
        X_train = np.array([i[0] for i in train_data])  # 训练集特征
        Y_train = np.array([i[1] for i in train_data])  # 训练集标签
        # X_test = np.array([i[0] for i in test_data])  # 测试集特征
        # y_test = np.array([i[1] for i in test_data])  # 测试集标签
        #print(len(X_train), len(y_train), len(X_test), len(y_test))
        '''可以在这里修改路径
        np.save('D:\\python\\pythontest\\data_npy\\train-images-idx3.npy', X_train)
        np.save('D:\\python\\pythontest\\data_npy\\train-labels-idx1.npy', y_train)
        np.save('D:\\python\\pythontest\\data_npy\\t10k-images-idx3.npy', X_test)
        np.save('D:\\python\\pythontest\\data_npy\\t10k-labels-idx1.npy', y_test)
    '''
    # 保存文件
    '''
        np.load('D:\\python\\pythontest\\data_npy\\train-images-idx3.npy',X_train)
        np.load('D:\\python\\pythontest\\data_npy\\train-images-idx1.npy',Y_train)
        np.load('D:\\python\\pythontest\\data_npy\\t10k-images-idx3.npy',X_test)
        np.load('D:\\python\\pythontest\\data_npy\\t10k-images-idx1.npy',Y_test)
'''
#上面实现了自制数据集的保存和读入使用
        self.img_list = X_train
        self.labels = Y_train
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                                            )  # 必须根据normallize_back函数参数更改所用的方差和均值！！！不然会变色！
        ''' 填空内容 '''

    def __getitem__(self, index):
        img = self.img_list[index]

        img = self.transform(img)

        label = self.labels[index]
        return img, label  # transform module has not been considered
        ''' 填空内容 '''

    def __len__(self):
        return len(self.img_list)


def normalize_back(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img*std+mean
    img = img.clamp(0, 1)
    return img


dataset = MyDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

for iteration, image in enumerate(dataloader):

    data, lable = image

    torchvision.utils.save_image(normalize_back(data),
                                 os.path.join(
                                     'C:\\Users\\冰山战役\\Desktop\\dataloader_project (2)\\result', f'{iteration}.jpg'),#在这里修改图片输出路径
                                 normalize=False)

# 'list' object has no attribute 'data'错误。
# cv2.imread 进来的图片数据是rgb，必须转换成tensor的bgr格式，不然会变色
