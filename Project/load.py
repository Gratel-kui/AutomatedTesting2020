import os
import numpy as np
import pickle

def load_CIFAR_batch(filename):
    with open(filename,'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        images = dict[b'data']
        labels = dict[b'labels']
        images = images.reshape(10000, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)
        labels = np.array(labels)
    return images, labels


def load_CIFAR_data(data_dir):
    '''load CIFAR data'''
    images_train = []
    labels_train = []
    for i in range(5):
        f = os.path.join(data_dir, 'data_batch_%d' % (i+1))
        print('loading', f)
        # 调用load_CIFAR_batch()获得批量的图像及其对应的标签
        image_batch, label_batch = load_CIFAR_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        Xtrain = np.concatenate(images_train)
        Ytrain = np.concatenate(labels_train)
        del image_batch, label_batch
    Xtest, Ytest = load_CIFAR_batch(os.path.join(data_dir, 'test_batch'))
    print('finished loadding CIFAR-10 data')

    # 返回训练集的图像和标签，测试集的图像和标签
    return Xtrain, Ytrain, Xtest, Ytest


def load_CIFAR100_test(filename):
    with open(filename,'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        images = dict[b'data']
        labels = dict[b'fine_labels']+dict[b'coarse_labels']
        images = images.reshape(10000, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)
        labels = np.array(labels)
    return images, labels


def load_CIFAR100_data(data_dir):
    '''load CIFAR data'''
    images_train = []
    labels_train = []
    f = os.path.join(data_dir, 'train')
    print('loading', f)
    with open(f,'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        print(dict.keys())
        images = dict[b'data']
        labels = dict[b'fine_labels']+dict[b'coarse_labels']
        images = images.reshape(50000, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)
        labels = np.array(labels)
    Xtrain, Ytrain = images, labels

    Xtest, Ytest = load_CIFAR100_test(os.path.join(data_dir, 'test'))
    print('finished loadding CIFAR-100 data')
    # 返回训练集的图像和标签，测试集的图像和标签
    return Xtrain, Ytrain, Xtest, Ytest


def result_data(data_dir):
    '''load resulted data'''
    dict_res = np.load(data_dir)
    Xres, Yres = dict_res['data'], dict_res['labels']
    return Xres, Yres




