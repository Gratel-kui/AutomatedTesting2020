import os
import numpy as np
import pickle as p
import load
import keras
from tensorflow import keras
import tensorflow as tf



ci_far10 = "../Data/cifar-10-batches-py/data_batch_2"
ci_far100 = "../Data/cifar-100-python/train"
model_names10 = ["CNN_with_dropout.h5",'CNN_without_dropout.h5','lenet5_with_dropout.h5','lenet5_without_dropout.h5','random1_cifar10.h5','random2_cifar10.h5','ResNet_v1.h5','ResNet_v2.h5']
model_names100 = ["CNN_with_dropout.h5",'CNN_without_dropout.h5','lenet5_with_dropout.h5','lenet5_without_dropout.h5','random1_cifar100.h5','random2_cifar100.h5','ResNet_v1.h5','ResNet_v2.h5']
model_dir_10 = "../model/cifar10/"
model_dir_100 = "../model/cifar100/"
methods = ['oringin','composite','zca']

def  eval_10(Xtest,Ytest):
    scores = []
    for x in range(8):
        model = keras.models.load_model(model_dir_10 + model_names10[x])
        y_pred = model.predict(Xtest)
        count = 0
        for i in range(len(y_pred)):
            # print(y_pred[i],np.argmax(y_pred[i]),Ytest[i],)
            if (np.argmax(y_pred[i]) == Ytest[i]):  # argmax函数找到最大值的索引，即为其类别
                count += 1
        score = count / len(y_pred)
        print(model_names10[x], '正确率为:%.2f%s' % (score * 100, '%'))
        scores.append(score)
    return scores


def eval_100(Xtest,Ytest):
    scores = []
    for x in range(8):
        model = keras.models.load_model(model_dir_100 + model_names100[x])
        y_pred = model.predict(Xtest)
        count = 0
        for i in range(len(y_pred)):
            # print(y_pred[i],np.argmax(y_pred[i]),Ytest[i+10000],)
            if (np.argmax(y_pred[i]) == Ytest[i]):  # argmax函数找到最大值的索引，即为其类别
                count += 1
        score = count / len(y_pred)
        print(model_names100[x], '正确率为:%.2f%s' % (score * 100, '%'))
        scores.append(score)
    return scores


def eval_my_model10(Xtest, Ytest):
    for x in range(3):
        for y in range(4):
            model = keras.models.load_model('../model/my_model_cifar10/'+methods[x]+'/'+model_names10[y])
            y_pred = model.predict(Xtest)
            count = 0
            for i in range(len(y_pred)):
                if (np.argmax(y_pred[i]) == Ytest[i]):  # argmax函数找到最大值的索引，即为其类别
                    count += 1
            score = count / len(y_pred)
            print("My_model",methods[x],' ', model_names10[y],'正确率为:%.2f%s' % (score * 100, '%'))


def eval_my_model100(Xtest, Ytest):
    for x in range(3):
        for y in range(4):
            model = keras.models.load_model('../model/my_model_cifar100/'+methods[x]+'/'+model_names100[y])
            y_pred = model.predict(Xtest)
            count = 0
            for i in range(len(y_pred)):
                if (np.argmax(y_pred[i]) == Ytest[i]):  # argmax函数找到最大值的索引，即为其类别
                    count += 1
            score = count / len(y_pred)
            print("My_model",methods[x],' ', model_names100[y],'正确率为:%.2f%s' % (score * 100, '%'))
