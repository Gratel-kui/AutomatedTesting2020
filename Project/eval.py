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


def liandan(train_images, train_labels, test_images, test_labels):
    from keras.utils import to_categorical

    checkpoint_path = 'cp-{epoch:04d}.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5)


    new_model = tf.keras.models.load_model('../model/cifar100/CNN_with_dropout.h5')
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    new_model.summary()
    new_model.fit(train_images,
                  train_labels,
                  epochs=100,
                  batch_size=200)
    new_model.save('../model/my_model.h5')


def eval_my_model(Xtest, Ytest):
    model = keras.models.load_model('../model/my_model.h5')
    y_pred = model.predict(Xtest)
    count = 0
    for i in range(len(y_pred)):
        # print(y_pred[i],np.argmax(y_pred[i]),Ytest[i+10000],)
        if (np.argmax(y_pred[i]) == Ytest[i]):  # argmax函数找到最大值的索引，即为其类别
            count += 1
    score = count / len(y_pred)
    print("My_model:", '正确率为:%.2f%s' % (score * 100, '%'))
