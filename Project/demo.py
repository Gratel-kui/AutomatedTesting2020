import methods
import load
import eval
import visialize
import numpy as np
import tensorflow as tf
import keras as ks



data_dir_100_rotate = '../Data/cifar100_result_data/rotate.npz'
data_dir_10 = "../Data/cifar-10-batches-py/"
data_dir_100 = "../Data/cifar-100-python/"
'''
print("Start loading.")
Xtrain, Ytrain, Xtest, Ytest = load.load_CIFAR_data(data_dir_10)
#Xres, Yres = methods.nothing(Xtrain,Ytrain,50000)
scores = eval.eval_10(Xtrain[:1000],Ytrain[:1000])
visialize.add("oringin",scores)
#np.savez('../Data/cifar10_result_data/rotate.npz',data=Xres,labels=Yres,allow_pickle=True)


# 第一种方法：rotate
#Xres, Yres = methods.rotate(Xtrain[:50000],Ytrain[:50000],50000)
#eval.eval_10(Xres,Yres)
#np.savez('../Data/cifar10_result_data/rotate.npz',data=Xres,labels=Yres,allow_pickle=True)
print('Done saving.')
Xres, Yres = load.result_data('../Data/cifar10_result_data/rotate.npz')
scores = eval.eval_10(Xres[:1000],Yres[:1000])
visialize.add("rotate",scores)

# 第二种方法：shift( including horizon and vertical)
#Xres, Yres = methods.shift(Xtrain[:50000],Ytrain[:50000],50000)
#np.savez('../Data/cifar10_result_data/shift.npz',data=Xres,labels=Yres,allow_pickle=True)
Xres, Yres = load.result_data('../Data/cifar10_result_data/shift.npz')
scores = eval.eval_10(Xres[:1000],Yres[:1000])
visialize.add("shift",scores)


# 第三种方法：zca白化
#Xres, Yres = methods.zca(Xtrain[:50000],Ytrain[:50000],50000)
#eval.eval_10(Xres,Yres)
#np.savez('../Data/cifar10_result_data/zca.npz',data=Xres,labels=Yres,allow_pickle=True)
#print('Done saving.')
Xres, Yres = load.result_data('../Data/cifar10_result_data/zca.npz')
scores = eval.eval_10(Xres[:1000],Yres[:1000])
visialize.add("zca",scores)


# 第四种方法：翻转
#Xres, Yres = methods.flip(Xtrain[:50000],Ytrain[:50000],50000)
#eval.eval_10(Xres,Yres)
#np.savez('../Data/cifar10_result_data/flip.npz',data=Xres,labels=Yres,allow_pickle=True)
#print('Done saving.')
Xres, Yres = load.result_data('../Data/cifar10_result_data/flip.npz')
scores = eval.eval_10(Xres[:1000],Yres[:1000])
visialize.add("flip",scores)


# 第五种方法： feature 标准化
#Xres, Yres = methods.feature(Xtrain[:50000],Ytrain[:50000],50000)
#eval.eval_10(Xres,Yres)
#np.savez('../Data/cifar10_result_data/feature.npz',data=Xres,labels=Yres,allow_pickle=True)
#print('Done saving.')
Xres, Yres = load.result_data('../Data/cifar10_result_data/feature.npz')
scores = eval.eval_10(Xres[:1000],Yres[:1000])
visialize.add("feature",scores)

'''
# dealing cifar100
Xtrain, Ytrain, Xtest, Ytest = load.load_CIFAR100_data(data_dir_100)

#print("nothing:")
Xres, Yres = methods.nothing(Xtrain[:1000],Ytrain[:1000],1000)
eval.eval_100(Xres,Yres)
#scores = eval.eval_100(Xtest[:1000],Ytest[:1000])
#visialize.add("oringin",scores)
#eval.liandan(Xtrain[:10000],Ytrain[:10000],Xtest[:10000],Ytest[:10000])
#eval.eval_my_model(Xtrain[:1000],Ytrain[:1000])


# 第一种方法：rotate
#Xres, Yres = methods.rotate(Xtrain[:50000],Ytrain[:50000],50000)
#eval.liandan(Xres,Yres)
#eval.eval_100(Xres,Yres)
#np.savez('../Data/cifar100_result_data/rotate.npz',data=Xres,labels=Yres,allow_pickle=True)
#print('Done saing.')
#Xres, Yres = load.result_data(data_dir_100_rotate)
#eval.liandan(Xtrain[:1000], Ytrain[:1000])
#eval.eval_100(Xres[:100],Yres[:100])
#Xres, Yres = load.result_data('../Data/cifar100_result_data/rotate.npz')
#scores = eval.eval_100(Xres[:1000],Yres[:1000])
#visialize.add("rotate",scores)


# 第二种方法：shift( including horizon and vertical)
#Xres, Yres = methods.shift(Xtrain[:50000],Ytrain[:50000],50000)
#eval.eval_100(Xres,Yres)
#np.savez('../Data/cifar100_result_data/shift.npz',data=Xres,labels=Yres,allow_pickle=True)
#print('Done saving.')
#Xres, Yres = load.result_data('../Data/cifar100_result_data/shift.npz')
#scores = eval.eval_100(Xres[:1000],Yres[:1000])
#visialize.add("shift",scores)

# 第三种方法：zca白化
#Xres, Yres = methods.zca(Xtrain[:50000],Ytrain[:50000],50000)
#eval.eval_100(Xres,Yres)
#np.savez('../Data/cifar100_result_data/zca.npz',data=Xres,labels=Yres,allow_pickle=True)
#print('Done saving.')
#Xres, Yres = load.result_data('../Data/cifar100_result_data/zca.npz')
#scores = eval.eval_100(Xres[:1000],Yres[:1000])
#visialize.add("zca",scores)

# 第四种方法：翻转
#Xres, Yres = methods.flip(Xtrain[:50000],Ytrain[:50000],50000)
#eval.eval_100(Xres,Yres)
#np.savez('../Data/cifar100_result_data/flip.npz',data=Xres,labels=Yres,allow_pickle=True)
#print('Done saving.')
#Xres, Yres = load.result_data('../Data/cifar100_result_data/flip.npz')
#scores = eval.eval_100(Xres[:1000],Yres[:1000])
#isialize.add("flip",scores)

# 第五种方法： feature 标准化
#Xres, Yres = methods.feature(Xtrain[:50000],Ytrain[:50000],50000)
#eval.eval_100(Xres,Yres)
#np.savez('../Data/cifar100_result_data/feature.npz',data=Xres,labels=Yres,allow_pickle=True)
#print('Done saving.')
#Xres, Yres = load.result_data('../Data/cifar100_result_data/feature.npz')
#scores = eval.eval_100(Xres[:1000],Yres[:1000])
#visialize.add("feature",scores)

# 第六种方法： sample 标准化
#Xres, Yres = methods.sample(Xtrain[:1000],Ytrain[:1000],1000)
#eval.eval_100(Xres,Yres)



#visialize.save()
