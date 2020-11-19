from keras.preprocessing.image import ImageDataGenerator


def nothing(images, labels, length):
    datagen_flip = ImageDataGenerator()
    #iter = datagen_flip.flow(x=images, y=labels, batch_size=10, shuffle=False,save_to_dir='../Data/oringin_data_png_cifar100')
    #Xtmp, Ytmp = iter.next()
    iter = datagen_flip.flow(x=images, y=labels, batch_size=length, shuffle=False )
    i = 0
    while i < 10:
        x_res, y_res = iter.next()
        i += 1
    print(x_res.shape)
    return x_res, y_res


# 旋转操作
def rotate(images, labels, length):
    print('rotate:')
    datagen_flip = ImageDataGenerator(
        rotation_range=90,
        #horizontal_flip=True,
        #vertical_flip=True,
        fill_mode='nearest'
    )
    #iter = datagen_flip.flow(x=images, y=labels, batch_size=10, shuffle=False, save_to_dir='../Data/operation_data_png_cifar100/rotate')
    #Xtmp, Ytmp = iter.next()
    iter = datagen_flip.flow(x=images, y=labels, batch_size=length,shuffle=False)
    i = 0
    while i<10:
        x_res, y_res = iter.next()
        i += 1
    print(x_res.shape)
    return x_res, y_res


def shift(images, labels, length):
    print('shift:')
    datagen_flip = ImageDataGenerator(
        width_shift_range=0.2,
        #height_shift_range=0.2,
        fill_mode='nearest'
    )
    #iter = datagen_flip.flow(x=images, y=labels, batch_size=10, shuffle=False,save_to_dir='../Data/operation_data_png_cifar100/shift')
    #Xtmp, Ytmp = iter.next()
    iter = datagen_flip.flow(x=images, y=labels, batch_size=length,shuffle=False)
    i = 0
    while i<10:
        x_res, y_res = iter.next()
        i += 1
    print(x_res.shape)
    return x_res, y_res


def zca(images, labels, length):
    print('zca:')
    datagen_zca = ImageDataGenerator(
        zca_whitening=True
    )
    datagen_zca.fit(images)
    #iter = datagen_zca.flow(x=images, y=labels, batch_size=10, shuffle=False,save_to_dir='../Data/operation_data_png_cifar100/zca')
    #Xtmp, Ytmp = iter.next()
    iter = datagen_zca.flow(x=images, y=labels, batch_size=length,shuffle=False)
    i = 0
    while i<10:
        x_res, y_res = iter.next()
        i += 1
    print(x_res.shape)
    return x_res, y_res


def flip(images, labels, length):
    print('flip:')
    datagen_zca = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    #iter = datagen_zca.flow(x=images, y=labels, batch_size=10, shuffle=False,save_to_dir='../Data/operation_data_png_cifar100/flip')
    #Xtmp, Ytmp = iter.next()
    iter = datagen_zca.flow(x=images, y=labels, batch_size=length,shuffle=False)
    i = 0
    while i<10:
        x_res, y_res = iter.next()
        i += 1
    print(x_res.shape)
    return x_res, y_res


def feature(images, labels, length):
    print('feature:')
    datagen_feature = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    datagen_feature.fit(images)
    #iter = datagen_feature.flow(x=images, y=labels, batch_size=10, shuffle=False,save_to_dir='../Data/operation_data_png_cifar100/feature')
    #Xtmp, Ytmp = iter.next()
    iter = datagen_feature.flow(x=images, y=labels, batch_size=length,shuffle=False)
    i = 0
    while i<10:
        x_res, y_res = iter.next()
        i += 1
    print(x_res.shape)
    return x_res, y_res


'''
def sample(images, labels, length):
    print('sample:')
    datagen_sample = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    datagen_sample.fit(images)
    #iter = datagen_sample.flow(x=images, y=labels, batch_size=10, shuffle=False,save_to_dir='../Data/operation_data_png_cifar100/sample')
    #Xtmp, Ytmp = iter.next()
    iter = datagen_sample.flow(x=images, y=labels, batch_size=length,shuffle=False)
    i = 0
    while i<10:
        x_res, y_res = iter.next()
        i += 1
    print(x_res.shape)
    return x_res, y_res
'''