import tensorflow as tf


model_names10 = ["CNN_with_dropout.h5",'CNN_without_dropout.h5','lenet5_with_dropout.h5','lenet5_without_dropout.h5']
model_names100 = ["CNN_with_dropout.h5",'CNN_without_dropout.h5','lenet5_with_dropout.h5','lenet5_without_dropout.h5']


def alchemy10(train_images, train_labels, method):
    for i in range(1):
        checkpoint_path = 'cp-{epoch:04d}.ckpt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=5)
        new_model = tf.keras.models.load_model('../model/cifar10/'+model_names10[i])
        optimizer = tf.keras.optimizers.Adam(learning_rate=8e-5)
        new_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        new_model.summary()
        new_model.fit(train_images,
                      train_labels,
                      epochs=50,
                      batch_size=128)
        new_model.save('../model/my_model_cifar10/'+method+"/"+model_names10[i])


def alchemy100(train_images, train_labels,method):
    checkpoint_path = 'cp-{epoch:04d}.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=5)

    for i in range(len(model_names100)):
        new_model = tf.keras.models.load_model('../model/cifar100/'+model_names100[i])
        optimizer = tf.keras.optimizers.Adam(learning_rate=8e-5)
        new_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        new_model.summary()
        new_model.fit(train_images,
                      train_labels,
                      epochs=50,
                      batch_size=128)
        new_model.save('../model/my_model_cifar100/'+method+"/"+model_names100[i])

