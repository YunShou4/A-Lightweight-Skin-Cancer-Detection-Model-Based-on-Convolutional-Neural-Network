import matplotlib.pyplot as plt
from dataAnalysing import dataLoad
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation,Concatenate,GlobalAveragePooling2D,Input
import random
from keras import optimizers



def buildModel():
    model = Sequential([
        Conv2D(32, 3, padding = "same", activation = "relu", input_shape = (28, 28, 3)),
        Conv2D(32, 3, padding="same", activation="relu"),
        Conv2D(32, 3, padding="same", activation="relu"),
        Conv2D(32, 3, padding="same", activation="relu"),


        Conv2D(64, 3, padding = "same", activation = "relu"),
        Conv2D(64, 3, padding="same", activation="relu"),
        Conv2D(64, 3, padding="same", activation="relu"),
        Conv2D(64, 3, padding="same", activation="relu"),

        Conv2D(128, 3, padding = "same", activation = "relu"),
        Conv2D(128, 3, padding="same", activation="relu"),
        Conv2D(128, 3, padding="same", activation="relu"),
        Conv2D(128, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(256, 3, padding="same", activation="relu"),
        Conv2D(256, 3, padding="same", activation="relu"),
        Conv2D(256, 3, padding="same", activation="relu"),
        Conv2D(256, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(512, 3, padding="same", activation="relu"),


        Flatten(),
        Dense(1024, activation = "relu"),
        Dense(8, activation = "softmax")
    ])

    return model

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'
    x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = Concatenate(axis=3)([left, right])


    return x


def vgg16model():
    model1 = Sequential([
        #两个卷积层
        Conv2D(64, 3, padding="same", activation="relu", input_shape=(28, 28, 3)),
        Conv2D(64, 3, padding="same", activation="relu"),

        #两个卷积层
        Conv2D(128, 3, padding="same", activation="relu"),
        Conv2D(128, 3, padding="same", activation="relu"),

        #三层卷积层
        Conv2D(256, 3, padding="same", activation="relu"),
        Conv2D(256, 3, padding="same", activation="relu"),
        Conv2D(256, 3, padding="same", activation="relu"),

        #三层卷积层+最大池化
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(512, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        # 三层卷积层+最大池化
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(512, 3, padding="same", activation="relu"),
        Conv2D(512, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(8, activation="softmax")
    ])

    return model1



def vgg16model_half():
    model1 = Sequential([
        #两个卷积层
        Conv2D(64, 3, padding="same", activation="relu", input_shape=(28, 28, 3)),
        Conv2D(64, 3, padding="same", activation="relu"),

        #两个卷积层
        Conv2D(128, 3, padding="same", activation="relu"),
        Conv2D(128, 3, padding="same", activation="relu"),


        MaxPooling2D(),
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(8, activation="softmax")



    ])
    return model1

def vgg16_half_fire():
    inputs = Input(shape=(28, 28, 3))
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = fire_module(x, fire_id=1, squeeze=8, expand=32)
    x = fire_module(x, fire_id=2, squeeze=8, expand=32)
    x = fire_module(x, fire_id=4, squeeze=16, expand=64)
    x = fire_module(x, fire_id=5, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = Dropout(0.5, name='drop5')(x)
    x = Conv2D(8, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(inputs, out, name='vgg16_half_fire')
    return model

def squeeze_net():
    inputs = Input(shape=(28, 28, 3))
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)


    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D()(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    x = Conv2D(8, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(inputs, out, name='squeezenet')

    return model



def plotInfo(history,name):
    plt.plot(history.history["loss"], label = "loss")
    plt.plot(history.history["val_loss"], label = "val_loss")
    plt.plot(history.history["accuracy"], label = "accuracy")
    plt.plot(history.history["val_accuracy"], label = "val_accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Loss and Acc")
    plt.legend()
    plt.grid(True)
    plt.savefig(name+'.png')
    plt.show()



if __name__ == "__main__":

    X = dataLoad("./data/data_expend.npy")
    y = dataLoad("./data/label_expend.npy")
    print(X.shape)
    print(y.shape)
    #shuffle
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]

    model = vgg16_half_fire()
    # compile model
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # print model info
    model.summary()
    # training
    history = model.fit(
        X, y,
        batch_size=128,
        epochs=15,
        validation_split=0.1,
        shuffle=True
    )

    # save model
    model.save("./vgg16_half_fire_e")

    # visualize
    plotInfo(history, "vgg16_half_fire_e")

    model = buildModel()
    # compile model
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # print model info
    model.summary()
    # training
    history = model.fit(
        X, y,
        batch_size=128,
        epochs=15,
        validation_split=0.1,
        shuffle=True
    )

    # save model
    model.save("./buildmodel_e")

    # visualize
    plotInfo(history, "buildmodel_e")

    # create model
    model = vgg16model()
    # compile model
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer = "adam", loss = "CategoricalCrossentropy", metrics = ["accuracy"])
    # print model info
    model.summary()
    # training
    history = model.fit(
        X, y,
        batch_size = 128,
        epochs = 15,
        validation_split = 0.1,
        shuffle = True
    )

    # save model
    model.save("./vgg16model_e")

    # visualize
    plotInfo(history, "vgg16_e")

    model = vgg16model_half()
    # compile model
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # print model info
    model.summary()
    # training
    history = model.fit(
        X, y,
        batch_size=128,
        epochs=15,
        validation_split=0.1,
        shuffle=True
    )

    # save model
    model.save("./vgg16model_half_e")

    # visualize
    plotInfo(history, "vgg16_half_e")

    model = squeeze_net()
    # compile model
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # print model info
    model.summary()
    # training
    history = model.fit(
        X, y,
        batch_size=128,
        epochs=15,
        validation_split=0.1,
        shuffle=True
    )

    # save model
    model.save("./squeeze_net_e")

    # visualize
    plotInfo(history, "squeeze_net_e")

