
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
ALGORITHM = "tf_net"
#ALGORITHM = "tf_conv"

#DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = IH * IW * IZ
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = IH * IW * IZ
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3
    IS = IH * IW * IZ


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

#(x, y) = (train data, train output)
def buildTFNeuralNet(x, y, eps = 6):
    inputs = tf.keras.Input(shape=x.shape[1],)
    layer1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(inputs)
    layer2 = tf.keras.layers.Dense(512, activation=tf.nn.sigmoid)(layer1)
    outputs = tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax)(layer2)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
    model.fit(x=x, y=y, epochs=20)
    
    return model


def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.2):
    
    data_shape = (IH, IW, IZ)
    model = tf.keras.Sequential()

    if (DATASET == "mnist_d"):

        #convnet
        model.add(tf.keras.layers.Conv2D(45, kernel_size=(6, 6), activation="relu", strides=(3, 3), input_shape=data_shape))
        model.add(tf.keras.layers.Conv2D(90, kernel_size=(3, 3), activation="relu", strides=(1, 1)))
        #model.add(tf.keras.layers.MaxPooling2D(pool_size=(26, 26)))
        #model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="sigmoid"))

        model.add(tf.keras.layers.Flatten())
        #neural net
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.sigmoid))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax))

        model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
        model.fit(x=x, y=y, epochs=15)

    if (DATASET == "mnist_f"):
        #convnet
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(9, 9), activation="relu", strides=(1, 1), padding='same', input_shape=data_shape))
        #model.add(tf.keras.layers.Conv2D(128, kernel_size=(6, 6), activation="relu", strides=(1, 1), padding='same'))
        #model.add(tf.keras.layers.Conv2D(64, kernel_size=(9, 9), activation="relu", strides=(1, 1), input_shape=data_shape))
        #model.add(tf.keras.layers.Conv2D(120, kernel_size=(6, 6), activation="relu", strides=(1, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", strides=(1, 1)))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", strides=(3, 3)))
        
        model.add(tf.keras.layers.Flatten())
        #neural net
        #if (dropout):
            #model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(900, activation=tf.nn.relu))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(900, activation=tf.nn.sigmoid))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax))

        model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
        model.fit(x=x, y=y, epochs=15)

    if (DATASET == "cifar_10"):
        #convnet
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same', input_shape=data_shape))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        #neural net
        model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax))

        model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
        model.fit(x=x, y=y, epochs=15)

    if (DATASET == "cifar_100_c"):
        #convnet
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same', input_shape=data_shape))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1)))
        #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        #model.add(tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1)))
        #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(tf.keras.layers.Flatten())
        #neural net
        model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(4096, activation=tf.nn.sigmoid))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax))

        model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
        model.fit(x=x, y=y, epochs=15)

    if (DATASET == "cifar_100_f"):
        #convnet
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same', input_shape=data_shape))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', dilation_rate=(1, 1), padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same'))
        #model.add(tf.keras.layers.Conv2D(128, kernel_size=(2, 2), activation='relu', dilation_rate=(1, 1), padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same'))
        #model.add(tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', dilation_rate=(1, 1), padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same'))
        model.add(tf.keras.layers.Conv2D(1024, kernel_size=(2, 2), activation='relu', dilation_rate=(1, 1), padding='same'))
        #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        #model.add(tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1), padding='same'))
        #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        #model.add(tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu', dilation_rate=(1, 1)))
        #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(tf.keras.layers.Flatten())
        #neural net
        model.add(tf.keras.layers.Dense(4096, activation=tf.nn.relu))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(4096, activation=tf.nn.sigmoid))
        if (dropout):
            model.add(tf.keras.layers.Dropout(dropRate))
        model.add(tf.keras.layers.Dense(y.shape[1], activation=tf.nn.softmax))

        model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
        model.fit(x=x, y=y, epochs=20)


    return model

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data()
    elif DATASET == "cifar_100_f":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode='fine')
    elif DATASET == "cifar_100_c":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode='coarse')
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def parse_args():
    global DATASET
    global ALGORITHM
    params = sys.argv[:]
    params.pop(0)
    while len(params) > 0:
        opt = params.pop(0)
        if (opt == '--dataset'):
            DATASET = params.pop(0)
        elif (opt == '--type'):
            alg = params.pop(0)
            if (alg == 'net'):
                ALGORITHM = 'tf_net'
            elif (alg == 'conv'):
                ALGORITHM = 'tf_conv'
            elif (alg == 'guesser'):
                ALGORITHM = 'guesser'
        elif (opt == '--help'):
            print("options:")
            print("--dataset [mnist_d, mnist_f, cifar_10, cifar_100_f, cifar_100_c]")
            print("--type [net, conv, guesser]")
            print("UNIMPLEMENTED: --output-weights [file_name]")
            quit()
        pass
    pass


def main():
    parse_args()
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
