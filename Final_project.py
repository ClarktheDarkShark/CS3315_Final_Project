# import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
# from keras import models
from keras.datasets import cifar10
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPool2D, AvgPool2D
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
# import math
from sklearn import tree, metrics
# from sklearn.model_selection import GridSearchCV
# import plotly.graph_objects as go
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import learning_curve
# from sklearn.datasets import load_digits
# from sklearn.svm import SVC
# from sklearn.model_selection import validation_curve
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import pickle


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 3))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 3))
X_train, X_test = X_train/255, X_test/255


def kNNClassifier(X_train, X_test, y_train, y_test, labels):

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    hyper_params = {'n_neighbors': [5, 10, 15]}

    #########################################################
    ###   UNCOMMENT TO LOAD MODEL       #####################
    # kNN_model = KNeighborsClassifier(n_neighbors=10)
    # best_parameter_search = GridSearchCV(kNN_model, hyper_params)
    # best_parameter_search.fit(X_train, y_train.ravel())
    # print(best_parameter_search.best_params_)


    #########################################################
    ###   UNCOMMENT TO TRAIN AND SAVE MODEL    ##############
    # kNN_model = KNeighborsClassifier(n_neighbors=best_parameter_search.best_params_['n_neighbors'])
    # kNN_model.fit(X_train, y_train.ravel())
    # # save the model to disk
    filename = 'kNN_model.sav'
    # pickle.dump(kNN_model, open(filename, 'wb'))


    # load the model from disk
    kNN_model = pickle.load(open(filename, 'rb'))
    result = kNN_model.score(X_test, y_test)
    print(result)

    y_pred = kNN_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Produces a plot of Test Accuracy per k neighbors (1 through 25)
    k_range = range(1, 26)
    scores = {}
    scores_list = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train.ravel())
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(k_range, scores_list)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.show()


    # Reshape data for displaying images with predictions and truth labels
    y_pred = y_pred.astype(int)
    fig, axes = plt.subplots(ncols=4, nrows=3, sharex=False, sharey=True, figsize=(16, 14))
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 3))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 3))
    X_train, X_test = X_train / 255, X_test / 255
    index = 0
    for i in range(3):
        for j in range(4):
            axes[i, j].set_title('actual:' + labels[y_test[index][0]] + '\n'
                                 + 'predicted:' + labels[y_pred[index]], fontsize=24)
            axes[i, j].imshow(X_test[index], cmap='gray')
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
            index += 1
    plt.show()

    print('kNN model complete')
    return kNN_model


def svmClassifier(X_train, X_test, y_train, y_test):
    filename = 'svc_model.sav'
    filename_linear = 'svc_model_linear.sav'

    print('Starting SVC model')
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    #########################################################
    ###   UNCOMMENT TO LOAD MODEL       #####################
    # # # load the model from disk
    # print('Loading SVC...')
    # svc_model = pickle.load(open(filename, 'rb'))
    # # result = svc_model.score(X_test, y_test)
    # print('Model loaded, printing results:')
    # # print(result)

    #########################################################
    ###   UNCOMMENT TO TRAIN NEW MODEL AND SAVE   ###########
    print('Fitting model...')
    svc_model = svm.SVC(kernel='rbf')
    svc_model.fit(X_train, y_train.ravel())
    print('Model fit complete')
    # save the model to disk
    pickle.dump(svc_model, open(filename_linear, 'wb'))

    y_pred = svc_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return svc_model


def decision_treeClassifier(X_train, X_test, y_train, y_test):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    print('Fitting model...')
    dt_model = DecisionTreeClassifier(max_depth=10)
    dt_model.fit(X_train, y_train.ravel())

    #########################################################
    ###   UNCOMMENT TO RUN GRIDSEARCH   #####################
    # hyper_params = {'max_depth': [2,10,50]}
    # search = GridSearchCV(dt_model, hyper_params)
    # search.fit(X_train, y_train)
    # print('Best Hyperparameters:', search.best_params_)
    #########################################################

    plt.figure(figsize=(15, 15))
    tree.plot_tree(dt_model,
                   filled=True)
    plt.show()

    y_pred = dt_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('Accuracy', accuracy_score(y_test, y_pred))
    print('Precision', precision_score(y_test, y_pred, average='micro'))
    print('Recall', recall_score(y_test, y_pred, average='micro'))

    return dt_model

def cnn_model(X_train, X_test, y_train, y_test):

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

    # number of classes
    number_classes = len(set(y_train.ravel()))

    # Build the model
    i = Input(shape=X_train[0].shape)
    for j in range(4):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    for j in range(4):
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(i)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    for j in range(4):
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    for j in range(4):
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    # Hidden layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    # last hidden layer i.e.. output layer
    x = Dense(number_classes, activation='softmax')(x)

    model = Model(i, x)

    # model description
    model.summary()

    model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=50,validation_data=(X_test, y_test), validation_split=0.2, callbacks=[es])

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.show()

    # model = model(X_train, y_train, X_test, y_test)
    model.save('cifar10_cnn_model')

    return model





#########################################################################
#########################################################################
# TO RUN EACH MODEL, UNCOMMENT:
# kNNClassifier(X_train, X_test, y_train, y_test, labels)
# svmClassifier(X_train, X_test, y_train, y_test)
# decision_treeClassifier(X_train, X_test, y_train, y_test)
cnn_model(X_train, X_test, y_train, y_test)