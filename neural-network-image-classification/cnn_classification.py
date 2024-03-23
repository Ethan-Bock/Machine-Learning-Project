#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.metrics
import joblib
import tensorflow as tf
import tensorflow.keras as keras

from cnn_common import *

################################################################
# CNN functions


def create_model(my_args, input_shape):
    model = keras.models.Sequential()

    #filters = how many basic shapes, kernel is how big of an area, pooling reduces resolution
    #con con pool
    model.add(keras.layers.Conv2D(filters=8,kernel_size=(2,2), padding="same", strides=(1,1), activation="relu", input_shape=input_shape))
    model.add(keras.layers.Conv2D(filters=8,kernel_size=(2,2), padding="same", strides=(1,1), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,1), padding="same", strides=(1,1), activation="relu"))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(2,2), padding="same", strides=(1,1), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,1), padding="same", strides=(1,1), activation="relu"))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(8, activation="relu"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], optimizer=keras.optimizers.Adam())
    return model

def do_cnn_fit(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_pseudo_fit_pipeline(my_args)
    pipeline.fit(X)
    X = pipeline.transform(X) # If the resulting array is sparse, use .todense()
    # reshape the 784 pixels into a 2D greyscale image
    X = np.reshape(X,[X.shape[0],28,28,1])
    
    model = create_model(my_args, X.shape[1:])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, verbose=1, callbacks=[early_stopping], validation_split=my_args.validation_split)

    # save the last file
    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump((pipeline, model), model_file)
    return

# CNN functions
################################################################

################################################################
# Evaluate existing models functions

def sklearn_metric(y, yhat):
    cm = sklearn.metrics.confusion_matrix(y, yhat)
    ###
    header = "+"
    for col in range(cm.shape[1]):
        header += "-----+"
    rows = [header]
    for row in range(cm.shape[0]):
        row_str = "|"
        for col in range(cm.shape[1]):
            row_str += "{:4d} |".format(cm[row][col])
        rows.append(row_str)
    footer = header
    rows.append(footer)
    table = "\n".join(rows)
    print(table)
    print()
    ###
    if cm.shape[0] == 2:
        precision = sklearn.metrics.precision_score(y, yhat)
        recall = sklearn.metrics.recall_score(y, yhat)
        f1 = sklearn.metrics.f1_score(y, yhat)
        print("precision: {}".format(precision))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))
    else:
        report = sklearn.metrics.classification_report(y, yhat)
        print(report)
    return

def show_score(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    if my_args.show_test:
        test_file = get_test_filename(my_args.test_file, train_file)
        if not os.path.exists(test_file):
            raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    basename = get_basename(train_file)

    X_train, y_train = load_data(my_args, train_file)
    if my_args.show_test:
        X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)

    if isinstance(pipeline, tuple):
        (pipeline, model) = pipeline
        X_train = pipeline.transform(X_train) # .todense()
        # reshape the 784 pixels into a 2D greyscale image
        X_train = np.reshape(X_train,[X_train.shape[0],28,28,1])
        yhat_train = np.argmax(model.predict(X_train), axis=1)
        print()
        print("{}: train: ".format(basename))
        print()
        sklearn_metric(y_train, yhat_train)
        print()

        if my_args.show_test:
            X_test = pipeline.transform(X_test) # .todense()
            X_test = np.reshape(X_test,[X_test.shape[0],28,28,1])
            yhat_test = np.argmax(model.predict(X_test), axis=1)
            print()
            print("{}: test: ".format(basename))
            print()
            print()
            sklearn_metric(y_test, yhat_test)
            print()

    else:
        yhat_train = pipeline.predict(X_train)
        print()
        print("{}: train: ".format(basename))
        print()
        sklearn_metric(y_train, yhat_train)
        print()

        if my_args.show_test:
            yhat_test = pipeline.predict(X_test)
            print()
            print("{}: test: ".format(basename))
            print()
            print()
            sklearn_metric(y_test, yhat_test)
            print()
        
    return

# Evaluate existing models functions
################################################################



def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Image Classification with CNN')
    parser.add_argument('action', default='cnn-fit',
                        choices=[ "cnn-fit", "score" ], 
                        nargs='?', help="desired action")

    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")

    # Pipeline configuration
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="label",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--categorical-missing-strategy', default="",   type=str,   help="strategy for missing categorical information")
    parser.add_argument('--numerical-missing-strategy', default="",   type=str,   help="strategy for missing numerical information")
    parser.add_argument('--print-preprocessed-data', default=0,         type=int,   help="0 = don't do the debugging print, 1 = do print (default=0)")

    parser.add_argument('--shuffle',                       action='store_true',  help="Shuffle data when loading.")
    parser.add_argument('--no-shuffle',    dest="shuffle", action='store_false', help="Do not shuffle data when loading.")
    parser.set_defaults(shuffle=True)


    # hyper parameters
    parser.add_argument('--validation-split', default=0.1,         type=float,   help="validation split fraction (default=0.1)")

    # debugging/observations
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")


    my_args = parser.parse_args(argv[1:])

    # Do any special fixes/checks here
    
    return my_args


def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'cnn-fit':
        do_cnn_fit(my_args)
    elif my_args.action == 'score':
        show_score(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))
    return

if __name__ == "__main__":
    main(sys.argv)