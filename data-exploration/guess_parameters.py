#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path
import itertools

import pandas as pd
import numpy as np

def get_data(filename):
    """
    Assumes column 0 is the instance index stored in the
    csv file.  If no such column exists, remove the
    index_col=0 parameter.
    """
    data = pd.read_csv(filename, index_col=0)
    return data

def get_feature_and_label_names(my_args, data):
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)

    return features, label


def load_data(my_args, filename):
    data = get_data(filename)
    feature_columns, label_column = get_feature_and_label_names(my_args, data)
    X = data[feature_columns]
    y = data[label_column]
    if my_args.instances > 0 and my_args.instances < X.shape[0]:
        X = X[:my_args.instances]
        y = y[:my_args.instances]
    return X, y

def do_evaluate_parameters(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    y = np.array(y)
    theta = np.array(my_args.parameters)
    offset = np.ones(shape=(X.shape[0],1))
    X = np.array(X)
    X = np.concatenate([offset, X], axis=1)
    # print("X.shape",X.shape)
    # print("y.shape",y.shape)
    # print("theta.shape", theta.shape)

    y_hat = np.sum(theta * X, axis=1)
    # print("y", y)
    # print("y_hat", y_hat)

    mse = 0.0
    for i in range(len(y)):
        dy = y[i] - y_hat[i]
        mse += dy*dy
    mse /= len(y)
    print("MSE:", mse)
    print("theta:", theta)
    return

def do_random_search(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    X = np.array(X)
    y = np.array(y)

    offset = np.ones(shape=(X.shape[0],1))
    X = np.concatenate([offset, X], axis=1)

    rng = np.random.default_rng()

    best_mse = 1.0e9
    best_theta = rng.random([X.shape[1]]) * 40.0 - 20.0

    for rep in range(my_args.search_iterations):
        theta = rng.random([X.shape[1]]) * 40.0 - 20.0

        y_hat = np.sum(theta * X, axis=1)

        mse = 0.0
        for i in range(len(y)):
            dy = y[i] - y_hat[i]
            mse += dy*dy
        mse /= len(y)

        if mse < best_mse:
            best_mse = mse
            best_theta = theta
        
    print("MSE:", best_mse)
    print("theta:", best_theta)

    return

def do_grid_search(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    X = np.array(X)
    y = np.array(y)

    offset = np.ones(shape=(X.shape[0],1))
    X = np.concatenate([offset, X], axis=1)

    minimums = np.array(my_args.minimums)
    maximums = np.array(my_args.maximums)
    grid_size = my_args.grid_size
    
    rng = np.random.default_rng()
    best_mse = 1.0e9
    best_theta = rng.random([X.shape[1]]) * 40.0 - 20.0

    product_args = []
    for i in range(minimums.shape[0]):
        delta = (maximums[i] - minimums[i])/(grid_size-1)
        values = [ minimums[i] + delta*j for j in range(grid_size) ]
        product_args.append(values)

    for theta in itertools.product(*product_args):
        theta = np.array(theta)
        y_hat = np.sum(theta * X, axis=1)

        mse = 0.0
        for i in range(len(y)):
            dy = y[i] - y_hat[i]
            mse += dy*dy
        mse /= len(y)

        if mse < best_mse:
            best_mse = mse
            best_theta = theta
        
    print("MSE:", best_mse)
    print("theta:", best_theta)

    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Guess Fit')
    parser.add_argument('action', default='evaluate-parameters',
                        choices=[ "evaluate-parameters", "random-search", "grid-search" ], 
                        nargs='?', help="desired action")
    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="label",    type=str,   help="column name for label")
    parser.add_argument('--parameters',    '-p', default=None, action="extend", nargs="+", type=float,
                        help="parameters to evaluate")
    parser.add_argument('--instances',     '-i', default=-1,  type=int,   help="number of data instances to use")
    parser.add_argument('--search-iterations',     '-s', default=10000,  type=int,   help="number of random trials to use")
    # grid search specifications
    parser.add_argument('--minimums',    '-m', default=None, action="extend", nargs="+", type=float,
                        help="minimum value for each parameter")
    parser.add_argument('--maximums',    '-M', default=None, action="extend", nargs="+", type=float,
                        help="maximum value for each parameter")
    parser.add_argument('--grid-size',     '-g', default=10,  type=int,   help="number of steps to use in all grid dimensions")
    
    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'evaluate-parameters':
        do_evaluate_parameters(my_args)
    elif my_args.action == 'random-search':
        do_random_search(my_args)
    elif my_args.action == 'grid-search':
        do_grid_search(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))
    
    return

if __name__ == "__main__":
    main(sys.argv)