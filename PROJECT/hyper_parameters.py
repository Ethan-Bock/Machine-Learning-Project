#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import numpy as np
import sklearn.tree
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.base
import sklearn.metrics
import sklearn.impute
import sklearn.model_selection
import scipy.stats
import joblib
import pprint

import matplotlib.pyplot as plt


class PipelineNoop(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self):
        return
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X


## Added code to attempt to use basic DT tree but was proved unsuccessful and unnecessary ##
class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()
        if isinstance(output, pd.DataFrame):
            if self.columns is not None:
                for col in self.columns:
                    if pd.api.types.is_object_dtype(output[col]):
                        output[col] = sklearn.preprocessing.LabelEncoder().fit_transform(output[col])
            else:
                for colname, col in output.iteritems():
                    if pd.api.types.is_object_dtype(col):
                        output[colname] = sklearn.preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)




# Pipeline member to display the data at this stage of the transformation.
class Printer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, title):
        self.title = title
        return
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        print("{}::type(X)".format(self.title), type(X))
        print("{}::X.shape".format(self.title), X.shape)
        if not isinstance(X, pd.DataFrame):
            print("{}::X[0]".format(self.title), X[0])
        print("{}::X".format(self.title), X)
        return X


class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
        self.mCategoricalPredictors = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
        self.mNumericalPredictors = []
        self.mLabels = ["math_score", "reading_score", "writing_score"]

        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        
        if do_predictors:
            if do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
            
        return

    def fit( self, X, y=None ):
        return self

    def transform( self, X, y=None ):
        values = X[self.mAttributes]
        return values


def get_test_filename(test_file, filename):
    if test_file == "":
        basename = get_basename(filename)
        test_file = "{}-test.csv".format(basename)
    return test_file

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))

    stub = "-train"
    if basename[len(basename)-len(stub):] == stub:
        basename = basename[:len(basename)-len(stub)]

    return basename

def get_model_filename(model_file, filename):
    if model_file == "":
        basename = get_basename(filename)
        model_file = "{}-model.joblib".format(basename)
    return model_file

def get_search_grid_filename(search_grid_file, filename):
    if search_grid_file == "":
        basename = get_basename(filename)
        search_grid_file = "{}-search-grid.joblib".format(basename)
    return search_grid_file

def get_data(filename):
    data = pd.read_csv(filename)
    return data

def load_data(my_args, filename):
    data = get_data(filename)
    feature_columns, label_columns = get_feature_and_label_names(my_args, data)
    X = data[feature_columns]
    y = data[label_columns]
    return X, y

### EDITED
def get_feature_and_label_names(my_args, data):
    label_columns = my_args.label
    feature_columns = my_args.features

    ##Check if label columns are given
    if not label_columns:
        raise ValueError("No label columns provided.")

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column not in label_columns:
                features.append(feature_column)

    return features, label_columns



def make_numerical_feature_pipeline(my_args):
    items = []
    
    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))

    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))

    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Numerical Preprocessing")))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline

def make_categorical_feature_pipeline(my_args):
    items = []
    
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))

    if my_args.categorical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.categorical_missing_strategy)))

    ### sklearn's decision tree classifier requires all input features to be numerical
    ### one hot encoding accomplishes this.
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Categorial Preprocessing")))

    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline(my_args):
    """
    Numerical features and categorical features are usually preprocessed
    differently. We split them out here, preprocess them, then merge
    the preprocessed features into one group again.
    """
    items = []

    items.append(("numerical", make_numerical_feature_pipeline(my_args)))
    items.append(("categorical", make_categorical_feature_pipeline(my_args)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline

def make_decision_tree_fit_pipeline(my_args):
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.print_preprocessed_data:
        items.append(("printer", Printer("Final Preprocessing")))
    dtargs = {
        "criterion": my_args.criterion,
        "splitter": my_args.splitter,
        "max_depth": my_args.max_depth,
        "min_samples_split": my_args.min_samples_split,
        "min_samples_leaf": my_args.min_samples_leaf,
        "max_features": my_args.max_features,
        "max_leaf_nodes": my_args.max_leaf_nodes,
        "min_impurity_decrease": my_args.min_impurity_decrease
    }
    items.append(("model", sklearn.tree.DecisionTreeClassifier(**dtargs)))
    return sklearn.pipeline.Pipeline(items)


########################################################################################
# Small set of parameters
########################################################################################

def make_numerical_predictor_params(my_args):
    params = { 
        "features__numerical__numerical-features-only__do_predictors" : [ True ],
        "features__numerical__numerical-features-only__do_numerical" : [ True ],
    }
    if my_args.numerical_missing_strategy:
        params["features__numerical__missing-data__strategy"] = [ 'most_frequent' ] # [ 'median', 'mean', 'most_frequent' ]
    if my_args.use_polynomial_features:
        params["features__numerical__polynomial-features__degree"] = [ 2 ] # [ 1, 2, 3 ]

    return params

def make_categorical_predictor_params(my_args):
    params = { 
        "features__categorical__categorical-features-only__do_predictors" : [ True ],
        "features__categorical__categorical-features-only__do_numerical" : [ False ],
        "features__categorical__encode-category-bits__categories": [ 'auto' ],
        "features__categorical__encode-category-bits__handle_unknown": [ 'ignore' ],
    }
    if my_args.categorical_missing_strategy:
        params["features__categorical__missing-data__strategy"] = [ 'most_frequent' ]
    return params

def make_predictor_params(my_args):
    p1 = make_numerical_predictor_params(my_args)
    p2 = make_categorical_predictor_params(my_args)
    p1.update(p2)
    return p1

def make_decision_tree_params_grid(my_args):
    params = make_predictor_params(my_args)
    tree_params = {
        "model__criterion": [ "log_loss" ], # [ "entropy", "gini" ],
        "model__splitter": [ "best" ], # [ "best", "random" ],
        "model__max_depth": [ 3 ], # [ 2, 3, 4 ],
        "model__min_samples_split": [ 0.04 ], # [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 ],
        "model__min_samples_leaf":  [ 0.02 ],  # [ 0.01, 0.02, 0.04, 0.1 ],
        "model__max_features":  [ "sqrt" ], # [ "sqrt", "log2", None ],
        "model__max_leaf_nodes": [ 4 ], # [ 2, 4, 8, 16, 32, 64, None ],
        "model__min_impurity_decrease": [ 0.8 ], # [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
    }
    params.update(tree_params)
    return params


########################################################################################
# Large set of parameters
########################################################################################

def make_numerical_predictor_params_big(my_args):
    params = { 
        "features__numerical__numerical-features-only__do_predictors" : [ True ],
        "features__numerical__numerical-features-only__do_numerical" : [ True ],
    }
    if my_args.numerical_missing_strategy:
        params["features__numerical__missing-data__strategy"] = [ 'mean', 'most_frequent' ]
    if my_args.use_polynomial_features:
        params["features__numerical__polynomial-features__degree"] = [ 1 ]
    return params

def make_categorical_predictor_params_big(my_args):
    params = { 
        "features__categorical__categorical-features-only__do_predictors" : [ True ],
        "features__categorical__categorical-features-only__do_numerical" : [ False ],
        "features__categorical__encode-category-bits__categories": [ 'auto' ],
        "features__categorical__encode-category-bits__handle_unknown": [ 'ignore' ],
    }
    if my_args.categorical_missing_strategy:
        params["features__categorical__missing-data__strategy"] = [ 'most_frequent' ]
    return params

def make_predictor_params_big(my_args):
    p1 = make_numerical_predictor_params_big(my_args)
    p2 = make_categorical_predictor_params_big(my_args)
    p1.update(p2)
    return p1

def make_decision_tree_params_grid_big(my_args):
    params = make_predictor_params_big(my_args)
    tree_params = {
        "model__criterion": ["gini", "entropy", "log_loss"], #["gini", "entropy", "log_loss"]
        "model__splitter": [ "best" ],
        "model__max_depth": [ 3, 4, 5 ], #[ 3, 4, 5 ]
        "model__min_samples_split": [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64 ], #[ 0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64 ],
        "model__min_samples_leaf":  [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64 ], #[ 0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.48, 0.64 ],
        "model__max_features":  [ "sqrt", "log2" ], #[ "sqrt", "log2"]
        "model__max_leaf_nodes": [2, 4, 8, 12, 16, 24, 32, None], #[2, 4, 8, 12, 16, 24, 32]
        "model__min_impurity_decrease": [ 0.0, 0.01, 0.02, 0.04, 0.06, 0.08], #[ 0.0, 0.01, 0.02, 0.04, 0.06, 0.08]
    }
    params.update(tree_params)
    return params

### EDIT
def do_grid_search(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_decision_tree_fit_pipeline(my_args)
    if my_args.big_params == 1:
        fit_params = make_decision_tree_params_grid_big(my_args)
    else:
        fit_params = make_decision_tree_params_grid(my_args)

    ### EDITED
    #Optimal score is 0.0
    search_grid = sklearn.model_selection.GridSearchCV(pipeline, fit_params, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    #Transfered from neg_root_mean_squared_error to neg_mean_absolute_error
    search_grid.fit(X, y)
    
    search_grid_file = get_search_grid_filename(my_args.search_grid_file, train_file)
    joblib.dump(search_grid, search_grid_file)

    model_file = get_model_filename(my_args.model_file, train_file)
    joblib.dump(search_grid.best_estimator_, model_file)
    return

def get_feature_names(pipeline, X):
    primary_feature_names = list(X.columns[:])
    if 'polynomial-features' in pipeline['features'].named_steps:
        secondary_powers = pipeline['features']['polynomial-features'].powers_
        feature_names = []
        for powers in secondary_powers:
            s = ""
            for i in range(len(powers)):
                for j in range(powers[i]):
                    if len(s) > 0:
                        s += "*"
                    s += primary_feature_names[i]
            feature_names.append(s)
            logging.info("powers: {}  s: {}".format(powers, s))
    else:
        logging.info("polynomial-features not in features: {}".format(pipeline['features'].named_steps))
        feature_names = primary_feature_names
    return feature_names

def get_scale_offset(pipeline, count):
    if 'scaler' in pipeline['features'].named_steps:
        scaler = pipeline['features']['scaler']
        logging.info("scaler: {}".format(scaler))
        logging.info("scale: {}  mean: {}  var: {}".format(scaler.scale_, scaler.mean_, scaler.var_))
        theta_scale = 1.0 / scaler.scale_
        intercept_offset = scaler.mean_ / scaler.scale_
    else:
        theta_scale = np.ones(count)
        intercept_offset = np.zeros(count)
        logging.info("scaler not in features: {}".format(pipeline['features'].named_steps))
    return theta_scale, intercept_offset

### EDIT
def show_function(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))
    
    X, y = load_data(my_args, train_file)
    pipeline = joblib.load(model_file)

    feature_names = get_feature_names(pipeline, X)
    scale, offset = get_scale_offset(pipeline, len(feature_names))

    features = pipeline['features']
    X = features.transform(X)
    regressor = pipeline['model']

    intercept_offset = 0.0
    for i in range(len(regressor.coef_)):
        intercept_offset += regressor.coef_[i] * offset[i]

    s = "{}".format(regressor.intercept_[0]-intercept_offset)
    for i in range(len(regressor.coef_)):
        if len(feature_names[i]) > 0:
            t = "({}*{})".format(regressor.coef_[i]*scale[i], feature_names[i])
        else:
            t = "({})".format(regressor.coef_[i])
        if len(s) > 0:
            s += " + "
        s += t

    basename = get_basename(train_file)
    print("{}: {}".format(basename, s))
    return


# Added my args so that I can access my_args.label
def sklearn_metric(my_args, y, yhat):
    encoder = MultiColumnLabelEncoder(columns=my_args.label)
    y_encoded = encoder.fit_transform(y)
    yhat_encoded = encoder.transform(yhat)
    y_ravel = y_encoded.to_numpy().ravel()
    yhat_encoded_ravel =  yhat_encoded.ravel()
    mae = sklearn.metrics.mean_absolute_error(y_ravel, yhat_encoded_ravel)
    print("Mean Absolute Error (MAE): {:.4f}".format(mae))

def test_score(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("Training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("Testing data file, '{}', does not exist.".format(test_file))

    # Load the test data
    X_test, y_test = load_data(my_args, test_file)
    
    # Load the trained model
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))
    pipeline = joblib.load(model_file)
    
    # Use the model to predict the target variable on the test data
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model performance on the test data
    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    print("Test MAE:", mae)
    return

def show_score(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    X_train, y_train = load_data(my_args, train_file)
    X_test, y_test = load_data(my_args, test_file)
    pipeline = joblib.load(model_file)
    
    basename = get_basename(train_file)
    yhat_train = pipeline.predict(X_train)
    print()
    print("{}: train: ".format(basename))
    sklearn_metric(my_args, y_train, yhat_train)
    print()
    
    if my_args.show_test:
        yhat_test = pipeline.predict(X_test)
        print()
        print("{}: test: ".format(basename))
        sklearn_metric(my_args, y_test, yhat_test)
        print()
        
    return

def do_fit(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    X, y = load_data(my_args, train_file)
    
    pipeline = make_decision_tree_fit_pipeline(my_args)
    pipeline.fit(X, y)

    model_file = get_model_filename(my_args.model_file, train_file)

    joblib.dump(pipeline, model_file)
    return

def show_model(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    test_file = get_test_filename(my_args.test_file, train_file)
    if not os.path.exists(test_file):
        raise Exception("testing data file, '{}', does not exist.".format(test_file))
    
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))

    pipeline = joblib.load(model_file)
    tree = pipeline['model']

    fig = plt.figure(figsize=(6, 6)) # size of figure in inches
    ax  = fig.add_subplot(1, 1, 1)

    sklearn.tree.plot_tree(tree, ax=ax)
    fig.tight_layout()
    fig.savefig("tree.png")
    plt.close(fig)
    
    return

def show_best_params(my_args):

    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))

    search_grid_file = get_search_grid_filename(my_args.search_grid_file, train_file)
    if not os.path.exists(search_grid_file):
        raise Exception("Search grid file, '{}', does not exist.".format(search_grid_file))
    search_grid = joblib.load(search_grid_file)

    pp = pprint.PrettyPrinter(indent=4)
    grid_Score = -(search_grid.best_score_)
    print("Best Score:", grid_Score)
    #print("Best Score:", search_grid.best_score_)
    print("Best Params:")
    
    # This was added so it would only prints model related params
    best_model_params = {k: v for k, v in search_grid.best_params_.items() if k.startswith('model__')}
    #Comment the line above out to get full params 

    pp.pprint(best_model_params)
    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Fit Data with Classification Model')
    parser.add_argument('action', default='DT',
                        choices=[ "DT", "score", "show-model", "test-score", "grid-search", "show-best-params"], 
                        nargs='?', help="desired action")
    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")
    parser.add_argument('--search-grid-file', '-g', default="", type=str,   help="name of file for the search grid (default is constructed from train file name when fitting)")
    parser.add_argument('--random-seed',   '-R', default=314159265,type=int,help="random number seed (-1 to use OS entropy)")
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label', '-l', nargs='+', default="label",   type=str,   help="column name for label")
    parser.add_argument('--use-polynomial-features', '-p', default=0,         type=int,   help="degree of polynomial features.  0 = don't use (default=0)")
    parser.add_argument('--use-scaler',    '-s', default=0,         type=int,   help="0 = don't use scaler, 1 = do use scaler (default=0)")
    parser.add_argument('--show-test',     '-S', default=0,         type=int,   help="0 = don't show test loss, 1 = do show test loss (default=0)")
    parser.add_argument('--categorical-missing-strategy', default="",   type=str,   help="strategy for missing categorical information")
    parser.add_argument('--numerical-missing-strategy', default="",   type=str,   help="strategy for missing numerical information")
    parser.add_argument('--print-preprocessed-data', default=0,         type=int,   help="0 = don't do the debugging print, 1 = do print (default=0)")

    # decision tree parameters
    parser.add_argument('--criterion', default="entropy",  choices=('entropy', 'gini'),      help="use entropy or gini for impurity calculations")
    parser.add_argument('--splitter', default="best",      choices=('best', 'random'),       help="use best feature found or random feature to split")
    parser.add_argument('--max-features', default=None,    choices=('auto', 'sqrt', 'log2'), help="how many features to examine when looking for best splitter (default=None)")
    parser.add_argument('--max-depth', default=None,             type=int,   help="maximum depth of tree to learn (default=None)")
    parser.add_argument('--min-samples-split', default=2,        type=int,   help="minimum number of samples required to split a node.")
    parser.add_argument('--min-samples-leaf', default=1,         type=int,   help="minimum number of samples required to create a leaf node.")
    parser.add_argument('--max-leaf-nodes', default=None,        type=int,   help="maximum number of leaf nodes in learned tree (default=None)")
    parser.add_argument('--min-impurity-decrease', default=0.0,  type=float, help="minimum improvement in impurity to create a split (default=0.0)")

    parser.add_argument('--n-search-iterations', default=10,     type=int,   help="number of random iterations in randomized grid search.")
    parser.add_argument('--big-params',          default=0,      type=int,   help="0 = use small hyper parameter set for grid search, 1 = use big set. (default=0)")
    my_args = parser.parse_args(argv[1:])

    # Do any special fixes/checks here
    allowed_categorical_missing_strategies = ("most_frequent")
    if my_args.categorical_missing_strategy != "":
        if my_args.categorical_missing_strategy not in allowed_categorical_missing_strategies:
            raise Exception("Missing categorical strategy {} is not in the allowed list {}.".format(my_args.categorical_missing_strategy, allowed_categorical_missing_strategies))

    allowed_numerical_missing_strategies = ("mean", "most_frequent")
    if my_args.numerical_missing_strategy != "":
        if my_args.numerical_missing_strategy not in allowed_numerical_missing_strategies:
            raise Exception("Missing numerical strategy {} is not in the allowed list {}.".format(my_args.numerical_missing_strategy, allowed_numerical_missing_strategies))
    return my_args

def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.WARN)
    if my_args.action == 'DT':
        do_fit(my_args)
    elif my_args.action == 'grid-search':
        do_grid_search(my_args)
    elif my_args.action == "show-model":
        show_model(my_args)
    elif my_args.action == "score":
        show_score(my_args)
    elif my_args.action == "test-score":
        test_score(my_args)
    elif my_args.action == "show-best-params":
        show_best_params(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))    
    return
if __name__ == "__main__":
    main(sys.argv)
    