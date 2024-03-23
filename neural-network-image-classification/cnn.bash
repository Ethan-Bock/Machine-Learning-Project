#!/bin/bash

if [ ! -e fashion-mnist-mini-train-validate.csv ]; then
    split=../030-regression-fit/split_data.py
    $split --data-file fashion-mnist-mini.csv --test-ratio 0.20
    $split --data-file fashion-mnist-mini-train.csv --test-ratio 0.30 --train-file fashion-mnist-mini-train-fit.csv --test-file fashion-mnist-mini-train-validate.csv
fi


### ARCHONE
#model.add(keras.layers.Dense(20, activation="relu"))
#model.add(keras.layers.Dropout(0.2))
#model.add(keras.layers.Dense(12, activation="relu"))
#model.add(keras.layers.Dropout(0.2))
#model.add(keras.layers.Dense(10, activation="softmax"))

### ARCHTWO
#model.add(keras.layers.Conv2D(filters=8,kernel_size=(2,2), padding="same", strides=(1,1), activation="relu", input_shape=input_shape))
#model.add(keras.layers.Conv2D(filters=8,kernel_size=(2,2), padding="same", strides=(1,1), activation="relu", input_shape=input_shape))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(16, activation="relu"))
#model.add(keras.layers.Dropout(0.2))
#model.add(keras.layers.Dense(8, activation="relu"))
#model.add(keras.layers.Dropout(0.5))
#model.add(keras.layers.Dense(10, activation="softmax"))


### ARCHTHREE
#model.add(keras.layers.Conv2D(filters=8,kernel_size=(2,2), padding="same", strides=(1,1), activation="relu", input_shape=input_shape))
#model.add(keras.layers.Conv2D(filters=8,kernel_size=(2,2), padding="same", strides=(1,1), activation="relu", input_shape=input_shape))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.Conv2D(filters=16, kernel_size=(4,4), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(16, activation="relu"))
#model.add(keras.layers.Dropout(0.3))
#model.add(keras.layers.Dense(10, activation="relu"))
#model.add(keras.layers.Dropout(0.3))
#model.add(keras.layers.Dense(10, activation="softmax"))

### ARCHFOUR
#model.add(keras.layers.Conv2D(filters=8,kernel_size=(2,2), padding="same", strides=(1,1), activation="relu", input_shape=input_shape))
#model.add(keras.layers.Conv2D(filters=8,kernel_size=(2,2), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,1), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.Conv2D(filters=16, kernel_size=(2,2), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,1), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2)))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(16, activation="relu"))
#model.add(keras.layers.Dropout(0.4))
#model.add(keras.layers.Dense(8, activation="relu"))
#model.add(keras.layers.Dropout(0.1))
#model.add(keras.layers.Dense(10, activation="softmax"))





# Name the model file after the architecture, presumably with some bookkeeping on your part.
model_file="model-description-here.joblib"

# Fit the new model
./cnn_classification.py cnn-fit --train-file fashion-mnist-mini-train-fit.csv --model-file ${model_file}
# Score the new model
./cnn_classification.py score --train-file fashion-mnist-mini-train-fit.csv --test-file fashion-mnist-mini-train-validate.csv --show-test 1 --model-file ${model_file}

#
# Record results, and observations.
# Go to Configure the network architecture above.
#