#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def load_image(path,shape,color_mode="rgb"):
    image = keras.preprocessing.image.load_img(path,target_size=shape,interpolation="nearest",color_mode=color_mode)
    image = keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    return image

def save_image(path,data):
    keras.preprocessing.image.save_img(path,data,scale=True)
    return

def zoom_in_image(data, scale):
    datas = [data]
    images = tf.image.resize(datas, size=(scale*data.shape[0], scale*data.shape[1]))
    return images[0]

def load_images(path_list,shape,color_mode="rgb"):
    images = []
    for path in path_list:
        image = load_image(path,shape,color_mode)
        images.append(image)
    return np.array(images)

def display_images(images,color_mode="rgb"):
    for i in range(images.shape[0]):
        if color_mode == "rgb":
            # rgb
            plt.imshow(images[i])
            plt.show()
        elif color_mode == "grayscale":
            # grayscale
            for j in range(images.shape[3]):
                plt.imshow(images[i,:,:,j],cmap="gray")
                plt.show()
        else:
            print("Unexpected color_mode:",color_mode)
    return