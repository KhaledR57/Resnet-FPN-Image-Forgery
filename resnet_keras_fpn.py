from __future__ import print_function

import datetime
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import applications, datasets, layers, models
from tensorflow.keras.optimizers import Adam

"""## Define plot functions and learning rate schedulers"""

    
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 0.9*epoch:
        lr *= 0.5e-3
    elif epoch > 0.8*epoch:
        lr *= 1e-3
    elif epoch > 0.6*epoch:
        lr *= 1e-2
    elif epoch > 0.4*epoch:
        lr *= 1e-1
    else: 
        lr = 1e-3
    print('Learning rate: ', lr)
    return lr

"""## Define Models"""

def FPNRes50e100(image_shape, class_number):
    resnet50Backbone = get_backbone_ResNet50(input_shape=image_shape)
    model = customFeaturePyramid2(resnet50Backbone, class_number)
    return model

def FPNRes50V2e100(image_shape, class_number):
    resnet50V2Backbone = get_backbone_ResNet50V2(input_shape=image_shape)
    model = customFeaturePyramid2(resnet50V2Backbone, class_number)
    return model

def FPNRes101e200(image_shape, class_number):
    resnet101Backbone = get_backbone_ResNet101(input_shape=image_shape)
    model = customFeaturePyramid2(resnet101Backbone, class_number)
    return model



def get_backbone_ResNet50(input_shape):
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=input_shape
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


def get_backbone_ResNet50V2(input_shape):
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50V2(
        include_top=False, input_shape=input_shape
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


def get_backbone_ResNet101(input_shape):
    """Builds ResNet101 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet101(
        include_top=False, input_shape=input_shape
    )
    
    c1_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv2_block1_out"]
    ]

    c2_output, c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block10_out", "conv4_block23_out", "conv5_block3_out"]
    ]

    return keras.Model(
        inputs=[backbone.inputs], outputs=[c1_output, c2_output,c3_output, c4_output, c5_output]
    )


class customFeaturePyramid2(keras.models.Model):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50, ResNet101 and V1 counterparts.
    """

    def __init__(self, backbone=None, class_number=2, **kwargs):
        super(customFeaturePyramid2, self).__init__(name="customFeaturePyramid2", **kwargs)
        self.backbone = backbone if backbone else get_backbone_ResNet50()
        self.class_number = class_number
        self.conv_c1_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c2_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")

        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)
        self.dense_d1 = keras.layers.Dense(64,
                                           activation='relu',
                                           kernel_initializer='he_uniform')
        self.dense_d2 = keras.layers.Dense(self.class_number,
                                           activation='sigmoid',
                                           kernel_initializer='he_normal')

    def call(self, images, training=False):
        c1_output, c2_output,c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p1_output = self.conv_c1_1x1(c1_output)
        p2_output = self.conv_c2_1x1(c2_output)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        
        p5_output = p5_output
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p2_output = p2_output + self.upsample_2x(p3_output)
        
        p2_output = self.conv_c5_3x3(p2_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)

        p2_output = keras.layers.Flatten()(p2_output)
        p3_output = keras.layers.Flatten()(p3_output)
        p4_output = keras.layers.Flatten()(p4_output)
        p5_output = keras.layers.Flatten()(p5_output)
        m1_output = keras.layers.Concatenate(axis=1)([p2_output,
                                                      p3_output,
                                                      p4_output,
                                                      p5_output,])
        m1_output = keras.layers.Flatten()(m1_output)
        m1_output = self.dense_d1(m1_output)
        m1_output = self.dense_d2(m1_output)
        return m1_output


"""model = FPNRes101e200(img_shape, num_classes)
model.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy'])
history = model.fit(train_img,
                    train_lab,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(test_img, test_lab),
                    shuffle=True)"""
                    
