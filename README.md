# Convolutional-Neural-Net-Pipeline-Optimization-and-Training

Keyword: scratch... I have no doubt this is very messy code, but a notebook style approach proved very effective for getting the intial building classification model off the ground. If these training steps were to be put into production, the code should be heavily refactored.

Since the two types of structures classified (residential, aka 'house', and non-residential aka 'barn') have significant overlap in appearance in the real world, I considered a training accuracy of 79 and validation accurace of 68 to be relatively good!

Also, although a variety of regions were sampled for training data (planimetrics we had received from consultants for some of our prior project areas around the country), regional skewing should still be expected based on visible light profiles. To counter this, I added an option to normalize scores to a specific percentage of 'residential' structures. The GIS analysts user would simply enter their estimate (generally around 30-50%) and the scoring system would enforce it.

The code used in production to convert from raw image files to the numpy-based, and more optimized TFRecord format is also added. Below are a few snippets I can share on the implementation of running the model over each 'TFRecord'.

from __future__ import absolute_import, division, print_function, unicode_literals

def MobileNetV2_CNN(latestModel, testSlice, numImages, pct_Barns=None):

    import os, time
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

    BATCH_SIZE=32
    test_batches = testSlice.batch(BATCH_SIZE)
    model = tf.keras.models.load_model(latestModel)

    def tensorToString(t):
        try:
            t_numpy = t.numpy()
            return t_numpy.decode("utf-8")
        except:
            return str(t)

    numBatches = -(-numImages//BATCH_SIZE) #awesome way of rounding up:
    #https://stackoverflow.com/questions/17141979/round-a-floating-point-number-down-to-the-nearest-integer

    pDict = {}
    for imageRecords,imageIds in test_batches.take(numBatches):
        predictions = model.predict(imageRecords)
        for i in range(len(predictions)): #i ranges from 0 to 31
            imageId = tensorToString(imageIds[i])
            prediction = predictions[i][0]
            pDict[imageId] = prediction
