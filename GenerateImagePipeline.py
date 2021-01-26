def GenerateImagePipeline(imDir):

    def reduceToThreeChannelMatrix(m):
        R, G, B, IR = m[:,:,0:1], m[:,:,1:2], m[:,:,2:3], m[:,:,3:4]
        return np.concatenate((IR, R/2+G/2, B), axis=2)

    import time, os, pickle
    from PIL import Image
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    start = time.time()
    imageTensors, nameTensors, labelTensors = [], [], []
    count = 0

    size = sum(os.path.getsize(os.path.join(imDir,f)) for f in os.listdir(imDir) if os.path.isfile(os.path.join(imDir,f)))
    print(str(size/1e7) + " MB of data in " + imDir)

    for imName in os.listdir(imDir):

        imPath = os.path.join(imDir, imName)
        pilIm = Image.open(imPath)
        imArray = np.array(pilIm)
        imArray = reduceToThreeChannelMatrix(imArray)

        imTensor = tf.convert_to_tensor((imArray/127.5)-1, dtype=tf.float32) # normalizes RGBI vals from -1 to 1
        imTensor = tf.image.resize(imTensor, (96, 96)) # resizes all images to square
        labelTensor = tf.convert_to_tensor(0)
        nameTensor = tf.convert_to_tensor(imName.encode('utf-8'))

        nameTensors.append(nameTensor)
        labelTensors.append(labelTensor)
        imageTensors.append(imTensor)
        count+=1

    sliceDataset = tf.data.Dataset.from_tensor_slices(((imageTensors, labelTensors), nameTensors))
    print(str(count) + ' image matrices collected in ' + str(time.time()-start) + ' seconds')
    return sliceDataset, count