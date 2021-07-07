from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
#import matplotlib.pyplot as plt
import numpy as np
import glob
import random as r
import json
import skimage.io as io
import skimage.transform as trans
from sklearn.model_selection import train_test_split
#from raster_classes import *
#import pickle
#import rasterio as rio
# import logging
# logger = tf.get_logger()
# logger.setLevel(logging.ERROR)


rocks_limit = 0

# done
def normalizeImage(imgB):
        img=(np.array(imgB, 'float')-128)/128
        return img


# done
def prepare_dataset(datasetPath, rocks_db, classes_structure, classLevel):
    with open(rocks_db) as data_file:
        rocks = json.load(data_file)

    if rocks_limit > 0:
        del rocks[rocks_limit:]    
        
    usableSet=[]
    for pedra in rocks:
        #print("Checking stone " + str(pedra["ID"]))
        # img = io.imread(os.path.join(datasetPath, pedra['Diretorio Img']))

        if (pedra["Largura da Imagem"] == 780):
            if (pedra["Altura da Imagem"] == 480):
                # if (img.shape[0]==480):
                #     if (img.shape[1]==780):
                        usableSet.append(pedra)

    # rocks=usableSet
    # totalN=len(rocks)
    # k=int(totalN*0.8)
    # r.seed(1)
    # trainSet = r.sample(rocks, k)
    # testSet = []
    # for pedra in rocks:
    #     if pedra not in trainSet:
    #         testSet.append(pedra)
    # return trainSet, testSet

    rocks=usableSet
    classes_list_L0 = classes_structure["classes_list_L0"]
    classes_list_L1 = classes_structure["classes_list_L1"]
    classes_list_L2 = classes_structure["classes_list_L2"]
    classes_L0_dict = dict(zip(classes_list_L0, list(range(0, len(classes_list_L0)))))
    classes_L1_dict = dict(zip(classes_list_L1, list(range(0, len(classes_list_L1)))))
    classes_L2_dict = dict(zip(classes_list_L2, list(range(0, len(classes_list_L2)))))

    classes_hierarchy_L0L1 = classes_structure["classes_hierarchy_L0L1"]
    classes_hierarchy_L1L2 = classes_structure["classes_hierarchy_L1L2"]

    datasetX = np.ndarray(len(usableSet))
    datasetY = np.ndarray(len(usableSet),  dtype=int)

    i=0
    for pedra in rocks:
        datasetX[i]=pedra["ID"]
        class_Name = pedra['Classe']
        class_ID = classes_L0_dict[class_Name]
        if classLevel > 0:
            superClassName = classes_hierarchy_L0L1[class_Name]
            superClassID = classes_L1_dict[superClassName]
            class_Name = superClassName
            class_ID = superClassID
            if classLevel > 1:
                superClassName = classes_hierarchy_L1L2[class_Name]
                superClassID = classes_L2_dict[superClassName]
                class_Name = superClassName
                class_ID = superClassID
        datasetY[i]=class_ID
        i=i+1

    # duplicate examples that are the only samples for their class, so that we can do stratified split
    filhos_unicos=[]
    filhos_unicos_class=[]
    for i in range(datasetY.shape[0]):
        c=datasetY[i]
        encontrei_outro=False
        for j in range(datasetY.shape[0]):
            outro_c=datasetY[j]
            if i!=j and outro_c==c:
                encontrei_outro=True
                break
        if not encontrei_outro:
            filhos_unicos.append(i)
            filhos_unicos_class.append(c)

    if len(filhos_unicos)>0:
        newDatasetX=np.ndarray(len(usableSet)+len(filhos_unicos))
        newDatasetX[0:len(usableSet)]=datasetX[:]
        i=len(usableSet)
        for c in filhos_unicos:
            newDatasetX[i]=c
            i=i+1

        newDatasetY=np.ndarray(len(usableSet)+len(filhos_unicos_class),  dtype=int)
        newDatasetY[0:len(usableSet)] = datasetY[:]
        i=len(usableSet)
        for c in filhos_unicos_class:
            newDatasetY[i]=c
            i=i+1

        datasetX=newDatasetX
        datasetY=newDatasetY



    '''Faz a separação do training set(80%) do test set(20%)'''
    X_train, X_test, y_train, y_test = train_test_split(datasetX, datasetY, test_size=0.2, random_state=1, stratify=datasetY, shuffle=True)

    '''Adiciona a variavel X_train à lista trainSet'''
    trainSet = []
    for i in range(0, len(X_train)):
        ID = X_train[i]
        for pedra in rocks:
            if (pedra['ID']==ID):
                trainSet.append(pedra)
                break

    '''Adiciona a variavel X_test à lista testSet'''
    testSet = []
    for i in range(0, len(X_test)):
        ID=X_test[i]
        for pedra in rocks:
            if (pedra['ID']==ID):
                testSet.append(pedra)
                break
    return trainSet, testSet


def augmentImages(aug_dict, img, input_size):
    if 'width_shift_range' in aug_dict:
        input_cropx = r.sample(aug_dict['width_shift_range'], 1)[0]
    else:
        input_cropx = 0
    if 'height_shift_range' in aug_dict:
        input_cropy = r.sample(aug_dict['height_shift_range'], 1)[0]
    else:
        input_cropy = 0
    if 'rotation_range' in aug_dict:
        rotation = r.sample(aug_dict['rotation_range'], 1)[0]
    else:
        rotation = 0
    if 'horizontal_flip' in aug_dict and aug_dict['horizontal_flip']:
        do_horizontal_flip = r.sample([False,True], 1)[0]
    else:
        do_horizontal_flip = False
    if 'vertical_flip' in aug_dict and aug_dict['vertical_flip']:
        do_vertical_flip = r.sample([False, True], 1)[0]
    else:
        do_vertical_flip = False

    img = img[input_cropy:input_cropy+input_size[0], input_cropx:input_cropx+input_size[1]]
    if rotation:
        img = trans.rotate(img, rotation)
    if do_horizontal_flip:
        img = img[:, ::-1]
    if do_vertical_flip:
        img = img[::-1, :]

    return img


def do_center_crop(img, input_size, inicial_size):

    input_cropx = int((inicial_size[1] - input_size[1])/2)
    input_cropy = int((160+188)/2)

    img = img[input_cropy:input_cropy+input_size[0], input_cropx:input_cropx+input_size[1]]

    return img


def trainGeneratorStones(classes_structure, classLevel, batch_size, datasetPath, trainSet, aug_dict, input_size=(128, 128, 3)):
    # i = 0
    classes_list_L0 = classes_structure["classes_list_L0"]
    classes_list_L1 = classes_structure["classes_list_L1"]
    classes_list_L2 = classes_structure["classes_list_L2"]
    classes_L0_dict = dict(zip(classes_list_L0, list(range(0,len(classes_list_L0)))))
    classes_L1_dict = dict(zip(classes_list_L1, list(range(0,len(classes_list_L1)))))
    classes_L2_dict = dict(zip(classes_list_L2, list(range(0,len(classes_list_L2)))))

    classes_hierarchy_L0L1 = classes_structure["classes_hierarchy_L0L1"]
    classes_hierarchy_L1L2 = classes_structure["classes_hierarchy_L1L2"]

    if classLevel == 0:
        class_count = len(classes_list_L0)
    elif classLevel == 1:
        class_count = len(classes_list_L1)
    else:
        class_count = len(classes_list_L2)

    if batch_size > 1:
        while 1:
            iRock = 0
            nBatches = int(np.ceil(len(trainSet)/batch_size))
            for batchID in range(nBatches):
                batch_images = np.zeros(((batch_size,) + input_size ))
                batch_classes = np.zeros((batch_size, class_count))

                iRockInBatch=0
                while iRockInBatch<batch_size:
                    if iRock < len(trainSet):
                        rock = trainSet[iRock]
                        img = io.imread(os.path.join(datasetPath, rock['Diretorio Img']))
                        img = normalizeImage(img)
                        class_Name = rock['Classe']
                        class_ID = classes_L0_dict[class_Name]
                        if classLevel>0:
                            superClassName= classes_hierarchy_L0L1[class_Name]
                            superClassID = classes_L1_dict[superClassName]
                            class_Name = superClassName
                            class_ID = superClassID
                            if classLevel>1:
                                superClassName = classes_hierarchy_L1L2[class_Name]
                                superClassID = classes_L2_dict[superClassName]
                                class_Name = superClassName
                                class_ID = superClassID
                        batch_classes[iRockInBatch][class_ID] = 1
                        img = augmentImages(aug_dict, img, input_size)
                        batch_images[iRockInBatch, :, :, :] = img

                        iRock += 1
                        iRockInBatch += 1
                    else:
                        batch_images = batch_images[0:iRockInBatch, :, :, :]
                        batch_classes = batch_classes[0:iRockInBatch]
                        break
                yield (batch_images, batch_classes)
    else:
        while 1:
            for rock in trainSet:
                img = io.imread(os.path.join(datasetPath, rock['Diretorio Img']))
                img = normalizeImage(img)
                class_Name = rock['Classe']
                class_ID = classes_L0_dict[class_Name]
                if classLevel > 0:
                    superClassName = classes_hierarchy_L0L1[class_Name]
                    superClassID = classes_L1_dict[superClassName]
                    class_Name = superClassName
                    class_ID = superClassID
                    if classLevel > 1:
                        superClassName = classes_hierarchy_L1L2[class_Name]
                        superClassID = classes_L2_dict[superClassName]
                        class_Name = superClassName
                        class_ID = superClassID
                img = augmentImages(aug_dict, img, input_size)
                img = np.zeros(input_size)

                cls = np.zeros(class_count)
                cls[class_ID] = 1
                img = np.array([img])

                yield (img, cls)


def testGeneratorStones(datasetPath, testSet, input_size, inicial_size):
    for rock in testSet:
        img = io.imread(os.path.join(datasetPath, rock['Diretorio Img']))

        img = do_center_crop(img, input_size, inicial_size)
        img = np.array([img])

        yield (img)


def get_class_gt(set, classes_structure, classLevel):
    class_gt = np.zeros((len(set),), 'int64')

    classes_list_L0 = classes_structure["classes_list_L0"]
    classes_list_L1 = classes_structure["classes_list_L1"]
    classes_list_L2 = classes_structure["classes_list_L2"]
    classes_L0_dict = dict(zip(classes_list_L0, list(range(0, len(classes_list_L0)))))
    classes_L1_dict = dict(zip(classes_list_L1, list(range(0, len(classes_list_L1)))))
    classes_L2_dict = dict(zip(classes_list_L2, list(range(0, len(classes_list_L2)))))

    classes_hierarchy_L0L1 = classes_structure["classes_hierarchy_L0L1"]
    classes_hierarchy_L1L2 = classes_structure["classes_hierarchy_L1L2"]

    i=0
    for rock in set:
        class_Name = rock['Classe']
        class_ID = classes_L0_dict[class_Name]
        if classLevel > 0:
            superClassName = classes_hierarchy_L0L1[class_Name]
            superClassID = classes_L1_dict[superClassName]
            class_Name = superClassName
            class_ID = superClassID
            if classLevel > 1:
                superClassName = classes_hierarchy_L1L2[class_Name]
                superClassID = classes_L2_dict[superClassName]
                class_Name = superClassName
                class_ID = superClassID
        class_gt[i] = int(class_ID)
        i=i+1
    return class_gt
