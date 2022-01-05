import os
import matplotlib.pyplot as plt
import json
import numpy as np
from tensorflow.keras.models import load_model
import skimage.io as io
from sklearn import metrics
from stoneGraphics import plot_image, plot_value_array
from data import prepare_dataset, get_class_gt, do_center_crop, testGeneratorStones

show_graphics = True
verbose = 0
classLevel = 2

# jbdm_v0 mobilenet_v2 mobilenet_v3 densenet nasnet efficientnet
# resnet_v2 inception_v3 inception_resnet_v2 vgg16 vgg19

modelFileNamesL2 = [
    ["L2_DenseNet121_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211204-211415.hdf5", 'densenet', 2],
    # ["L2_DenseNet169_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211204-224302.hdf5", 'densenet', 2],
    ["L2_DenseNet201_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211205-001437.hdf5", 'densenet', 2],
    # ["L2_EfficientNetB0_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211205-084702.hdf5", 'efficientnet', 2],
    # ["L2_EfficientNetB7_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211205-094835.hdf5", 'efficientnet', 2],
    # ["L2_jbdm_v0_CL512_DO0_Njbdm_v0_B16_E100_LR0.0001_20211204-162639.hdf5", 'jbdm_v0', 2],
    # ["L2_InceptionResNetV2_CL512_DO0_Ninception_resnet_v2_B16_E100_LR0.0001_20211205-145402.hdf5", 'inception_resnet_v2', 2],
    # ["L2_InceptionV3_CL512_DO0_Ninception_v3_B16_E100_LR0.0001_20211205-133228.hdf5", 'inception_v3', 2],
    # ["L2_MobileNetV2_CL512_DO0_Nmobilenet_v2_B16_E100_LR0.0001_20211205-015537.hdf5", 'mobilenet_v2', 2],
    # ["L2_MobileNetV3Small_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211205-030822.hdf5", 'mobilenet_v3', 2],
    # ["L2_MobileNetV3Large_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211205-040406.hdf5", 'mobilenet_v3', 2],
    # ["L2_NASNetMobile_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211205-050128.hdf5", 'nasnet', 2],
    # ["L2_NASNetLarge_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211205-062446.hdf5", 'nasnet', 2],
    # ["L2_ResNet152V2_CL512_DO0_Nresnet_v2_B16_E100_LR0.0001_20211205-115507.hdf5", 'resnet_v2', 2],
    # ["L2_VGG16_CL512_DO0_Nvgg16_B16_E100_LR0.0001_20211204-183702.hdf5", 'vgg16', 2],
    # ["L2_VGG19_CL512_DO0_Nvgg19_B16_E100_LR0.0001_20211204-195406.hdf5", 'vgg19', 2],
    ]
modelFileNamesL1 = [
    ["L1_DenseNet121_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211203-214746.hdf5", 'densenet', 1],
    # ["L1_DenseNet169_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211203-231105.hdf5", 'densenet', 1],
    ["L1_DenseNet201_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211204-003808.hdf5", 'densenet', 1],
    # ["L1_EfficientNetB0_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211204-085413.hdf5", 'efficientnet', 1],
    # ["L1_EfficientNetB7_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211204-095036.hdf5", 'efficientnet', 1],
    # ["L1_jbdm_v0_CL512_DO0_Njbdm_v0_B16_E100_LR0.0001_20211203-171759.hdf5", 'jbdm_v0', 1],
    # ["L1_InceptionResNetV2_CL512_DO0_Ninception_resnet_v2_B16_E100_LR0.0001_20211204-145041.hdf5", 'inception_resnet_v2', 1],
    # ["L1_InceptionV3_CL512_DO0_Ninception_v3_B16_E100_LR0.0001_20211204-133214.hdf5", 'inception_v3', 1],
    # ["L1_MobileNetV2_CL512_DO0_Nmobilenet_v2_B16_E100_LR0.0001_20211204-021124.hdf5", 'mobilenet_v2', 1],
    # ["L1_MobileNetV3Small_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211204-032016.hdf5", 'mobilenet_v3', 1],
    # ["L1_MobileNetV3Large_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211204-041310.hdf5", 'mobilenet_v3', 1],
    # ["L1_NASNetMobile_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211204-050651.hdf5", 'nasnet', 1],
    # ["L1_NASNetLarge_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211204-062539.hdf5", 'nasnet', 1],
    # ["L1_ResNet152V2_CL512_DO0_Nresnet_v2_B16_E100_LR0.0001_20211204-115651.hdf5", 'resnet_v2', 1],
    # ["L1_VGG16_CL512_DO0_Nvgg16_B16_E100_LR0.0001_20211203-191753.hdf5", 'vgg16', 1],
    # ["L1_VGG19_CL512_DO0_Nvgg19_B16_E100_LR0.0001_20211203-203134.hdf5", 'vgg19', 1],
    ]
modelFileNamesL0 = [
    ["L0_DenseNet121_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211202-035253.hdf5", 'densenet', 0],
    # ["L0_DenseNet169_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211202-051755.hdf5", 'densenet', 0],
    ["L0_DenseNet201_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211202-064528.hdf5", 'densenet', 0],
    # ["L0_EfficientNetB0_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211205-213422.hdf5", 'efficientnet', 0],
    # ["L0_EfficientNetB7_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211205-222732.hdf5", 'efficientnet', 0],
    # ["L0_jbdm_v0_CL512_DO0_Njbdm_v0_B16_E100_LR0.0001_20211202-165845.hdf5", 'jbdm_v0', 0],
    # ["L0_InceptionResNetV2_CL512_DO0_Ninception_resnet_v2_B16_E100_LR0.0001_20211206-031726.hdf5", 'inception_resnet_v2', 0],
    # ["L0_InceptionV3_CL512_DO0_Ninception_v3_B16_E100_LR0.0001_20211206-020354.hdf5", 'inception_v3', 0],
    # ["L0_MobileNetV2_CL512_DO0_Nmobilenet_v2_B16_E100_LR0.0001_20211202-190234.hdf5", 'mobilenet_v2', 0],
    # ["L0_MobileNetV3Small_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211202-201145.hdf5", 'mobilenet_v3', 0],
    # ["L0_MobileNetV3Large_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211202-210142.hdf5", 'mobilenet_v3', 0],
    # ["L0_NASNetMobile_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211202-215247.hdf5", 'nasnet', 0],
    # ["L0_NASNetLarge_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211203-000721.hdf5", 'nasnet', 0],
    # ["L0_ResNet152V2_CL512_DO0_Nresnet_v2_B16_E100_LR0.0001_20211206-003014.hdf5", 'resnet_v2', 0],
    # ["L0_VGG16_CL512_DO0_Nvgg16_B16_E100_LR0.0001_20211202-012130.hdf5", 'vgg16', 0],
    # ["L0_VGG19_CL512_DO0_Nvgg19_B16_E100_LR0.0001_20211202-023634.hdf5", 'vgg19', 0],
    ]

if classLevel == 0:
    modelFileNames = modelFileNamesL0
elif classLevel == 1:
    modelFileNames = modelFileNamesL1
elif classLevel == 2:
    modelFileNames = modelFileNamesL2

datasetPath = "rocks_db/"  # "D:"
resultsPath = "results/"
modelsPath = "models/" + "L" + str(classLevel)
rocks_db = "rocks_db/rocks_db_corrected.json"
stone_classes_list_L0 = "stone_classes_list_L0.json"
stone_classes_list_L1 = "stone_classes_list_L1.json"
stone_classes_list_L2 = "stone_classes_list_L2.json"
stone_classes_hierarchy_L0L1 = "stone_classes_hierarchy_L0L1.json"
stone_classes_hierarchy_L1L2 = "stone_classes_hierarchy_L1L2.json"

with open(stone_classes_list_L0) as data_file:
    classes_list_L0 = json.load(data_file)
with open(stone_classes_list_L1) as data_file:
    classes_list_L1 = json.load(data_file)
with open(stone_classes_list_L2) as data_file:
    classes_list_L2 = json.load(data_file)

with open(stone_classes_hierarchy_L0L1) as data_file:
    classes_hierarchy_L0L1 = json.load(data_file)
with open(stone_classes_hierarchy_L1L2) as data_file:
    classes_hierarchy_L1L2 = json.load(data_file)

classes_structure = dict()
classes_structure["classes_list_L0"] = classes_list_L0
classes_structure["classes_list_L1"] = classes_list_L1
classes_structure["classes_list_L2"] = classes_list_L2
classes_structure["classes_hierarchy_L0L1"] = classes_hierarchy_L0L1
classes_structure["classes_hierarchy_L1L2"] = classes_hierarchy_L1L2
classes_L0_dict = dict(zip(classes_list_L0, list(range(0, len(classes_list_L0)))))
classes_L1_dict = dict(zip(classes_list_L1, list(range(0, len(classes_list_L1)))))
classes_L2_dict = dict(zip(classes_list_L2, list(range(0, len(classes_list_L2)))))

if classLevel == 0:
    classes = classes_list_L0
elif classLevel == 1:
    classes = classes_list_L1
else:
    classes = classes_list_L2

inicial_size = (480, 780, 3)
net_input_size = (128, 128, 3)

trainSet, testSet = prepare_dataset(
    rocks_db=rocks_db,
    datasetPath=datasetPath,
    classes_structure=classes_structure,
    classLevel=classLevel)

test_augmentation_args = dict()

y_test_gt = get_class_gt(testSet, classes_structure, classLevel)

if show_graphics:
    plt.figure(figsize=(10, 10))
    plt.title("Stone Recog Dataset")
    num_rows = 2
    num_cols = 4
    num_images = num_rows * num_cols
    for i in range(num_images):
        rock = testSet[i]
        image = io.imread(os.path.join(datasetPath, rock['Diretorio Img']))
        image = do_center_crop(image, net_input_size, inicial_size)
        class_Name = rock['Classe']
        classNameFull = str(rock["ID"]) + " - " + class_Name
        class_ID = classes_L0_dict[class_Name]
        if classLevel > 0:
            superClassName = classes_hierarchy_L0L1[class_Name]
            superClassID = classes_L1_dict[superClassName]
            class_Name = superClassName
            # classNameFull = classNameFull + " - " + class_Name
            class_ID = superClassID
            if classLevel > 1:
                superClassName = classes_hierarchy_L1L2[class_Name]
                superClassID = classes_L2_dict[superClassName]
                class_Name = superClassName
                # classNameFull = classNameFull + " - " + class_Name
                class_ID = superClassID
        plt.subplot(num_rows, num_cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        l0class = rock['Classe']
        l1class = classes_hierarchy_L0L1[l0class]
        l2class = classes_hierarchy_L1L2[l1class]
        plt.xlabel(l0class + "\n" + l1class + "\n" + l2class + "\n" + rock["Pais"] + "\n" + rock["Densidade aparente"])
        plt.title(rock["Nome da Pedra"])
    plt.tight_layout()
    plt.show()

for model_i in range(len(modelFileNames)):
    modelFileName = modelFileNames[model_i][0]
    mod_normalization = modelFileNames[model_i][1]
    classLevel = modelFileNames[model_i][2]

    # load model
    modelFilePath = os.path.join(modelsPath, modelFileName)
    model = load_model(modelFilePath)

    # test with model
    testGene = testGeneratorStones(datasetPath, testSet, input_size=net_input_size, inicial_size=inicial_size, normalization=mod_normalization, augmentation=test_augmentation_args)

    y_test_predictions = model.predict(testGene)
    # if mod_base == 'jbdm_v1':  # model 1 outputs 3 classifier activations
    #     y_test_predictions = y_test_predictions[0]
    y_test_predict = np.argmax(y_test_predictions, axis=-1)

    # evaluate model
    # jaccard = metrics.jaccard_score(y_test_gt, y_test_predict)
    # print("Jaccard:", jaccard)
    # dice = (2*jaccard)/(jaccard+1)
    accuracy = metrics.accuracy_score(y_test_gt, y_test_predict)
    print("Model: {}".format(modelFileName))
    print(" Test set Accuracy:", accuracy)

    confusionMatrix = metrics.confusion_matrix(
        y_test_gt,
        y_test_predict,
        labels=range(len(classes)))
    if verbose == 2:
        print("Confusion matrix:")
        print(confusionMatrix)
    modelFileNameWOExtension, modelFileNameExtension = os.path.splitext(modelFileName)
    np.savetxt(
        os.path.join(resultsPath, "confusionMatrix_" + modelFileNameWOExtension + ".csv"),
        confusionMatrix,
        delimiter=' ')

    classificationReport = metrics.classification_report(
        y_test_gt,
        y_test_predict,
        labels=range(len(classes)))
    if verbose == 2:
        print("Classification Report:")
        print(classificationReport)
    text_file = open(
        os.path.join(resultsPath, "classificationReport_" + modelFileNameWOExtension + ".txt"),
        "w")
    text_file.write(classificationReport)
    text_file.close()

    if verbose == 2:
        prfs = metrics.precision_recall_fscore_support(
            y_test_gt,
            y_test_predict,
            labels=range(len(classes)))
        print("Class\t(n)\tPrecision\tRecall\tF-score\tSupport")
        for i in range(0, len(classes)):
            print("{}\t({})\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(classes[i], i, prfs[0][i], prfs[1][i], prfs[2][i], prfs[3][i]))

    if show_graphics:  # show predictions for first images
        num_rows = 4
        num_cols = 4
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            rock = testSet[i]
            image = io.imread(os.path.join(datasetPath, rock['Diretorio Img']))
            image = do_center_crop(image, net_input_size, inicial_size)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            plot_image(y_test_predictions[i], y_test_gt[i], classes, image, rock['ID'])
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot_value_array(y_test_predictions[i], y_test_gt[i])
        plt.tight_layout()
        plt.show()
