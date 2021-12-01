import os
import numpy as np
import json
import random as r
from datetime import datetime
from model import jbdm_v0, SR_DenseNet121, SR_DenseNet169, SR_DenseNet201, SR_VGG16, SR_VGG19, SR_MobileNetV2, SR_MobileNetV3Small, SR_MobileNetV3Large, SR_ResNet152V2, SR_InceptionV3, SR_InceptionResNetV2
from data import prepare_dataset, trainGeneratorStones
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
# from custom_Callbacks import BatchLossHistory, MetricsValues

classLevel = 2
batch_size = 16
epochs = 100
mod_base = 'DenseNet201'
mod_classifierLayer = 512
mod_normalization = 'densenet'
lr = 1e-3

modelFileName = mod_base + '_' + str(mod_classifierLayer) + '_N' + mod_normalization + "_L" + str(classLevel) + "_B" + str(batch_size) + "_E" + str(epochs) + "_LR" + str(lr) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".hdf5"

datasetPath = "rocks_db/"
# resultsPath = "results/"
modelsPath = "models/"
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
classes_structure["classes_list_L0"]=classes_list_L0
classes_structure["classes_list_L1"]=classes_list_L1
classes_structure["classes_list_L2"]=classes_list_L2
classes_structure["classes_hierarchy_L0L1"]=classes_hierarchy_L0L1
classes_structure["classes_hierarchy_L1L2"]=classes_hierarchy_L1L2

if classLevel == 0:
    classes = classes_list_L0
elif classLevel == 1:
    classes = classes_list_L1
else:
    classes = classes_list_L2

inicial_size = (480, 780, 3)
# net_input_size = (224, 224, 3)
net_input_size = (128, 128, 3)

crop_size = (128, 128, 3)
trainSize = -1  # -1 for all
testSize = -1  # -1 for all

trainSet, testSet = prepare_dataset(
    rocks_db=rocks_db,
    datasetPath=datasetPath,
    classes_structure=classes_structure,
    classLevel=classLevel)

full_augmentation_args = dict(
    width_shift_range=list(range(0, inicial_size[1] - crop_size[1])),
    height_shift_range= [0] + list(range(160, 188)),
    rotation_range=[0, 90, 180, 270],
    horizontal_flip=True,
    vertical_flip=True
)
# no_augmentation_args = dict()
augmentation_args = full_augmentation_args

if trainSize > 0:
    trainSet = trainSet[0:trainSize]
if testSize > 0:
    testSet = testSet[0:testSize]

Ntrain = len(trainSet)
steps_per_epoch = int(np.ceil(Ntrain/batch_size))

trainGene = trainGeneratorStones(classes_structure=classes_structure, classLevel=classLevel, batch_size=batch_size, datasetPath=datasetPath, trainSet=trainSet, aug_dict=augmentation_args, crop_size=crop_size, net_input_size=net_input_size, normalization=mod_normalization)
valGene = trainGeneratorStones(classes_structure=classes_structure, classLevel=classLevel, batch_size=batch_size, datasetPath=datasetPath, trainSet=trainSet, aug_dict=no_augmentation_args, crop_size=crop_size, net_input_size=net_input_size, normalization=mod_normalization)
NVal = len(testSet)
validation_steps = int(np.ceil(NVal/batch_size))

if mod_base == 'jbdm_v0':
    model = jbdm_v0.build(num_class=len(classes), input_size=net_input_size, lr=lr)
elif mod_base == "DenseNet121":
    model = SR_DenseNet121.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == "DenseNet169":
    model = SR_DenseNet169.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == "DenseNet201":
    model = SR_DenseNet201.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == 'MobileNetV2':
    model = SR_MobileNetV2.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == 'MobileNetV3Small':
    model = SR_MobileNetV3Small.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == "MobileNetV3Large":
    model = SR_MobileNetV3Large.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == 'VGG19':
    model = SR_VGG19.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == 'VGG16':
    model = SR_VGG16.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == 'ResNet152V2':
    model = SR_ResNet152V2.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == 'InceptionV3':
    model = SR_InceptionV3.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)
elif mod_base == 'InceptionResNetV2':
    model = SR_InceptionResNetV2.build(num_class=len(classes), input_size=net_input_size, classifierLayer=mod_classifierLayer, lr=lr)


modelFilePath = os.path.join(modelsPath, modelFileName)
model_checkpoint = ModelCheckpoint(modelFilePath, monitor='val_loss', verbose=1, save_best_only=True)
r.seed(1)

log_dir = os.path.join("logs", "fit", "L" + str(classLevel), mod_base + '_' + str(mod_classifierLayer) + '_N' + mod_normalization + "_L" + str(classLevel) + "_B" + str(batch_size) + "_E" + str(epochs) + "_LR" + str(lr) + "_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# batch_history = BatchLossHistory()
# plot_callback = MetricsValues(figsize=(12, 6))

# class_weights = skl.utils.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

training_history = model.fit(trainGene,
                             steps_per_epoch=steps_per_epoch,
                             epochs=epochs,
                             callbacks=[model_checkpoint,
                                        tensorboard_callback,
                                        # batch_history,
                                        # plot_callback
                                        ],
                             validation_data=valGene,
                             validation_steps=validation_steps
                             )

print("History:")
print(training_history.history)
# print("Batch history batch_losses:")
# print(batch_history.batch_losses)
# print("Batch history batch_accuracies:")
# print(batch_history.batch_accuracies)
print("Keys:")
print(training_history.history.keys())

# metrics = "grafs\\" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + mod_base + '_' + str(mod_classifierLayer) + '_N' + mod_normalization + "_L" + str(classLevel) + "_B" + str(batch_size) + "_E" + str(epochs) + ".png"
# plt.savefig(metrics)
# plt.show()

print(max(training_history.history))
print("completed")
