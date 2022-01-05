from data import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

# classLevels = [0, 1, 2]
classLevels = [0]
batch_size = 16
epochs = 100
lr = 1e-6

old_modelFileNamesL2 = [
    ["L2_DenseNet121_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211204-211415.hdf5", 'densenet', 2],
    ["L2_DenseNet169_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211204-224302.hdf5", 'densenet', 2],
    ["L2_DenseNet201_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211205-001437.hdf5", 'densenet', 2],
    ["L2_EfficientNetB0_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211205-084702.hdf5", 'efficientnet', 2],
    ["L2_EfficientNetB7_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211205-094835.hdf5", 'efficientnet', 2],
    ["L2_jbdm_v0_CL512_DO0_Njbdm_v0_B16_E100_LR0.0001_20211204-162639.hdf5", 'jbdm_v0', 2],
    ["L2_InceptionResNetV2_CL512_DO0_Ninception_resnet_v2_B16_E100_LR0.0001_20211205-145402.hdf5", 'inception_resnet_v2', 2],
    ["L2_InceptionV3_CL512_DO0_Ninception_v3_B16_E100_LR0.0001_20211205-133228.hdf5", 'inception_v3', 2],
    ["L2_MobileNetV2_CL512_DO0_Nmobilenet_v2_B16_E100_LR0.0001_20211205-015537.hdf5", 'mobilenet_v2', 2],
    ["L2_MobileNetV3Small_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211205-030822.hdf5", 'mobilenet_v3', 2],
    ["L2_MobileNetV3Large_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211205-040406.hdf5", 'mobilenet_v3', 2],
    ["L2_NASNetMobile_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211205-050128.hdf5", 'nasnet', 2],
    ["L2_NASNetLarge_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211205-062446.hdf5", 'nasnet', 2],
    ["L2_ResNet152V2_CL512_DO0_Nresnet_v2_B16_E100_LR0.0001_20211205-115507.hdf5", 'resnet_v2', 2],
    ["L2_VGG16_CL512_DO0_Nvgg16_B16_E100_LR0.0001_20211204-183702.hdf5", 'vgg16', 2],
    ["L2_VGG19_CL512_DO0_Nvgg19_B16_E100_LR0.0001_20211204-195406.hdf5", 'vgg19', 2],
    ]
old_modelFileNamesL1 = [
    ["L1_DenseNet121_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211203-214746.hdf5", 'densenet', 1],
    ["L1_DenseNet169_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211203-231105.hdf5", 'densenet', 1],
    ["L1_DenseNet201_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211204-003808.hdf5", 'densenet', 1],
    ["L1_EfficientNetB0_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211204-085413.hdf5", 'efficientnet', 1],
    ["L1_EfficientNetB7_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211204-095036.hdf5", 'efficientnet', 1],
    ["L1_jbdm_v0_CL512_DO0_Njbdm_v0_B16_E100_LR0.0001_20211203-171759.hdf5", 'jbdm_v0', 1],
    ["L1_InceptionResNetV2_CL512_DO0_Ninception_resnet_v2_B16_E100_LR0.0001_20211204-145041.hdf5", 'inception_resnet_v2', 1],
    ["L1_InceptionV3_CL512_DO0_Ninception_v3_B16_E100_LR0.0001_20211204-133214.hdf5", 'inception_v3', 1],
    ["L1_MobileNetV2_CL512_DO0_Nmobilenet_v2_B16_E100_LR0.0001_20211204-021124.hdf5", 'mobilenet_v2', 1],
    ["L1_MobileNetV3Small_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211204-032016.hdf5", 'mobilenet_v3', 1],
    ["L1_MobileNetV3Large_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211204-041310.hdf5", 'mobilenet_v3', 1],
    ["L1_NASNetMobile_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211204-050651.hdf5", 'nasnet', 1],
    ["L1_NASNetLarge_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211204-062539.hdf5", 'nasnet', 1],
    ["L1_ResNet152V2_CL512_DO0_Nresnet_v2_B16_E100_LR0.0001_20211204-115651.hdf5", 'resnet_v2', 1],
    ["L1_VGG16_CL512_DO0_Nvgg16_B16_E100_LR0.0001_20211203-191753.hdf5", 'vgg16', 1],
    ["L1_VGG19_CL512_DO0_Nvgg19_B16_E100_LR0.0001_20211203-203134.hdf5", 'vgg19', 1],
    ]
old_modelFileNamesL0 = [
    ["L0_DenseNet121_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211202-035253.hdf5", 'densenet', 0],
    ["L0_DenseNet169_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211202-051755.hdf5", 'densenet', 0],
    ["L0_DenseNet201_CL512_DO0_Ndensenet_B16_E100_LR0.0001_20211202-064528.hdf5", 'densenet', 0],
    ["L0_EfficientNetB0_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211205-213422.hdf5", 'efficientnet', 0],
    ["L0_EfficientNetB7_CL512_DO0_Nefficientnet_B16_E100_LR0.0001_20211205-222732.hdf5", 'efficientnet', 0],
    ["L0_jbdm_v0_CL512_DO0_Njbdm_v0_B16_E100_LR0.0001_20211202-165845.hdf5", 'jbdm_v0', 0],
    ["L0_InceptionResNetV2_CL512_DO0_Ninception_resnet_v2_B16_E100_LR0.0001_20211206-031726.hdf5", 'inception_resnet_v2', 0],
    ["L0_InceptionV3_CL512_DO0_Ninception_v3_B16_E100_LR0.0001_20211206-020354.hdf5", 'inception_v3', 0],
    ["L0_MobileNetV2_CL512_DO0_Nmobilenet_v2_B16_E100_LR0.0001_20211202-190234.hdf5", 'mobilenet_v2', 0],
    ["L0_MobileNetV3Small_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211202-201145.hdf5", 'mobilenet_v3', 0],
    ["L0_MobileNetV3Large_CL512_DO0_Nmobilenet_v3_B16_E100_LR0.0001_20211202-210142.hdf5", 'mobilenet_v3', 0],
    ["L0_NASNetMobile_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211202-215247.hdf5", 'nasnet', 0],
    ["L0_NASNetLarge_CL512_DO0_Nnasnet_B16_E100_LR0.0001_20211203-000721.hdf5", 'nasnet', 0],
    ["L0_ResNet152V2_CL512_DO0_Nresnet_v2_B16_E100_LR0.0001_20211206-003014.hdf5", 'resnet_v2', 0],
    ["L0_VGG16_CL512_DO0_Nvgg16_B16_E100_LR0.0001_20211202-012130.hdf5", 'vgg16', 0],
    ["L0_VGG19_CL512_DO0_Nvgg19_B16_E100_LR0.0001_20211202-023634.hdf5", 'vgg19', 0],
    ]

datasetPath = "rocks_db/"
resultsPath = "results/predict/"
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

classes_structure=dict()
classes_structure["classes_list_L0"]=classes_list_L0
classes_structure["classes_list_L1"]=classes_list_L1
classes_structure["classes_list_L2"]=classes_list_L2
classes_structure["classes_hierarchy_L0L1"]=classes_hierarchy_L0L1
classes_structure["classes_hierarchy_L1L2"]=classes_hierarchy_L1L2

trainSize = -1  # -1 for all
testSize = -1  # -1 for all
inicial_size = (480, 780, 3)
crop_size = (128, 128, 3)
# net_input_size = (224, 224, 3)
net_input_size = (128, 128, 3)

for classLevel in classLevels:
    if classLevel == 0:
        classes = classes_list_L0
    elif classLevel == 1:
        classes = classes_list_L1
    else:
        classes = classes_list_L2

    if classLevel == 0:
        old_modelFileNames = old_modelFileNamesL0
    elif classLevel == 1:
        old_modelFileNames = old_modelFileNamesL1
    else:
        old_modelFileNames = old_modelFileNamesL2

    trainSet, testSet = prepare_dataset(rocks_db=rocks_db, datasetPath=datasetPath, classes_structure=classes_structure, classLevel=classLevel)

    full_augmentation_args = dict(
        width_shift_range=list(range(0, inicial_size[1] - crop_size[1])),
        height_shift_range= [0] + list(range(160, 188)),
        rotation_range=[0, 90, 180, 270],
        horizontal_flip=True,
        vertical_flip=True
    )
    no_augmentation_args = dict()
    augmentation_args = full_augmentation_args
    train_augmentation_args = full_augmentation_args
    val_augmentation_args = no_augmentation_args

    if trainSize > 0:
        trainSet = trainSet[0:trainSize]
    if testSize > 0:
        testSet = testSet[0:testSize]

    Ntrain = len(trainSet)
    steps_per_epoch = int(np.ceil(Ntrain/batch_size))

    for old_model in old_modelFileNames:
        old_modelFileName = old_model[0]
        mod_normalization = old_model[1]

        modelsSubPath = "L" + str(classLevel)
        old_modelFilePath = os.path.join(modelsPath, modelsSubPath, old_modelFileName)
        new_model_base_name = old_modelFileName + '_retrainedWith' + '_E' + str(epochs) + '_LR' + str(lr) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        new_modelFileName = new_model_base_name + '.hdf5'

        trainGene = trainGeneratorStones(classes_structure=classes_structure, classLevel=classLevel, batch_size=batch_size, datasetPath=datasetPath, trainSet=trainSet, aug_dict=train_augmentation_args, crop_size=crop_size, net_input_size=net_input_size, normalization=mod_normalization)
        valGene = trainGeneratorStones(classes_structure=classes_structure, classLevel=classLevel, batch_size=batch_size, datasetPath=datasetPath, trainSet=testSet, aug_dict=val_augmentation_args, crop_size=crop_size, net_input_size=net_input_size, normalization=mod_normalization)
        NVal = len(testSet)
        validation_steps = int(np.ceil(NVal/batch_size))

        model = load_model(old_modelFilePath)
        model.trainable = True
        print(model.optimizer.learning_rate)
        model.optimizer.learning_rate.assign(lr)
        print(model.optimizer.learning_rate)
        # tf.keras.backend.set_value(model.optimizer.learning_rate, lr)

        new_modelFilePath = os.path.join(modelsPath, new_modelFileName)
        model_checkpoint = ModelCheckpoint(new_modelFilePath, monitor='val_loss', verbose=1, save_best_only=True)
        r.seed(1)

        log_dir = os.path.join("logs", "fit", "L" + str(classLevel), new_model_base_name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # class_weights = skl.utils.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        training_history = model.fit(trainGene,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=epochs,
                                     callbacks=[model_checkpoint,
                                                tensorboard_callback,
                                                # batch_history,
                                                # plot
                                                ],
                                     validation_data=valGene,
                                     validation_steps=validation_steps
                                     )
        print("Finished training {}".format(new_modelFileName))
print("completed")
