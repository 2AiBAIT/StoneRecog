from data import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

classLevel = 0
batch_size = 16
epochs = 100
# old_modelFileName = 'DenseNet121_512_Ndensenet_L0_B16_E100_LR0.001_20211106-220536.hdf5'
old_modelFileName = "DenseNet201_512_Ndensenet_L0_B16_E100_LR0.001_20211105-185037.hdf5"
mod_normalization = 'densenet'
lr = 1e-6

datasetPath = "rocks_db/"
resultsPath = "results/predict/"
modelsPath = "models/"

new_modelFileName = 'retrain' + old_modelFileName + 'with' + '_E' + str(epochs) + '_LR' + str(lr) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.hdf5'
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

if classLevel == 0:
    classes = classes_list_L0
elif classLevel == 1:
    classes = classes_list_L1
else:
    classes = classes_list_L2

trainSize = -1  # -1 for all
testSize = -1  # -1 for all
inicial_size = (480, 780, 3)
crop_size = (128, 128, 3)
# net_input_size = (224, 224, 3)
net_input_size = (128, 128, 3)

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

old_modelFilePath = os.path.join(modelsPath, old_modelFileName)
model = load_model(old_modelFilePath)
model.trainable = True
print(model.optimizer.learning_rate)
model.optimizer.learning_rate.assign(lr)
print(model.optimizer.learning_rate)
# tf.keras.backend.set_value(model.optimizer.learning_rate, lr)

new_modelFilePath = os.path.join(modelsPath, new_modelFileName)
model_checkpoint = ModelCheckpoint(new_modelFilePath, monitor='val_loss', verbose=1, save_best_only=True)
r.seed(1)

log_dir = os.path.join("logs", "fit", 'retrain' + old_modelFileName + 'with' + '_E' + str(epochs) + '_LR' + str(lr) + datetime.now().strftime("%Y%m%d-%H%M%S"))
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
print("completed")
