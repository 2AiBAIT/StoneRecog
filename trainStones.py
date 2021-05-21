from model import *
from data import *
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# from classes_rocks import pedras_classes
from tensorflow.keras.callbacks import ModelCheckpoint

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

classLevel = 2
batch_size = 32
epochs = 25

datasetPath = "rocks_db/"
#datasetPath = "D:"
resultsPath = "results/predict/"
modelsPath = "models/"
modelFileName= "jbdm_v0_L" + str(classLevel) + "_B" + str(batch_size) + "_E" + str(epochs) + ".hdf5"
rocks_db = "rocks_db/rocks_db_corrected.json"
stone_classes_list_L0 = "stone_classes_list_L0.json"
stone_classes_list_L1 = "stone_classes_list_L1.json"
stone_classes_list_L2 = "stone_classes_list_L2.json"
stone_classes_hierarchy_L0L1 ="stone_classes_hierarchy_L0L1.json"
stone_classes_hierarchy_L1L2 ="stone_classes_hierarchy_L1L2.json"

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
input_size = (128, 128, 3)

trainSet, testSet = prepare_dataset(rocks_db=rocks_db, datasetPath=datasetPath, classes_structure=classes_structure, classLevel=classLevel)

full_augmentation_args = dict(
    width_shift_range=list(range(0, inicial_size[1] - input_size[1])),
    height_shift_range= [0] + list(range(160, 188)),
    rotation_range=[0, 90, 180, 270],
    horizontal_flip=True,
    vertical_flip=True
)
no_augmentation_args = dict()

augmentation_args = full_augmentation_args

class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.batch_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracies.append(logs.get('acc'))


# batch_history = BatchLossHistory()

if trainSize > 0:
    trainSet = trainSet[0:trainSize]
if testSize > 0:
    testSet = testSet[0:testSize]

Ntrain = len(trainSet)
steps_per_epoch = int(np.ceil(Ntrain/batch_size))

trainGene = trainGeneratorStones(classes_structure=classes_structure, classLevel=classLevel, batch_size=batch_size, datasetPath=datasetPath, trainSet=trainSet, aug_dict=augmentation_args, input_size=input_size)
valGene = trainGeneratorStones(classes_structure=classes_structure, classLevel=classLevel, batch_size=batch_size, datasetPath=datasetPath, trainSet=trainSet, aug_dict=no_augmentation_args, input_size=input_size)
NVal = len(testSet)
validation_steps = int(np.ceil(NVal/batch_size))
model = jbdm_v0(input_size=input_size, num_class=len(classes))
modelFilePath = os.path.join(modelsPath, modelFileName)
model_checkpoint = ModelCheckpoint(modelFilePath, monitor='val_loss', verbose=1, save_best_only=True)
r.seed(1)
history = model.fit_generator(
    trainGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint
        #, batch_history
        ],
    validation_data=valGene,
    validation_steps=validation_steps)

print("History:")
print(history.history)
# print("Batch history batch_losses:")
# print(batch_history.batch_losses)
# print("Batch history batch_accuracies:")
# print(batch_history.batch_accuracies)
