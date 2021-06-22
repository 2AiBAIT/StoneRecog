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
mod_ver = 1

datasetPath = "rocks_db/"
#datasetPath = "D:"
resultsPath = "results/predict/"
modelsPath = "models/"
modelFileName = "jbdm_v" + str(mod_ver) + "_L" + str(classLevel) + "_B" + str(batch_size) + "_E" + str(epochs) + ".hdf5"
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
        self.batch_accuracies.append(logs.get('accuracy'))


# batch_history = BatchLossHistory()


class MetricsValues(tf.keras.callbacks.Callback):
    def __init__(self, figsize=None):
        super(MetricsValues, self).__init__()
        self.figsize = figsize

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        # self.fig = plt.figure()

        self.logs = []

    def on_batch_end(self, batch, logs=None):
        self.i = self.i

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        #self.acc.append(logs.get('accuracy'))
        self.acc.append(logs.get('prob_accuracy'))
        #self.val_acc.append(logs.get('val_accuracy'))
        self.val_acc.append(logs.get('val_prob_accuracy'))
        self.i += 1

    def on_train_end(self, logs):
        clear_output(wait=True)
        f, (gra1, gra2) = plt.subplots(1, 2, figsize=self.figsize)

        # gra1.set_yscale('log')
        gra1.plot(self.x, self.losses, label="loss")
        gra1.plot(self.x, self.val_losses, label="val_loss")
        gra1.title.set_text('Model Loss')
        gra1.set_xlabel('Epoch')
        gra1.set_ylabel('Loss')
        # gra1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        gra1.legend(['training', 'validation'], loc='upper left')

        gra2.plot(self.x, self.acc, label="accuracy")
        gra2.plot(self.x, self.val_acc, label="val_accuracy")
        gra2.title.set_text('Model Accuracy')
        gra2.set_xlabel('Epoch')
        gra2.set_ylabel('Accuracy')
        gra2.legend(['training', 'validation'], loc='upper left')


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
model = jbdm_v1.build(input_size=input_size, num_class=len(classes))
modelFilePath = os.path.join(modelsPath, modelFileName)
model_checkpoint = ModelCheckpoint(modelFilePath, monitor='val_loss', verbose=1, save_best_only=True)
r.seed(1)

logdir = "logdir\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
metrics = "grafs\\" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_v" + str(mod_ver) + "_L" + str(classLevel) + "_B"\
          + str(batch_size) + "_E" + str(epochs) + ".png"

plot = MetricsValues(figsize=(12, 6))

training_history = model.fit(trainGene,
                             steps_per_epoch=steps_per_epoch,
                             epochs=epochs,
                             callbacks=[model_checkpoint, batch_history, plot],
                             validation_data=valGene,
                             validation_steps=validation_steps)

print("History:")
print(history.history)
# print("Batch history batch_losses:")
# print(batch_history.batch_losses)
# print("Batch history batch_accuracies:")
# print(batch_history.batch_accuracies)
print("Keys:")
print(training_history.history.keys())

plt.savefig(metrics)
plt.show()

print(max(training_history.history))
print("completed")
