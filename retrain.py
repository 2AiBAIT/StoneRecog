from model import *
from data import *
# from trainStones import *


classLevel = 2
batch_size = 32
epochs = 5
mod_ver = 1

datasetPath = "rocks_db/"

inicial_size = (480, 780, 3)
input_size = (128, 128, 3)


modelsPath = "models/"
modelFileName = "jbdm_v" + str(mod_ver) + "_L" + str(classLevel) + "_B" + str(batch_size) + "_E" + str(epochs) + ".hdf5"
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

modelFilePath = os.path.join(modelsPath, modelFileName)
model = jbdm_v1.build(input_size=input_size, num_class=len(classes), pretrained_weights=modelFilePath)
model.summary()

# plot_model(model, to_file='Pedras.png')

model.trainable = False
base_output = model.layers[-2].output # layer number obtained from model summary above
new_output = tf.keras.layers.Dense(len(classes), activation="softmax")(base_output)
modelNew = tf.keras.models.Model(
    inputs=model.inputs, outputs=new_output)
modelNew.summary()

modelNew.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

trainGene = trainGeneratorStones(classes_structure=classes_structure, classLevel=classLevel, batch_size=batch_size, datasetPath=datasetPath, trainSet=trainSet, aug_dict=augmentation_args, input_size=input_size)

training_history = model.fit(trainGene,
                             steps_per_epoch=steps_per_epoch,
                             epochs=epochs,
                             callbacks=[model_checkpoint, batch_history, plot],
                             validation_data=valGene,
                             validation_steps=validation_steps)

print("History:")
print(training_history.history)
print("Keys:")
print(training_history.history.keys())
