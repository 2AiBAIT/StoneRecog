from model import *
from data import *
import matplotlib.pyplot as plt
from sklearn import metrics

classLevel = 2
batch_size = 32
epochs = 50
mod_ver = 1

datasetPath = "rocks_db/" #"D:"
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
classes_L0_dict = dict(zip(classes_list_L0, list(range(0, len(classes_list_L0)))))
classes_L1_dict = dict(zip(classes_list_L1, list(range(0, len(classes_list_L1)))))
classes_L2_dict = dict(zip(classes_list_L2, list(range(0, len(classes_list_L2)))))

if classLevel==0:
    classes=classes_list_L0
elif classLevel==1:
    classes=classes_list_L1
else:
    classes=classes_list_L2
    
inicial_size = (480, 780, 3)
input_size = (128, 128, 3)

trainSet, testSet = prepare_dataset(rocks_db=rocks_db, datasetPath=datasetPath, classes_structure=classes_structure, classLevel=classLevel)
y_test_gt=get_class_gt(testSet, classes_structure, classLevel)

plt.figure(figsize=(10,10))
plt.title("Stone Recog Dataset")
num_rows = 2
num_cols = 4
num_images = num_rows*num_cols
for i in range(num_images):
    rock = testSet[i]
    image = io.imread(os.path.join(datasetPath, rock['Diretorio Img']))
    image=do_center_crop(image, input_size, inicial_size)
    class_Name = rock['Classe']
    classNameFull=str(rock["ID"]) + " - " + class_Name
    class_ID = classes_L0_dict[class_Name]
    if classLevel > 0:
        superClassName = classes_hierarchy_L0L1[class_Name]
        superClassID = classes_L1_dict[superClassName]
        class_Name = superClassName
        #classNameFull = classNameFull + " - " + class_Name
        class_ID = superClassID
        if classLevel > 1:
            superClassName = classes_hierarchy_L1L2[class_Name]
            superClassID = classes_L2_dict[superClassName]
            class_Name = superClassName
            #classNameFull = classNameFull + " - " + class_Name
            class_ID = superClassID
    plt.subplot(num_rows,num_cols,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    l0class=rock['Classe']
    l1class = classes_hierarchy_L0L1[l0class]
    l2class = classes_hierarchy_L1L2[l1class]
    plt.xlabel(l0class + "\n" + l1class + "\n" + l2class + "\n" + rock["Pais"] + "\n" + rock["Densidade aparente"])
    plt.title(rock["Nome da Pedra"])
plt.tight_layout()
plt.show()

#load model
modelFilePath = os.path.join(modelsPath, modelFileName)
model = jbdm_v0(input_size=input_size, num_class=len(classes), pretrained_weights=modelFilePath)

#test with model
testGene = testGeneratorStones(datasetPath, testSet, input_size=input_size, inicial_size=inicial_size)
NTest = len(testSet)
y_test_predictions = model.predict_generator(testGene, NTest, verbose=1)
y_test_predictions = y_test_predictions[2]
y_test_predict = np.argmax(y_test_predictions, axis=-1)

# show predictions for first images
def plot_image(predictions_array, true_label, image):

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    #plt.imshow(image[..., 0], cmap=plt.cm.binary)
    plt.imshow(image, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% \n({})".format(classes[predicted_label],
                                           100 * np.max(predictions_array),
                                           classes[true_label]),
               color=color)


def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    rock = testSet[i]
    image = io.imread(os.path.join(datasetPath, rock['Diretorio Img']))
    image = do_center_crop(image, input_size, inicial_size)
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(y_test_predictions[i], y_test_gt[i], image)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(y_test_predictions[i], y_test_gt[i])
plt.tight_layout()
plt.show()

# evaluate model
#jaccard = metrics.jaccard_score(y_test_gt, y_test_predict)
#dice = (2*jaccard)/(jaccard+1)
accuracy = metrics.accuracy_score(y_test_gt, y_test_predict)
prfs = metrics.precision_recall_fscore_support(y_test_gt, y_test_predict, labels=range(len(classes)))
confusionMatrix = metrics.confusion_matrix(y_test_gt, y_test_predict, labels=range(0, len(classes)))
np.savetxt("confusionMatrix" + "_L" + str(classLevel) + "_B" + str(batch_size) + "_E" + str(epochs) +  ".csv", confusionMatrix, delimiter=' ')
classificationReport=metrics.classification_report(y_test_gt, y_test_predict)

# print("Jaccard:", jaccard)
print("Test set Accuracy:", accuracy)
print("Confusion matrix:")
print(confusionMatrix)
print("Classification Report:")
print(classificationReport)

for i in range(0, len(classes)):
    print("Class", classes[i], "(", i, ") Precision, Recall, F-score, Support:", prfs[0][i], prfs[1][i], prfs[2][i], prfs[3][i])


