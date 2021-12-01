from data import *
import matplotlib.pyplot as plt
from sklearn import metrics

model_i = 0
# jbdm_v0 mobilenet_v2 mobilenet_v3 densenet nasnet efficientnet
# resnet_v2 inception_v3 inception_resnet_v2 vgg16 vgg19
modelFileNames = [
    ["DenseNet201_512_Ndensenet_L2_B16_E100_LR0.001_20211201-125101.hdf5", 'densenet', 2],
    ["DenseNet201_512_Ndensenet_L1_B16_E100_LR0.001_20211201-012147.hdf5", 'densenet', 1],
    ["DenseNet201_512_Ndensenet_L0_B16_E100_LR0.001_20211105-185037.hdf5", 'densenet', 0], # 0.662830
    ["DenseNet121_512_Ndensenet_L0_B16_E100_LR0.001_20211106-220536.hdf5", 'densenet', 0],
    ["DenseNet169_512_Ndensenet_L0_B16_E100_LR0.001_20211106-200411.hdf5", 'densenet', 0],
    # ["retrainDenseNet121_512_Ndensenet_L0_B16_E100_LR0.001_20211106-220536.hdf5_E100_20211108-014059.hdf5", 'densenet', 0],
    # ["retrainDenseNet121_512_Ndensenet_L0_B16_E100_LR0.001_20211106-220536.hdf5with_E100_LR1e-06_20211108-021041.hdf5", 'densenet', 0], # 0.6616
    ["EfficientNetB0_512_Nefficientnet_L0_B16_E100_LR0.001_20211104-233513.hdf5", 'efficientnet', 0],
    ["EfficientNetB7_512_Nefficientnet_L0_B16_E100_LR0.001_20211105-004712.hdf5", 'efficientnet', 0],
    ["InceptionResNetV2_512_Ninception_resnet_v2_L0_B16_E100_LR0.001_20211105-232306.hdf5", 'inception_resnet_v2', 0],
    ["InceptionV3_512_Ninception_v3_L0_B16_E100_LR0.001_20211107-144638.hdf5", 'inception_v3', 0],
    ["jbdm_v0_L0_B32_E25.hdf5", 'jbdm_v0', 0],
    ["jbdm_v0_L1_B32_E25.hdf5", 'jbdm_v0', 1],
    ["jbdm_v0_L2_B32_E25.hdf5", 'jbdm_v0', 2],
    ["MobileNetV2_512_Nmobilenet_v2_L0_B16_E100_LR0.001_20211106-091528.hdf5", 'mobilenet_v2', 0],
    ["MobileNetV3Large_512_Nmobilenet_v3_L0_B16_E100_LR0.001_20211106-173348.hdf5", 'mobilenet_v3', 0],
    ["MobileNetV3Small_512_Nmobilenet_v3_L0_B16_E100_LR0.001_20211106-150559.hdf5", 'mobilenet_v3', 0],
    ["NASNetLarge_512_Nnasnet_L0_B16_E100_LR0.001_20211105-202544.hdf5", 'nasnet', 0],
    ["NASNetMobile_512_Nnasnet_L0_B16_E100_LR0.0001_20211105-154125.hdf5", 'nasnet', 0],
    ["NASNetMobile_512_Nnasnet_L0_B16_E100_LR0.001_20211105-141633.hdf5", 'nasnet', 0],
    ["ResNet152V2_512_Nresnet_v2_L0_B16_E100_LR0.001_20211107-125757.hdf5", 'resnet_v2', 0],
    ["VGG16_512_Nvgg16_L0_B16_E100_LR0.001_20211107-190135.hdf5", 'vgg16', 0],
    ["VGG19_512_Nvgg19_L0_B16_E100_LR0.001_20211107-163139.hdf5", 'vgg19', 0],
    ]

modelFileName = modelFileNames[model_i][0]
mod_normalization = modelFileNames[model_i][1]
classLevel = modelFileNames[model_i][2]

datasetPath = "rocks_db/"  # "D:"
resultsPath = "results/"
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

# load model
modelFilePath = os.path.join(modelsPath, modelFileName)
model = tf.keras.models.load_model(modelFilePath)

# test with model
testGene = testGeneratorStones(datasetPath, testSet, input_size=net_input_size, inicial_size=inicial_size, normalization=mod_normalization, augmentation=test_augmentation_args)
NTest = len(testSet)
y_test_predictions = model.predict(testGene, NTest, verbose=1)
# if mod_base == 'jbdm_v1':  # model 1 outputs 3 classifier activations
#     y_test_predictions = y_test_predictions[0]
y_test_predict = np.argmax(y_test_predictions, axis=-1)

# evaluate model
# jaccard = metrics.jaccard_score(y_test_gt, y_test_predict)
# print("Jaccard:", jaccard)
# dice = (2*jaccard)/(jaccard+1)
accuracy = metrics.accuracy_score(y_test_gt, y_test_predict)
print("Test set Accuracy:", accuracy)

confusionMatrix = metrics.confusion_matrix(
    y_test_gt,
    y_test_predict,
    labels=range(len(classes)))
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
print("Classification Report:")
print(classificationReport)
text_file = open(
    os.path.join(resultsPath, "classificationReport_" + modelFileNameWOExtension + ".txt"),
    "w")
text_file.write(classificationReport)
text_file.close()

prfs = metrics.precision_recall_fscore_support(
    y_test_gt,
    y_test_predict,
    labels=range(len(classes)))
print("Class\t(n)\tPrecision\tRecall\tF-score\tSupport")
for i in range(0, len(classes)):
    print("{}\t({})\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(classes[i], i, prfs[0][i], prfs[1][i], prfs[2][i], prfs[3][i]))

print("Test set Accuracy:", accuracy)

# show predictions for first images
def plot_image(predictions_array, true_label, image, imageID):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # plt.imshow(image[..., 0], cmap=plt.cm.binary)
    plt.imshow(image, cmap=plt.cm.binary)
    # plt.title(str(imageID))

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% \n({})".format(
        classes[predicted_label],
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
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    rock = testSet[i]
    image = io.imread(os.path.join(datasetPath, rock['Diretorio Img']))
    image = do_center_crop(image, net_input_size, inicial_size)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(y_test_predictions[i], y_test_gt[i], image, rock['ID'])
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(y_test_predictions[i], y_test_gt[i])
plt.tight_layout()
plt.show()
