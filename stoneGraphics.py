import matplotlib.pyplot as plt
import numpy as np


def plot_image(predictions_array, true_label, classes, image, imageID):
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
