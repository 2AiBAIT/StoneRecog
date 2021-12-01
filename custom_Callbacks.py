import  tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt


class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.batch_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracies.append(logs.get('accuracy'))


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
