import tensorflow as tf
from tensorflow.python.keras.layers import Input

# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


class jbdm_v0(object):
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None, lr=1e-3):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_size),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(num_class, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Model summary")
        print(model.summary())

        if pretrained_weights:
            model.load_weights(pretrained_weights)
        return model


class SR_MobileNetV2():  # jbdm_v2_32():
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_MobileNetV3Small():  # jbdm_v2_5():
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        baseModel = tf.keras.applications.MobileNetV3Small(weights='imagenet',
                                                           include_top=False,
                                                           input_tensor=Input(shape=input_size)
                                                           )
        print("Base Model summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_MobileNetV3Large():  # jbdm_v2_7():
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        baseModel = tf.keras.applications.MobileNetV3Large(weights='imagenet',
                                                           include_top=False,
                                                           input_tensor=Input(shape=input_size)
                                                           )
        print("Base Model summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_InceptionResNetV2():  # jbdm_v3():
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet')
        baseModel = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_DenseNet201():  # jbdm_v4():  # DenseNet201
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.densenet.DenseNet201(weights='imagenet')
        baseModel = tf.keras.applications.densenet.DenseNet201(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_DenseNet169():  # jbdm_v4_1():  # DenseNet169
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.densenet.DenseNet169(weights='imagenet')
        baseModel = tf.keras.applications.densenet.DenseNet169(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_DenseNet121():  # jbdm_v4_2():  # DenseNet121
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.densenet.DenseNet121(weights='imagenet')
        if pretrained_weights is None:
            weights = 'imagenet'
        else:
            weights = None
        baseModel = tf.keras.applications.densenet.DenseNet121(weights=weights,
                                                               include_top=False,
                                                               input_tensor=Input(shape=input_size)
                                                               )
        print("Base Model summary")
        print(baseModel.summary())
        if not retrainAll:
            baseModel.trainable = False

        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_NASNetMobile():  # jbdm_v5(): # NASNetMobile
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.nasnet.NASNetMobile(weights='imagenet')
        # print("Base Mobile Model summary")
        # print(baseModel.summary())
        baseModel = tf.keras.applications.nasnet.NASNetMobile(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model no top summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_NASNetLarge():  # jbdm_v5_5(): # NASNetLarge
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.nasnet.NASNetLarge(weights='imagenet')
        # print("Base Mobile Model summary")
        # print(baseModel.summary())
        baseModel = tf.keras.applications.nasnet.NASNetLarge(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model no top summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_EfficientNetB0():  # jbdm_v6(): # EfficientNetB0
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        baseModel = tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet')
        print("Base Model summary")
        print(baseModel.summary())

        baseModel = tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model no top summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_EfficientNetB7():  # jbdm_v6_7(): # EfficientNetB7
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.efficientnet.EfficientNetB7(weights='imagenet')
        # print("Base Model summary")
        # print(baseModel.summary())
        baseModel = tf.keras.applications.efficientnet.EfficientNetB7(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model no top summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_ResNet152V2():  #
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.ResNet152V2(weights='imagenet')
        # print("Base Model summary")
        # print(baseModel.summary())
        baseModel = tf.keras.applications.ResNet152V2(weights='imagenet',
                                                      include_top=False,
                                                      input_tensor=Input(shape=input_size)
                                                      )
        print("Base Model no top summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_InceptionV3():  #
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
        # print("Base Model summary")
        # print(baseModel.summary())
        baseModel = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet',
                                                      include_top=False,
                                                      input_tensor=Input(shape=input_size)
                                                      )
        print("Base Model no top summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_VGG19():  #
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.vgg19.VGG19(weights='imagenet')
        # print("Base Model summary")
        # print(baseModel.summary())
        baseModel = tf.keras.applications.vgg19.VGG19(weights='imagenet',
                                                      include_top=False,
                                                      input_tensor=Input(shape=input_size)
                                                      )
        print("Base Model no top summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_VGG16():  #
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, dropout=0, pretrained_weights=None, lr=1e-3, retrainAll=False):
        # baseModel = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        # print("Base Model summary")
        # print(baseModel.summary())
        baseModel = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                                      include_top=False,
                                                      input_tensor=Input(shape=input_size)
                                                      )
        print("Base Model no top summary")
        print(baseModel.summary())
        if retrainAll:
            baseModel.trainable = True
        else:
            baseModel.trainable = False
        # baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        if classifierLayer > 0:
            if dropout > 0:
                base_output = tf.keras.layers.Dropout(dropout)(base_output)
            base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model
