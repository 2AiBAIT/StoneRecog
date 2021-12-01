import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout \
    , Flatten, Concatenate, Reshape, Activation
from tensorflow.python.keras.regularizers import l2
from lrn import LRN


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


class jbdm_v1(object):
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        putin = Input(shape=input_size)

        conv1_7x7_s2 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu',
                              name='conv1/7x7_s2', kernel_regularizer=l2(0.0002))(putin)
        pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1/3x3_s2')(conv1_7x7_s2)
        pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)
        conv2_3x3_reduce = Conv2D(64, kernel_size=(1, 1), padding='valid', activation='relu', name='conv2/3x3_reduce',
                                  kernel_regularizer=l2(0.0002))(pool1_norm1)
        conv2_3x3 = Conv2D(192, kernel_size=(3, 3), padding='same', activation='relu', name='conv2/3x3',
                           kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
        conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
        pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2/3x3_s2')(conv2_norm2)

        inception_3a_1x1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu', name='inception_3a/1x1',
                                  kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_3x3_reduce = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_3x3 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='inception_3a/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
        inception_3a_5x5_reduce = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_5x5 = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', name='inception_3a/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
        inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3a/pool')(
            pool2_3x3_s2)
        inception_3a_pool_proj = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
        inception_3a_output = Concatenate(axis=-1, name='inception_3a/output')(
            [inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj])

        inception_3b_1x1 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='inception_3b/1x1',
                                  kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_3x3_reduce = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_3b/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_3a_output)
        inception_3b_3x3 = Conv2D(192, kernel_size=(3, 3), padding='same', activation='relu', name='inception_3b/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
        inception_3b_5x5_reduce = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_3b/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_3a_output)
        inception_3b_5x5 = Conv2D(96, kernel_size=(5, 5), padding='same', activation='relu', name='inception_3b/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
        inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3b/pool')(
            inception_3a_output)
        inception_3b_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_3b/pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
        inception_3b_output = Concatenate(axis=-1, name='inception_3b/output')(
            [inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj])

        inception_4a_1x1 = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4a/1x1',
                                  kernel_regularizer=l2(0.0002))(inception_3b_output)
        inception_4a_3x3_reduce = Conv2D(96, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4a/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_3b_output)
        inception_4a_3x3 = Conv2D(208, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4a/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
        inception_4a_5x5_reduce = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4a/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_3b_output)
        inception_4a_5x5 = Conv2D(48, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4a/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
        inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4a/pool')(
            inception_3b_output)
        inception_4a_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_4a/pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
        inception_4a_output = Concatenate(axis=-1, name='inception_4a/output')(
            [inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj])

        loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss1/ave_pool')(inception_4a_output)
        loss1_conv = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='loss1/conv',
                            kernel_regularizer=l2(0.0002))(loss1_ave_pool)
        loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_conv)
        loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
        loss1_flatten = Flatten()(loss1_drop_fc)
        loss1_classifier = Dense(num_class, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_flatten)
        loss1_classifier_act = Activation('softmax')(loss1_classifier)

        inception_4b_1x1 = Conv2D(160, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4b/1x1',
                                  kernel_regularizer=l2(0.0002))(inception_4a_output)
        inception_4b_3x3_reduce = Conv2D(112, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4b/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4a_output)
        inception_4b_3x3 = Conv2D(224, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4b/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
        inception_4b_5x5_reduce = Conv2D(24, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4b/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4a_output)
        inception_4b_5x5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4b/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
        inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4b/pool')(
            inception_4a_output)
        inception_4b_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_4b/pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)
        inception_4b_output = Concatenate(axis=-1, name='inception_4b/output')(
            [inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj])

        inception_4c_1x1 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4c/1x1',
                                  kernel_regularizer=l2(0.0002))(inception_4b_output)
        inception_4c_3x3_reduce = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4c/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4b_output)
        inception_4c_3x3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4c/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
        inception_4c_5x5_reduce = Conv2D(24, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4c/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4b_output)
        inception_4c_5x5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4c/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
        inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4c/pool')(
            inception_4b_output)
        inception_4c_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_4c/pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)
        inception_4c_output = Concatenate(axis=-1, name='inception_4c/output')(
            [inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj])

        inception_4d_1x1 = Conv2D(112, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4d/1x1',
                                  kernel_regularizer=l2(0.0002))(inception_4c_output)
        inception_4d_3x3_reduce = Conv2D(144, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4d/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4c_output)
        inception_4d_3x3 = Conv2D(288, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4d/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
        inception_4d_5x5_reduce = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4d/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4c_output)
        inception_4d_5x5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4d/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
        inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4d/pool')(
            inception_4c_output)
        inception_4d_pool_proj = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_4d/pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)
        inception_4d_output = Concatenate(axis=-1, name='inception_4d/output')(
            [inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj])

        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d_output)
        loss2_conv = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', name='loss2/conv',
                            kernel_regularizer=l2(0.0002))(loss2_ave_pool)
        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_conv)
        loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
        loss2_flatten = Flatten()(loss2_drop_fc)
        loss2_classifier = Dense(num_class, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_flatten)
        loss2_classifier_act = Activation('softmax')(loss2_classifier)

        inception_4e_1x1 = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu', name='inception_4e/1x1',
                                  kernel_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_reduce = Conv2D(160, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4e/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4d_output)
        inception_4e_3x3 = Conv2D(320, kernel_size=(3, 3), padding='same', activation='relu', name='inception_4e/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
        inception_4e_5x5_reduce = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_4e/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4d_output)
        inception_4e_5x5 = Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', name='inception_4e/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
        inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4e/pool')(
            inception_4d_output)
        inception_4e_pool_proj = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_4e/pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)
        inception_4e_output = Concatenate(axis=-1, name='inception_4e/output')(
            [inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj])

        inception_5a_1x1 = Conv2D(256, kernel_size=(1, 1), padding='same', activation='relu', name='inception_5a/1x1',
                                  kernel_regularizer=l2(0.0002))(inception_4e_output)
        inception_5a_3x3_reduce = Conv2D(160, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_5a/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_4e_output)
        inception_5a_3x3 = Conv2D(320, kernel_size=(3, 3), padding='same', activation='relu', name='inception_5a/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_5a_3x3_reduce)
        inception_5a_5x5_reduce = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_5a/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_4e_output)
        inception_5a_5x5 = Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', name='inception_5a/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_5a_5x5_reduce)
        inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5a/pool')(
            inception_4e_output)
        inception_5a_pool_proj = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_5a/pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)
        inception_5a_output = Concatenate(axis=-1, name='inception_5a/output')(
            [inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj])

        inception_5b_1x1 = Conv2D(384, kernel_size=(1, 1), padding='same', activation='relu', name='inception_5b/1x1',
                                  kernel_regularizer=l2(0.0002))(inception_5a_output)
        inception_5b_3x3_reduce = Conv2D(192, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_5b/3x3_reduce', kernel_regularizer=l2(0.0002))(
            inception_5a_output)
        inception_5b_3x3 = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu', name='inception_5b/3x3',
                                  kernel_regularizer=l2(0.0002))(inception_5b_3x3_reduce)
        inception_5b_5x5_reduce = Conv2D(48, kernel_size=(1, 1), padding='same', activation='relu',
                                         name='inception_5b/5x5_reduce', kernel_regularizer=l2(0.0002))(
            inception_5a_output)
        inception_5b_5x5 = Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu', name='inception_5b/5x5',
                                  kernel_regularizer=l2(0.0002))(inception_5b_5x5_reduce)
        inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5b/pool')(
            inception_5a_output)
        inception_5b_pool_proj = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu',
                                        name='inception_5b/pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)
        inception_5b_output = Concatenate(axis=-1, name='inception_5b/output')(
            [inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj])

        pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')(inception_5b_output)
        pool5_drop_7x7_s1 = Dropout(rate=0.4)(pool5_7x7_s1)
        loss3_flatten = Flatten()(pool5_drop_7x7_s1)
        loss3_classifier = Dense(num_class, name='loss3/classifier', kernel_regularizer=l2(0.0002))(loss3_flatten)
        loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        model = Model(inputs=putin, outputs=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

        # model = Model(inputs=putin, outputs=[loss1_classifier_act])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Model summary")
        print(model.summary())

        if pretrained_weights:
            model.load_weights(pretrained_weights)

        return model


class jbdm_v2():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(base_output)
        base_output = tf.keras.layers.Flatten(name="flatten")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_05():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.Flatten(name="flatten")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_06():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.Flatten(name="flatten")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_1():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.Flatten(name="flatten")(base_output)
        base_output = tf.keras.layers.Dense(2048, activation="relu")(base_output)
        base_output = tf.keras.layers.Dropout(0.5)(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_2():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.Flatten(name="flatten")(base_output)
        base_output = tf.keras.layers.Dropout(0.5)(base_output)
        base_output = tf.keras.layers.Dense(2048, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_25():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        # baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        base_output = tf.keras.layers.Dropout(0.5)(base_output)
        base_output = tf.keras.layers.Dense(256, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_26():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        base_output = tf.keras.layers.Dropout(0.5)(base_output)
        base_output = tf.keras.layers.Dense(256, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_27():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        base_output = tf.keras.layers.Dropout(0.25)(base_output)
        base_output = tf.keras.layers.Dense(256, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_28():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        base_output = tf.keras.layers.Dense(256, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_29():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        base_output = tf.keras.layers.Dense(256, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_3():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2()
        # baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
        #                                                            include_top=False,
        #                                                            input_tensor=Input(shape=input_size)
        #                                                            )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_31():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        base_output = tf.keras.layers.Dense(128, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_MobileNetV2():  # jbdm_v2_32():
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, retrainAll=False):
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
        base_output = tf.keras.layers.Dense(classifierLayer, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class jbdm_v2_33():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None):
        baseModel = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
        base_output = tf.keras.layers.Dense(1024, activation="relu")(base_output)
        new_output = tf.keras.layers.Dense(num_class, activation="softmax")(base_output)
        new_model = tf.keras.models.Model(inputs=baseModel.inputs,
                                          outputs=new_output)
        new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        print("Model summary")
        print(new_model.summary())
        if pretrained_weights:
            new_model.load_weights(pretrained_weights)
        return new_model


class SR_MobileNetV3Small():  # jbdm_v2_5():
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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


class jbdm_v3_1():
    def build(num_class, input_size=(128, 128, 3), pretrained_weights=None, lr=1e-3):
        # baseModel = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet')
        baseModel = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',
                                                                   include_top=False,
                                                                   input_tensor=Input(shape=input_size)
                                                                   )
        print("Base Model summary")
        print(baseModel.summary())
        baseModel.trainable = False
        # base_output = baseModel.layers[-2].output # layer number obtained from model summary above
        base_output = baseModel.output
        base_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
        base_output = tf.keras.layers.Dense(classifierLayer=512, activation="relu")(base_output)
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
    def build(num_class, input_size=(128, 128, 3), classifierLayer=512, pretrained_weights=None, lr=1e-3, retrainAll=False):
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
