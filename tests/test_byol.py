from tftk.image.dataset import Mnist
from tftk.image.dataset import Food101

from tftk.image.dataset import ImageDatasetUtil
from tftk.image.model.classification import SimpleClassificationModel
from tftk.callback import CallbackBuilder
from tftk.optimizer import OptimizerBuilder
from tftk import Context

from tftk.image.model.representation import SimpleRepresentationModel, add_projection_layers

from tftk.train.image import ImageTrain

from tftk import ENABLE_SUSPEND_RESUME_TRAINING, ResumeExecutor

import tensorflow as tf









class MovingAverageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model
    
    def on_train_begin(self, logs=None):
        print("Starting training")

    def on_train_end(self, logs=None):
        print("Stop training")

    def on_epoch_begin(self, epoch, logs=None):
        print("\nStart epoch")

    def on_epoch_end(self, epoch, logs=None):
        print("\nOn epoch end, updating moving average")
        w1 = self.model.get_weights()
        w2 = []
        for a in w1:
            print(type(a))
            w2.append( a*0.8 )
        self.model.set_weights(w2)

def get_moving_average_callback(model):
    m = model
    def moving_average(loss, acc):
        print("on epoch end")
        w1 = m.get_weights()
        w2 = []
        for a in w1:
            print(type(a))
            w2.append( a*0.8 )
        m.set_weights(w2)
    return moving_average

def custom_loss(y_pred, y_true):
    y_1, y_2 = y_pred
    diff = y_1 - y_2
    loss = tf.keras.backend.abs(diff)
    return loss

def reinforcement(data):
    img = data["image"]
    label = data["label"]
    return ([img,img],[img,img])

    

# supervised

def supervised_dataset(dataset:tf.data.Dataset, max_label:int)->tf.data.Dataset:
    filtered = dataset.filter(lambda data:data['label'] < max_label)

    def supervised_transform(data):
        image = data['image']
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        label = data['label']
        label = tf.one_hot(label, max_label)
        return image, label

    return filtered.map(supervised_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def pretext_dataset(dataset:tf.data.Dataset, start_label:int)->tf.data.Dataset:
    filtered = dataset.filter(lambda data:data['label'] >= start_label)

    def supervised_transform(data):
        image = data['image']
        image = tf.cast(image, tf.float32)
        image = image / 255.0


    def random_transform(image):
        pass


if __name__ == '__main__':

    context = Context.init_context(TRAINING_NAME='')
    # ENABLE_SUSPEND_RESUME_TRAINING()

    BATCH_SIZE = 500
    CLASS_NUM = 10
    IMAGE_SIZE = 28
    EPOCHS = 2
    SHUFFLE_SIZE = 1000






    # if IS_SUSPEND_RESUME_TRAIN() == True and IS_ON_COLABOLATORY_WITH_GOOGLE_DRIVE()== True:
    

    # train, train_len = Mnist.get_train_dataset()
    # validation, validation_len = Mnist.get_test_dataset()

    # train = train.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))
    # validation = validation.map(ImageDatasetUtil.image_reguralization()).map(ImageDatasetUtil.one_hot(CLASS_NUM))

    # train = train.map(reinforcement)

    # online_model = SimpleRepresentationModel.get_representation_model(input_shape=(28,28,1))
    # target_model = SimpleRepresentationModel.get_representation_model(input_shape=(28,28,1))

    # print(online_model.layers)
    # online_projection_model = add_projection_layers(online_model)
    # target_projection_model = add_projection_layers(target_model)

    # input_online = online_model.layers[0].input
    # input_target = target_model.layers[0].input

    # output_online = online_model.layers[-1].output
    # output_target = target_model.layers[-1].output

    # mearged_model = tf.keras.Model(inputs=[input_online,input_target], outputs=[output_online,output_target])
    # mearged_model.summary()

    # optimizer = OptimizerBuilder.get_optimizer(name="rmsprop")
    # callbacks = CallbackBuilder.get_callbacks(tensorboard=False, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.25,early_stopping_patience=16)

    # mearged_model.compile(optimizer=optimizer, loss=custom_loss)

    # train = train.take(10)
    # y = mearged_model.predict(train)
    # print(y)
    # optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs)



    # online_projection = add_projection_layers(online_model)
    # target_projection = add_projection_layers(target_model)

    # inputs = [online_projection.input, target_projection.input]
    # outputs = [online_projection.output, target_projection.output]

    # total_model = tf.keras.Model(inputs=inputs, outputs=outputs)


    # optimizer = OptimizerBuilder.get_optimizer(name="rmsprop")
    # model = SimpleClassificationModel.get_model(input_shape=(IMAGE_SIZE,IMAGE_SIZE,1),classes=CLASS_NUM)

    # callbacks = CallbackBuilder.get_callbacks(tensorboard=False, reduce_lr_on_plateau=True,reduce_patience=5,reduce_factor=0.25,early_stopping_patience=16)
    # callbacks.append(MovingAverageCallback(model))

    # ImageTrain.train_image_classification(train_data=train,train_size=train_len,batch_size=BATCH_SIZE,validation_data=validation,validation_size=validation_len,shuffle_size=SHUFFLE_SIZE,model=model,callbacks=callbacks,optimizer=optimizer,loss="categorical_crossentropy",max_epoch=EPOCHS)

    # w1 = model.get_weights()
    # # print(type(w1))

    # w2 = []
    # for a in w1:
    #     print(type(a))
    #     w2.append( a*0.8 )

    # model.set_weights(w2)
