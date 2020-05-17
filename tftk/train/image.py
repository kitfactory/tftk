from typing import List
import tensorflow as tf

from tftk.image.dataset import ImageDatasetUtil
from tftk import IS_SUSPEND_RESUME_TRAIN, ResumeExecutor

class ImageTrain():

    def __init__(self):
        pass
    
    @classmethod
    def train_image_classification(
        cls,
        train_data:tf.data.Dataset, train_size:int, batch_size:int,
        validation_data:tf.data.Dataset, validation_size:int,
        shuffle_size:int,
        model:tf.keras.Model,
        callbacks:List[tf.keras.callbacks.Callback],
        optimizer:tf.keras.optimizers.Optimizer,
        loss:tf.keras.losses.Loss,
        max_epoch:int = 5, resume:bool = True):

        # dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

        train_data = train_data.map(ImageDatasetUtil.dict_to_classification_tuple(),num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
        if shuffle_size != 0:
            train_data = train_data.shuffle(shuffle_size)
        train_data = train_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        validation_data = validation_data.map(ImageDatasetUtil.dict_to_classification_tuple(),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_data = validation_data.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # model.compile(optimizer=optimizer,  loss=loss,  metrics="val_loss")

        model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

        model.summary()

        initial_epoch = 0
        exe = ResumeExecutor.get_instance()

        if exe.is_resumable_training()==True:
            print("This is resume training!!")
            exe.resume_model(model)
            resume_val = exe.resume_values()
            initial_epoch, _, _  = resume_val
            initial_epoch = initial_epoch + 1
            print("resume epoch", initial_epoch, "max_epoch", max_epoch)
        else:
            print("Not resume training")

        """
        optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
        sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
        distribute=None, **kwargs
        """
        steps_per_epoch = train_size//batch_size
        validation_steps = validation_size//batch_size
        model.fit(
            train_data,
            callbacks=callbacks,
            validation_data=validation_data,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=max_epoch, initial_epoch=initial_epoch)

"""
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
    use_multiprocessing=False, **kwargs
)
"""