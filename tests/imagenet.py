import tensorflow as tf
import tensorflow_datasets as tfds

imagenet_train = tfds.load("imagenet_resized/64x64",split="train")
