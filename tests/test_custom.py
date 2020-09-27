import tensorflow as tf

# data = [[(1.0,1.0),(2.0,2.0)],[(3.0,3.0),(4.0,4.0)],[(5.0,5.0),(6.0,6.0)]]
# dataset = tf.data.Dataset.from_tensor_slices(data).batch(1)
# print(dataset)

# class MyAxBLayer(tf.keras.layers.Layer):

#     def __init__(self):
#         super(MyAxBLayer,self).__init__()

#     def build(self, input_shape):
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=[1]),
#             dtype='float32',
#             trainable=True, name='w')
#         tf.print(self.w)

#     def call(self, inputs):
#         tf.print("call x ", inputs )
#         return tf.add(inputs, self.w)

# input = tf.keras.layers.Input(shape=[1,])
# y = MyAxBLayer()(input)

# class Lin(tf.keras.layers.Layer):
#     def __init__(self, output_dim=32):
#         super(Lin, self).__init__()
#         self.output_dim = output_dim
#         self.batch_size = None

#     def build(self, input_shape):
#         print('build')
#         print(input_shape)
#         self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]),self.output_dim],trainable=True)
#         print('kernel')
#         print(self.kernel)

#     def call(self, x):
#         print('call')
#         print(x)
#         return x

# # outpus = Lin(1)(inputs)


# # model = tf.keras.Model(inputs=inputs,outputs=outpus)
# # model.compile(optimizer='sgd',loss='mse')
# # model.fit(dataset,epochs=1)

# # layer = Lin(11)
# # zeros = tf.zeros([10,5])
# # print(zeros)
# # print(layer(zeros))
# # print(layer.trainable_weights)


# inputs = tf.keras.layers.Input(shape=(2,2))
# outpus = Lin(1)(inputs)
# model = tf.keras.Model(inputs=inputs,outputs=outpus)
# model.compile(optimizer='sgd',loss='mse')
# model.fit(dataset,epochs=1)



class Antirectifier(tf.keras.layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super(Antirectifier, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer=self.initializer,
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
        pos = tf.nn.relu(inputs)
        neg = tf.nn.relu(-inputs)
        concatenated = tf.concat([pos, neg], axis=-1)
        mixed = tf.matmul(concatenated, self.kernel)
        return mixed

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Antirectifier, self).get_config()
        config = {"initializer": tf.keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))



batch_size = 100
num_classes = 10
epochs = 20

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import tensorflow_datasets as tfds

def map_fn(data):
    image = tf.reshape(data['image'], shape=[784])
    image = tf.cast(image, tf.float32)
    image = image /255.0
    y = data['label']
    return image, y

train = tfds.load('mnist',split='train').map(map_fn)
test = tfds.load('mnist',split='test').map(map_fn)

train_actual = train.take(50000).skip(10000).batch(batch_size).repeat()
validation_actual = train.skip(50000).take(10000).batch(batch_size).repeat()

print(train)

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Build the model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(256),
        Antirectifier(),
        tf.keras.layers.Dense(256),
        Antirectifier(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10),
    ]
)

# Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(train_actual, steps_per_epoch=500, validation_data=validation_actual, validation_steps=100, epochs=epochs)

# Test the model
# model.evaluate(x_test, y_test)
