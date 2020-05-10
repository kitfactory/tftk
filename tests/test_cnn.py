import tensorflow as tf

x = [
    [[1,1,1],
    [1,1,1],
    [1,1,1]],
    [[2,2,2],
    [2,2,2],
    [2,2,2]], 
    [[3,3,3],
    [3,3,3],
    [3,3,3]]
]

tx = tf.Variable(x)
tx = tf.reshape(tx, shape=(-1,3,3,3))
print(tx)

y = [2,3,4]
ty = tf.Variable(y)
ty = tf.reshape(ty, shape=(-1,1,1,3))
print(ty)

def multiply(inputs):
    ix = inputs[0]
    iy = inputs[1]
    iz = inputs[2]

    ret = []
    for i in range(iz):
        ret.append(ix[i] * iy[i])

    return tf.keras.backend.stack(ret)

layer = tf.keras.layers.Lambda(multiply,output_shape=(3,3,3))

tz = layer([tx,ty,3],training=True)

print(tz)




