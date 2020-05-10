import tensorflow as tf

mixed_precision = False

def USE_MIXED_PRECISION():
    mixed_precision = True
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def IS_MIXED_PRECISION()->bool:
    policy = tf.keras.mixed_precision.experimental.global_policy()
    print(policy)
    return mixed_precision

"""
--- ポリシーの設定
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
---

--optimizerをdynamicでラップ
optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
--
"""
