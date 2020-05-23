""" 混合精度による計算を有効にする処理

"""

import tensorflow as tf

from . context import Context

mixed_precision = False


def USE_MIXED_PRECISION():
    """ 学習冒頭で呼び出すことで、混合精度を有効にする

    """
    Context.get_instance()[Context.MIXED_PRECISION]=True
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

def IS_MIXED_PRECISION()->bool:
    """ 混合精度を利用した学習を実施しているかを確認する。

    

    """
    policy = tf.keras.mixed_precision.experimental.global_policy()
    context = Context.get_instance()
    return context[Context.MIXED_PRECISION]


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
