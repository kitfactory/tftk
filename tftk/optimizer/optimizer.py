import tensorflow as tf
import tftk

class OptimizerBuilder():
    """学習で利用するOptimizerを提供する。混合精度の場合も自動的に調整する。

    """

    @classmethod
    def get_optimizer(cls,name:str="sgd",**kwargs)->tf.keras.optimizers.Optimizer:
        """Optimizerを取得する。
        
        Arguments:
            name {str} --　オプティマイザの名称{sgd/adadelta,rmsprop デフォルト sgd}

            lr {float}  -- SGD,Adadelta,RMSpropの学習率(デフォルト 0.05) , adadeltaの場合は学習率(デフォルト 1.0)
            momentum {float} -- SGDのモーメンタム,デフォルト 0.0
            nestrov {bool} -- SGDのnestrov デフォルト 0.9
            rho {float} -- Adadelta rho デフォルト 0.95
            epsilon {float} -- Adadeltaのepsilon デフォルトNone
            decay {float} -- Adadeltaのdecay デフォルト0.0
        
        Returns:
            tf.keras.optimizers.Optimizer -- [description]
        """
        ret = None
        if name == "sgd":
            learning_rate = kwargs.get("lr", 0.04)
            momentum = kwargs.get("sgd_momentum", 0.9)
            nesterov = kwargs.get("sgd_nestrov" , False)
            ret = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum,nesterov=nesterov)
        elif name == "adadelta":
            # lr=1.0, rho=0.95, epsilon=None, decay=0.0
            lr = kwargs.get("lr", 1.0)
            rho = kwargs.get("adadelta_rho", 0.95)
            epsilon = kwargs.get("adadelta_epsilon",None)
            decay = kwargs.get("adadelta_decay",0.0)
            print("lr",lr,"rho", rho , "epsiloon", epsilon, "decay",decay)
            ret = tf.keras.optimizers.Adadelta(lr=lr,rho=rho,epsilon=epsilon,decay=decay)
        elif name=="adam":
            # learning_rate = kwargs.get("lr",0.001)
            # beta_1 = kwargs.get("beta_1",0.9)
            # beta_2 = kwargs.get("beta_2",0.999)
            # epsilon = kwargs.get("epsilon",1e-8)
            # ret = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
            ret = tf.keras.optimizers.Adam()
        elif name =='rmsprop':
            lr = kwargs.get("lr", 0.04)
            ret = tf.keras.optimizers.RMSprop(lr=lr)
        
        if tftk.IS_MIXED_PRECISION() == True:
            ret = tf.keras.mixed_precision.experimental.LossScaleOptimizer(ret, loss_scale='dynamic')       

        return ret


    