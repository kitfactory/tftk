import tensorflow as tf
import tftk

class OptimizerBuilder():

    @classmethod
    def get_optimizer(cls,name:str,**kwargs)->tf.keras.optimizers.Optimizer:
        """Optimizerを取得する。
        
        Arguments:
            name {str} --　オプティマイザの名称{sgd/adadelta}

            lr {float}  -- SGDの学習率(デフォルト 0.01) , adadeltaの学習率(デフォルト 1.0)
            sgd_momentum {float} -- SGDのモーメンタム,デフォルト 0.0
            sgd_nestrov {bool} -- SGDのnestrov
            adadelta_rho {float} -- デフォルト 0.95
            adadelta_epsilon {float} -- adadeltaのepsilon デフォルトNone
            adadelta_decay {float} -- adadeltaのdecay デフォルト0.0
        
        Returns:
            tf.keras.optimizers.Optimizer -- [description]
        """
        ret = None
        if name == "sgd":
            learning_rate = kwargs.get("lr", 0.01)
            momentum = kwargs.get("sgd_momentum", 0.0)
            nesterov = kwargs.get("sgd_nestrov" , False)
            print("learning_rate",learning_rate,"momentum",momentum,"nestrov",nesterov)
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
            lr = kwargs.get("lr", 0.001)
            ret = tf.keras.optimizers.RMSprop(lr=lr)
        
        if tftk.IS_MIXED_PRECISION() == True:
            ret = tf.keras.mixed_precision.experimental.LossScaleOptimizer(ret, loss_scale='dynamic')       

        return ret


    