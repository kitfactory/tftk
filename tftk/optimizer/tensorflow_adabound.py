

import tensorflow as tf

# from tensorflow.python.eager import context
# from tensorflow.python.framework import ops # tf.
# from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import resource_variable_ops
# from tensorflow.python.ops import state_ops
# from tensorflow.python.ops import variable_scope
# from tensorflow.python.training import optimizer
# from tensorflow.python.ops.clip_ops import clip_by_value

class AdaBoundOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.001, final_lr=0.1, beta1=0.9, beta2=0.999,
                 gamma=1e-3, epsilon=1e-8, amsbound=False,
                 use_locking=False, name="AdaBound"):
        super(AdaBoundOptimizer, self).__init__(name)
        self._set_hyper("learning_rate", learning_rate)
        # self.lr = learning_rate
        self._final_lr = final_lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._gamma = gamma
        self._amsbound = amsbound

        self.lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        graph = None if tf.executing_eagerly() else tf.get_default_graph()
        create_new = self._get_non_slot_variable("beta1_power", graph) is None
        if not create_new and tf.in_graph_mode():
            create_new = (self._get_non_slot_variable("beta1_power", graph).graph is not first_var.graph)

        if create_new:
            self._create_non_slot_variable(initial_value=self._beta1,
                                           name="beta1_power",
                                           colocate_with=first_var)
            self._create_non_slot_variable(initial_value=self._beta2,
                                           name="beta2_power",
                                           colocate_with=first_var)
            self._create_non_slot_variable(initial_value=self._gamma,
                                           name="gamma_multi",
                                           colocate_with=first_var)
        # Create slots for the first and second moments.
        for v in var_list :
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)


    def _prepare(self):
        self.lr_t = tf.convert_to_tensor(self.lr)
        self._base_lr_t = tf.convert_to_tensor(self.lr)
        self._beta1_t = tf.convert_to_tensor(self._beta1)
        self._beta2_t = tf.convert_to_tensor(self._beta2)
        self._epsilon_t = tf.convert_to_tensor(self._epsilon)
        self._gamma_t = tf.convert_to_tensor(self._gamma)

    def _apply_dense(self, grad, var):
        graph = None if tf.executing_eagerly() else tf.get_default_graph()
        beta1_power = tf.cast(self._get_non_slot_variable("beta1_power", graph=graph), var.dtype.base_dtype)
        beta2_power = tf.cast(self._get_non_slot_variable("beta2_power", graph=graph), var.dtype.base_dtype)
        lr_t = tf.cast(self.lr_t, var.dtype.base_dtype)
        base_lr_t = tf.cast(self._base_lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
        gamma_multi = tf.cast(self._get_non_slot_variable("gamma_multi", graph=graph), var.dtype.base_dtype)

        step_size = (lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = self._final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_multi + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma_multi))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = tf.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = tf.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self._amsbound :
            vhat_t = tf.assign(vhat, tf.maximum(v_t, vhat))
            v_sqrt = tf.sqrt(vhat_t)
        else :
            vhat_t = tf.assign(vhat, vhat)
            v_sqrt = tf.sqrt(v_t)


        # Compute the bounds
        step_size_bound = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * tf.clip_by_value(step_size_bound, lower_bound, upper_bound)

        var_update = tf.assign_sub(var, bounded_lr, use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        graph = None if tf.executing_eagerly() else tf.get_default_graph()
        beta1_power = tf.cast(self._get_non_slot_variable("beta1_power", graph=graph), grad.dtype.base_dtype)
        beta2_power = tf.cast(self._get_non_slot_variable("beta2_power", graph=graph), grad.dtype.base_dtype)
        lr_t = tf.cast(self.lr_t, grad.dtype.base_dtype)
        base_lr_t = tf.cast(self._base_lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, grad.dtype.base_dtype)
        gamma_multi = tf.cast(self._get_non_slot_variable("gamma_multi", graph=graph), var.dtype.base_dtype)

        step_size = (lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = self._final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_multi + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma_multi))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = tf.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = tf.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self._amsbound:
            vhat_t = tf.assign(vhat, tf.maximum(v_t, vhat))
            v_sqrt = tf.sqrt(vhat_t)
        else:
            vhat_t = tf.assign(vhat, vhat)
            v_sqrt = tf.sqrt(v_t)

        # Compute the bounds
        step_size_bound = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * tf.clip_by_value(step_size_bound, lower_bound, upper_bound)

        var_update = tf.assign_sub(var, bounded_lr, use_locking=self._use_locking)

        return tf.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        graph = None if tf.executing_eagerly() else tf.get_default_graph()
        beta1_power = tf.cast(self._get_non_slot_variable("beta1_power", graph=graph), var.dtype.base_dtype)
        beta2_power = tf.cast(self._get_non_slot_variable("beta2_power", graph=graph), var.dtype.base_dtype)
        lr_t = tf.cast(self.lr_t, var.dtype.base_dtype)
        base_lr_t = tf.cast(self._base_lr_t, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
        gamma_t = tf.cast(self._gamma_t, var.dtype.base_dtype)

        step_size = (lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))
        final_lr = self._final_lr * lr_t / base_lr_t
        lower_bound = final_lr * (1. - 1. / (gamma_t + 1.))
        upper_bound = final_lr * (1. + 1. / (gamma_t))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = tf.assign(m, m * beta1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = tf.assign(v, v * beta2_t, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self._amsbound:
            vhat_t = tf.assign(vhat, tf.maximum(v_t, vhat))
            v_sqrt = tf.sqrt(vhat_t)
        else:
            vhat_t = tf.assign(vhat, vhat)
            v_sqrt = tf.sqrt(v_t)

        # Compute the bounds
        step_size_bound = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * tf.clip_by_value(step_size_bound, lower_bound, upper_bound)

        var_update = tf.assign_sub(var, bounded_lr, use_locking=self._use_locking)

        return tf.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: tf.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with tf.control_dependencies(
                [tf.scatter_add(x, i, v)]):
                # [resource_variable_ops.resource_scatter_add(x, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with tf.control_dependencies(update_ops):
            graph = None if tf.executing_eagerly() else tf.get_default_graph()
            beta1_power = self._get_non_slot_variable("beta1_power", graph=graph)
            beta2_power = self._get_non_slot_variable("beta2_power", graph=graph)
            gamma_multi = self._get_non_slot_variable("gamma_multi", graph=graph)
            with tf.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
                update_gamma = gamma_multi.assign(
                    gamma_multi + self._gamma_t,
                    use_locking=self._use_locking)
        return tf.group(*update_ops + [update_beta1, update_beta2, update_gamma],
                                      name=name_scope)


    def get_config(self):
        """Returns the config of the optimimizer.
        An optimizer config is a Python dictionary (serializable)
        containing the configuration of an optimizer.
        The same optimizer can be reinstantiated later
        (without any saved state) from this configuration.
        Returns:
            Python dictionary.
        """
        config = {"name": self._name}
        if hasattr(self, "lr"):
            config["learning_rate"] = self._serialize_hyperparameter("learning_rate")
        if hasattr(self, "_final_lr"):
            config["final_lr"] = self._final_lr
        if hasattr(self, "_beta1"):
            config["beta1"] = self._beta1
        if hasattr(self, "_beta2"):
            config["beta2"] = self._beta2
        if hasattr(self, "_gamma"):
            config["gamma"] = self._gamma
        if hasattr(self, "_asmbound"):
            config["asmbound"] = self._amsbound

        return config