import tensorflow as tf


class RNVPLoss:
    def __init__(self, system, network):
        self.system = system
        self.nn = network

    def supervised_acceptance_init(self, y_true, y_pred, factor_supervision=100, factor_jacobian=1000):
        x, w, y_det, y, sigma_x, j_x, z, sigma_y, j_y = self.nn.split_output(y_pred)

        log_pprop_xy = -tf.reduce_sum(w ** 2, axis=1) / 2.0 - tf.reduce_sum(tf.log(sigma_x), axis=1)
        w_y = (x - z) / sigma_y
        log_pprop_yx = -tf.reduce_sum(w_y ** 2, axis=1) / 2.0 - tf.reduce_sum(tf.log(sigma_y), axis=1)
        dE = self.system.energy_tf(y) - self.system.energy_tf(x)
        rev = log_pprop_xy - log_pprop_yx
        acc = tf.abs(dE + rev)

        return acc

