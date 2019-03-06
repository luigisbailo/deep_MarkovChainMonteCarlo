import keras
import tensorflow as tf
import numpy as np

Model = keras.models.Model
Add = keras.layers.Add
Multiply = keras.layers.Multiply
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Input = keras.layers.Input
Lambda = keras.layers.Lambda
concatenate = keras.layers.concatenate
adam = keras.optimizers.Adam
max_norm = keras.constraints.max_norm
Subtract = Lambda(lambda inputs: inputs[0] - inputs[1])


class Network:
    def __init__(self, system, n_blocks, s_nodes, t_nodes, nnodes_sigma, scaling_mu=0.001,
                 scaling_sigma=0.005, scaling_s=0.001, dim=76, fixed_sigma=False, dimer_split=1.5,
                 layer_activation='relu', split_cond=None, split_dims_conditions=None):

        self.dim = dim
        self.scaling_mu = scaling_mu
        self.scaling_sigma = scaling_sigma
        self.nblocks = n_blocks
        self.fixed_sigma = fixed_sigma
        self.dimer_split = dimer_split
        self.layer_activation = layer_activation
        self.system = system
        self.model = None
        if split_cond is None:
            self.split_cond = self._split_cond
        else:
            self.split_cond = split_cond
        self.s_nodes = s_nodes
        self.t_nodes = t_nodes
        self.nnodes_sigma = nnodes_sigma
        self.scaling_s = scaling_s
        self.system = system
        self.N = self.dim // 2
        self.RNVP_blocks = []
        self.generate_network()

    def generate_layers(self):
        # define RNVP blocks
        for i in range(self.nblocks):
            self.RNVP_blocks.append(self.make_layers(self.s_nodes, self.t_nodes, name_layers='block_' + str(i),
                                                     activation=self.layer_activation, term_linear=True))

    def generate_network(self):
        input_x = Input(shape=(self.dim,), name='x')
        input_w = Input(shape=(self.dim,), name='w')

        self.generate_layers()
        self.sigma_layers = []
        for i, s_dim in enumerate(self.nnodes_sigma):
            self.sigma_layers.append(Dense(s_dim, activation='softplus', name='sigma_layer_' + str(i)))
        self.sigma_layers.append(Dense(self.dim, activation='softplus', name='sigma_layer_' + str(i + 1)))
        self.sigma_layers.append(Lambda(self.scale, arguments={'factor': self.scaling_sigma}, name='scale_sigma'))

        y_forward, j_y_forward = self.assemble_forward(input_x, self.RNVP_blocks)
        y_backward, j_y_backward = self.assemble_backward(input_x, self.RNVP_blocks)

        x_yf_yb = concatenate([input_x, y_forward, y_backward])
        y = Lambda(self.select_direction, arguments={'size': self.dim, 'dimer_split': self.dimer_split})(x_yf_yb)
        x_jf_jb = concatenate([input_x, j_y_forward, j_y_backward])
        j_x = Lambda(self.select_direction, arguments={'size': self.N, 'dimer_split': self.dimer_split})(x_jf_jb)

        if self.fixed_sigma == 0.:
            sigma_x = self.concat_layers(input_x, self.sigma_layers)
        else:
            sigma_x = Lambda(self._fixed_sigma)(input_x)

        noise = Multiply()([sigma_x, input_w])
        y_noisy = Add()([y, noise])

        if self.fixed_sigma == 0:
            sigma_y = self.concat_layers(y_noisy, self.sigma_layers)
        else:
            sigma_y = Lambda(self._fixed_sigma)(input_x)

        z_forward, j_z_forward = self.assemble_forward(y_noisy, self.RNVP_blocks)
        z_backward, j_z_backward = self.assemble_backward(y_noisy,self.RNVP_blocks)

        z_fb = concatenate([input_x, z_backward, z_forward])
        z = Lambda(self.select_direction, arguments={'size': self.dim, 'dimer_split': self.dimer_split})(z_fb)

        x_jb_jf = concatenate([input_x, j_z_backward, j_z_forward])
        j_y = Lambda(self.select_direction, arguments={'size': self.N, 'dimer_split': self.dimer_split})(x_jb_jf)

        outputs = [input_x, input_w, y, y_noisy, sigma_x, z, sigma_y, j_x, j_y]
        outputs = concatenate(outputs, name='concatenate_outputs')

        self.model = Model(inputs=[input_x, input_w], outputs=outputs)
        print('Model created successfully. Number of parameters:', self.model.count_params())

    @staticmethod
    def scale(x, factor):
        return factor * x

    def _fixed_sigma(self, x):
        return x*0.+self.fixed_sigma

    def select_direction(self, x_f_b, size, dimer_split):

        x = x_f_b[:, :self.dim]
        f = x_f_b[:, self.dim:(self.dim+size)]
        b = x_f_b[:, (self.dim+size):(self.dim+2*size)]
        cond = self.split_cond(x, dimer_split)
        return tf.where(cond, f, b)

    def _split_cond(self, x, dimer_split):
        return tf.abs(x[:, 0] - x[:, 2]) < dimer_split

    def assemble_forward(self, x, nicer_blocks):

        x0 = Lambda(self.split_x0)(x)
        x1 = Lambda(self.split_x1)(x)
        j = []
        j_intermediate = []
        for i, block in enumerate(nicer_blocks):
            if isinstance(block, list):
                x0, x1, _j = self.add_forward(x0, x1, block)
                j.append(_j)
                j_intermediate.append(_j)
        y0y1_forward = concatenate([x0, x1])
        y_forward = Lambda(self.merge_x0x1)(y0y1_forward)
        log_j = Add()(j)
        return y_forward, log_j

    def assemble_backward(self, x, nicer_blocks):

        x0 = Lambda(self.split_x0)(x)
        x1 = Lambda(self.split_x1)(x)
        j = []
        j_intermediate = []
        for i in reversed(range(len(nicer_blocks))):
            block = nicer_blocks[i]
            if isinstance(block, list):
                x0, x1, _j = self.add_backward(x0, x1, block)
                j.append(_j)
                j_intermediate.append(_j)
        y0y1_backward = concatenate([x0, x1])
        y_backward = Lambda(self.merge_x0x1)(y0y1_backward)
        log_j = Add()(j)
        return y_backward, log_j

    def add_forward(self, x0, x1, block):

        f_s_layers, f_t_layers = block[0]
        g_s_layers, g_t_layers = block[1]
        s0 = self.concat_scaling_layers(x0, f_s_layers)
        s0_exp = Lambda(tf.exp)(s0)
        t0 = self.concat_layers(x0, f_t_layers)
        y0 = x0
        y1 = Multiply()([x1, s0_exp])
        y1 = Add()([y1, t0])
        s1 = self.concat_scaling_layers(y1, g_s_layers)
        s1_exp = Lambda(tf.exp)(s1)
        t1 = self.concat_layers(y1, g_t_layers)
        z0 = Multiply()([y0, s1_exp])
        z0 = Add()([z0, t1])
        z1 = y1
        return z0, z1, Add()([s0, s1])

    def add_backward(self, z0, z1, block):

        f_s_layers, f_t_layers = block[0]
        g_s_layers, g_t_layers = block[1]
        t1 = self.concat_layers(z1, g_t_layers)
        s1 = Lambda(tf.negative)(self.concat_scaling_layers(z1, g_s_layers))
        s1_exp = Lambda(tf.exp)(s1)
        y1 = z1
        y0 = Subtract([z0, t1])
        y0 = Multiply()([y0, s1_exp])
        t0 = self.concat_layers(y0, f_t_layers)
        s0 = Lambda(tf.negative)(self.concat_scaling_layers(y0, f_s_layers))
        s0_exp = Lambda(tf.exp)(s0)
        x0 = y0
        x1 = Subtract([y1, t0])
        x1 = Multiply()([x1, s0_exp])
        return x0, x1, Add()([s0, s1])

    def make_layers(self, s_nodes, t_nodes, name_layers, activation='relu',
                    term_linear=True):

        f_s_layers = []
        f_t_layers = []
        g_s_layers = []
        g_t_layers = []
        for i in range(len(s_nodes)):
            f_s_layers.append(Dense(s_nodes[i], activation=activation, name=name_layers + '_f_s_' + str(i)))
            g_s_layers.append(Dense(s_nodes[i], activation=activation, name=name_layers + '_g_s_' + str(i)))

        f_s_layers.append(Dense(int(self.dim / 2), activation='tanh', name=name_layers + '_f_s_tanh'))
        g_s_layers.append(Dense(int(self.dim / 2), activation='tanh', name=name_layers + '_g_s_tanh'))
        f_s_layers.append(Dense(1, activation='linear', name=name_layers + '_f_s_linear'))
        g_s_layers.append(Dense(1, activation='linear', name=name_layers + '_g_s_linear'))
        for i in range(len(t_nodes)):
            f_t_layers.append(Dense(t_nodes[i], activation=activation, name=name_layers + '_f_t_' + str(i)))
            g_t_layers.append(Dense(t_nodes[i], activation=activation, name=name_layers + '_g_t_' + str(i)))
        if term_linear:
            f_t_layers.append(Dense(int(self.dim / 2), activation='linear', name=name_layers + '_f_t_linear_layer'))
            g_t_layers.append(Dense(int(self.dim / 2), activation='linear', name=name_layers + '_g_t_linear_layer'))
        f_t_layers.append(Lambda(self.scale, arguments={'factor': self.scaling_mu}, name=name_layers +
                                                                                               '_f_t_scale_mu'))
        g_t_layers.append(Lambda(self.scale, arguments={'factor': self.scaling_mu}, name=name_layers +
                                                                                               '_g_t_scale_mu'))
        f_layers = [f_s_layers, f_t_layers]
        g_layers = [g_s_layers, g_t_layers]
        return [f_layers, g_layers]

    def concat_scaling_layers(self, inp, layers):
        layer = inp
        for l in layers[:-1]:
            layer = l(layer)
        scaling = layers[-1](inp)
        scaling = Lambda(self.scale, arguments={'factor': self.scaling_s})(scaling)
        return Multiply()([layer, scaling])

    @staticmethod
    def concat_layers(inp, layers):
        layer = inp
        for l in layers:
            layer = l(layer)
        return layer

    @staticmethod
    def split_x0(x):
        return x[:, ::2]

    @staticmethod
    def split_x1(x):
        return x[:, 1::2]

    def merge_x0x1(self, x0x1):
        x0 = x0x1[:, 0:int(self.dim / 2)]
        x1 = x0x1[:, int(self.dim / 2):self.dim]
        x0_exp = tf.expand_dims(x0, 2)
        x1_exp = tf.expand_dims(x1, 2)
        concat_x0x1 = tf.concat([x0_exp, x1_exp], 2)
        return tf.reshape(concat_x0x1, [-1, self.dim])

    def merge_x0x1j(self, x0x1):
        x0 = x0x1[:, 0:int(self.dim / 2)]
        x1 = x0x1[:, int(self.dim / 2):self.dim]
        j = x0x1[:, self.dim:]
        j = tf.reshape(j, [-1, self.N])
        x0_exp = tf.expand_dims(x0, 2)
        x1_exp = tf.expand_dims(x1, 2)
        concat_x0x1 = tf.concat([x0_exp, x1_exp], 2)
        concat_x0x1 = tf.reshape(concat_x0x1, [-1, self.dim])
        return tf.concat([concat_x0x1, j], 1)

    def split_output(self, out, return_dict=False):

        x = out[:, :self.dim]
        w = out[:, self.dim:2 * self.dim]
        y = out[:, 2 * self.dim:3 * self.dim]
        y_noisy = out[:, 3 * self.dim:4 * self.dim]
        sigma_x = out[:, 4 * self.dim:5 * self.dim]
        z = out[:, 5 * self.dim:6 * self.dim]
        sigma_y = out[:, 6 * self.dim:7 * self.dim]
        j_x = out[:, 7 * self.dim:7 * self.dim + self.N]
        j_y = out[:, 7 * self.dim + self.N:8 * self.dim]

        if return_dict:
            dict = {'x': x, 'w': w, 'y': y, 'y_noisy': y_noisy, 'sigma_x': sigma_x, 'z': z, 'sigma_y': sigma_y,
                    'j_x': j_x, 'j_y': j_y}
            return dict
        else:
            return [x, w, y, y_noisy, sigma_x, j_x, z, sigma_y, j_y]

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def train_pair(self, x0, y0, loss, validation_data=None, learning_rate=0.001, batchsize=2000,
                   redraws=10, nepochs=100, return_samples=False, noise_scale=1., verbose=True, callbacks=[],
                   clipnorm=None, ncopies=1, shuffle=True):
        if clipnorm is not None:
            optimizer = adam(lr=learning_rate, clipnorm=clipnorm)
        else:
            optimizer = adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss)
        N_data = len(x0)
        x0 = np.concatenate([x0 for i in range(ncopies)])
        y0 = np.concatenate([y0 for i in range(ncopies)])
        for i in range(redraws):
            print('Redraw ', i, '/', redraws, ':')
            w = noise_scale * np.random.normal(size=(N_data * ncopies, self.dim)).astype(np.float32)
            xTrain = x0
            yTrain = y0
            if not (validation_data is None):
                w_validation = np.random.normal(size=(validation_data[0].shape[0], self.dim))
                validation = [[validation_data[0], w_validation], validation_data[1]]
            else:
                validation = None
            loss = self.model.fit([xTrain, w], yTrain, epochs=nepochs, batch_size=batchsize, validation_split=0.,
                                  shuffle=shuffle, callbacks=callbacks, validation_data=validation)

    def generate_output(self, x, label_assigner=None, split_output=True, return_dict=True):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x_shape = x.shape
        if len(x_shape) == 1 and x_shape[0] == self.dim:
            n_samples = 1
            x = x.reshape(n_samples, self.dim)
        else:
            n_samples = x_shape[0]
        if label_assigner is not None:
            x = label_assigner.assign_labels(x)
        pred = self.model.predict([x, np.random.randn(n_samples, self.dim).astype(np.float32)])
        if split_output:
            return self.split_output(pred, return_dict=return_dict)
        else:
            return pred

    def pacc_mh_np (self, x, beta=1, w=None, y=None, sigma_x=None, z=None, sigma_y=None):
        if isinstance(x, dict):
            pred = x
            x, w, y, sigma_x, z, sigma_y = pred['x'], pred['w'], pred['y_noisy'], pred['sigma_x'], pred['z'], pred['sigma_y']
        log_pprop_xy = -np.sum(w ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_x), axis=1)
        w_y = (x - z) / sigma_y
        log_pprop_yx = -np.sum(w_y ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_y), axis=1)
        dE = self.system.energy(y) - self.system.energy(x) + log_pprop_xy - log_pprop_yx
        pacc = np.minimum(np.exp(-dE*beta), 1.)
        return pacc