import numpy as np
import keras


class TrainNetwork:
    def __init__(self, nn, particle_system, loss, configurations):

        self.nn = nn
        self.particle_system = particle_system
        self.loss = loss
        self.configurations = configurations

    def train_init(self, n_validation_set):

        training_input_sh = np.concatenate([self.configurations[:-n_validation_set],
                                            self.configurations[-n_validation_set:]],
                                            axis=0)
        training_labels_sh = np.concatenate([self.configurations[:-n_validation_set],
                                            self.configurations[-n_validation_set:]],
                                            axis=0)


        cb = callback_check_phase_space_volume(training_input_sh[-n_validation_set:], self.nn, self.particle_system,5)

        self.nn.train_pair(training_input_sh[:-n_validation_set], training_labels_sh[:-n_validation_set],
                           self.loss.supervised_acceptance_init,
                           validation_data=[training_input_sh[-2 * n_validation_set:],
                                                            training_labels_sh[-2 * n_validation_set:]],
                           learning_rate=0.001, nepochs=5, redraws=1, batchsize=4096,
                           clipnorm=0.01, ncopies=5)


class callback_check_phase_space_volume(keras.callbacks.Callback):

    def __init__(self, test_data, nn, system, stride):
        self.nn = nn
        self.test_data = test_data
        self.system = system
        self.stride = stride

    def on_epoch_end(self, epoch, logs={}):
        if not epoch % self.stride:

            test_data_open = self.test_data[np.argwhere(np.abs(self.test_data[:, 0]-self.test_data[:, 2]) > 1.5)]
            test_data_closed = self.test_data[np.argwhere(np.abs(self.test_data[:, 0]-self.test_data[:, 2]) < 1.5)]
            test_data_open = test_data_open.reshape(len(test_data_open), self.system.dim)
            test_data_closed = test_data_closed.reshape(len(test_data_closed), self.system.dim)
            out_open = self.nn.generate_output(test_data_open, split_output=True, return_dict=True)
            out_closed = self.nn.generate_output(test_data_closed, split_output=True, return_dict=True)

            x_open, y_closed, w_open, sigma_x_open, z_open, sigma_y_open = \
                out_open['x'], out_open['y'], out_open['w'], out_open['sigma_x'], out_open['z'], out_open['sigma_y']
            x_closed, y_open, w_closed, sigma_x_closed, z_closed, sigma_y_closed = \
                out_closed['x'], out_closed['y'], out_closed['w'], out_closed['sigma_x'], out_closed['z'], out_closed['sigma_y']

            if len(y_open) > 0:
                E_x_closed = self.system.energy(x_closed)
                E_y_open = self.system.energy(y_open)
                mean_de = np.mean(E_y_open-E_x_closed)
                print('Mean dE: ', mean_de)
                log_pprop_xy = -np.sum(w_closed ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_x_closed), axis=1)
                w_y_closed = (x_closed - z_closed) / sigma_y_closed
                log_pprop_yx = -np.sum(w_y_closed ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_y_closed), axis=1)
                mean_rev = np.mean(log_pprop_xy - log_pprop_yx)
                print('Mean rev: ', mean_rev)
                pacc = self.nn.pacc_mh_np(out_closed)
                mean_pacc = np.mean(pacc)
                print('Mean acceptance probability ', mean_pacc)

            if len(y_closed) > 0:
                n_part = len(y_closed[0]) // 2
                E_x_open = self.system.energy(x_open)
                E_y_closed = self.system.energy(y_closed)
                mean_de = np.mean(E_y_closed-E_x_open)
                print('Mean dE: ', mean_de)
                log_pprop_xy = -np.sum(w_open ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_x_open), axis=1)
                w_y_open = (x_open - z_open) / sigma_y_open
                log_pprop_yx = -np.sum(w_y_open ** 2, axis=1) / 2.0 - np.sum(np.log(sigma_y_open), axis=1)
                mean_rev = np.mean(log_pprop_xy - log_pprop_yx)
                print('Mean rev: ', mean_rev)
                pacc = self.nn.pacc_mh_np(out_open)
                mean_pacc = np.mean(pacc)
                print('Mean acceptance probability ', mean_pacc)