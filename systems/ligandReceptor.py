import numpy as np
import tensorflow as tf


def ensure_traj(X):
    if np.ndim(X) == 2:
        return X
    if np.ndim(X) == 1:
        return np.array([X])
    raise ValueError('Incompatible array with shape: ', np.shape(X))


def distance_matrix_squared(crd1, crd2, dim=2):
    """ Returns the distance matrix or matrices between particles
    Parameters
    ----------
    crd1 : array or matrix
        first coordinate set
    crd2 : array or matrix
        second coordinate set
    dim : int
        dimension of particle system. If d=2, coordinate vectors are [x1, y1, x2, y2, ...]
    """
    crd1 = ensure_traj(crd1)
    crd2 = ensure_traj(crd2)
    n = int(np.shape(crd1)[1]/dim)

    crd1_components = [np.tile(np.expand_dims(crd1[:, i::dim], 2), (1, 1, n)) for i in range(dim)]
    crd2_components = [np.tile(np.expand_dims(crd2[:, i::dim], 2), (1, 1, n)) for i in range(dim)]
    D2_components = [(crd1_components[i] - np.transpose(crd2_components[i], axes=(0, 2, 1)))**2 for i in range(dim)]
    D2 = np.sum(D2_components, axis=0)
    return D2


class LigandReceptor (object):

    params_default = {'nsolvent': 0,
                      'eps': 1.0,  # LJ prefactor
                      'rm': 1.0,  # LJ particle size
                      'k_rep': 20.0,  # dimer force constant
                      'sigma_rep': 1.0,  # dimer force constant
                      'box_halfsize': 3.0,
                      'box_k': 100.0,  # box repulsion force constant
                      'grid_k': 0.0,  # restraint strength to particle grid (to avoid permutation)
                      }
    def __init__(self, params=None):
        # set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables
        self.nparticles = params['nsolvent'] + 4
        self.dim = int(3 * self.nparticles)
        self.grid = self.init_positions(params['dimer_dmid'])

        # create mask matrix to help computing particle interactions
        self.mask_matrix = np.ones((self.nparticles, self.nparticles), dtype=np.float32)
        for i in range(self.nparticles):
            self.mask_matrix[i, i] = 0.0

#    [i,j,k]
#    i - system configuration
#    j - particle number - li
#    k - cartesian coordinate - ligand A(0), ligand B(1), receptor A(2), receptor B(3), receptor C(4)
    def ligandAB_distance(self, x):
        return np.sqrt((x[:, 0, 0] - x[:, 1, 0])**2 + (x[:, 0, 1] - x[:, 1, 1])**2 + (x[:, 0, 2] - x[:, 1, 2])**2)

    def ligandAB_distance_tf(self, x):
        return tf.sqrt((x[:, 0, 0] - x[:, 1, 0])**2 + (x[:, 0, 1] - x[:, 1, 1])**2 + (x[:, 0, 2] - x[:, 1, 2])**2)

    def ligandA_receptorA_distance(self, x):
        return np.sqrt((x[:, 0, 0] - x[:, 2, 0])**2 + (x[:, 0, 1] - x[:, 2, 1])**2 + (x[:, 0, 2] - x[:, 2, 2])**2)

    def ligandA_receptorA_distance_tf(self, x):
        return tf.sqrt((x[:, 0, 0] - x[:, 2, 0])**2 + (x[:, 0, 1] - x[:, 2, 1])**2 + (x[:, 0, 2] - x[:, 2, 2])**2)

    def ligandB_receptorB_distance(self, x):
        return np.sqrt((x[:, 1, 0] - x[:, 3, 0])**2 + (x[:, 1, 1] - x[:, 3, 1])**2 + (x[:, 1, 2] - x[:, 3, 2])**2)

    def ligandB_receptorB_distance_tf(self, x):
        return tf.sqrt((x[:, 1, 0] - x[:, 3, 0])**2 + (x[:, 1, 1] - x[:, 3, 1])**2 + (x[:, 1, 2] - x[:, 3, 2])**2)

    def ligandA_receptorC_distance(self, x):
        return np.sqrt((x[:, 0, 0] - x[:, 4, 0])**2 + (x[:, 0, 1] - x[:, 4, 1])**2 + (x[:, 0, 2] - x[:, 4, 2])**2)

    def ligandA_receptorC_distance_tf(self, x):
        return np.sqrt((x[:, 0, 0] - x[:, 4, 0])**2 + (x[:, 0, 1] - x[:, 4, 1])**2 + (x[:, 0, 2] - x[:, 4, 2])**2)

    def repulsive_energy(self, conf):
        x_comp = conf[:, :, 0]
        y_comp = conf[:, :, 1]
        z_comp = conf[:, :, 2]
        n = np.shape(x_comp)[1]
        x_comp = np.tile(np.expand_dims(x_comp, 2), [1, 1, n])
        y_comp = np.tile(np.expand_dims(y_comp, 2), [1, 1, n])
        z_comp = np.tile(np.expand_dims(z_comp, 2), [1, 1, n])
        d_x = x_comp - np.transpose(x_comp, perm=[0, 2, 1])
        d_y = y_comp - np.transpose(y_comp, perm=[0, 2, 1])
        d_z = z_comp - np.transpose(z_comp, perm=[0, 2, 1])
        d2 = np.sqrt(d_x**2 + d_y**2 + d_z**2)
        d2 = np.where(d2 < self.params['sigma_rep'], d2, self.params['sigma_rep'])
        energy = 0.5 * self.params['k_rep'] * (1-d2/self.params['sigma_rep'])
        return np.sum(energy, axis=(1, 2))

    def repulsive_energy(self, conf):
        x_comp = conf[:, :, 0]
        y_comp = conf[:, :, 1]
        z_comp = conf[:, :, 2]
        n = np.shape(x_comp)[1]
        x_comp = tf.tile(tf.expand_dims(x_comp, 2), [1, 1, n])
        y_comp = tf.tile(tf.expand_dims(y_comp, 2), [1, 1, n])
        z_comp = tf.tile(tf.expand_dims(z_comp, 2), [1, 1, n])
        d_x = x_comp - tf.transpose(x_comp, perm=[0, 2, 1])
        d_y = y_comp - tf.transpose(y_comp, perm=[0, 2, 1])
        d_z = z_comp - tf.transpose(z_comp, perm=[0, 2, 1])
        d2 = tf.sqrt(d_x**2 + d_y**2 + d_z**2)
        d2 = tf.where(d2 < self.params['sigma_rep'], d2, self.params['sigma_rep'])
        energy = 0.5 * self.params['k_rep'] * (1-d2/self.params['sigma_rep'])
        return tf.reduce_sum(energy, axis=(1, 2))

