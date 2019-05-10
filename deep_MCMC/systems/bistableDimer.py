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


class bistableDimer (object):

    params_default = {'nsolvent': 36,
                      'eps': 1.0,  # LJ prefactor
                      'rm': 1.0,  # LJ particle size
                      'dimer_slope': -1,  # dimer slope parameter
                      'dimer_a': 25.0,  # dimer x2 parameter
                      'dimer_b': 10.0,  # dimer x4 parameter
                      'dimer_dmid': 1.5,  # dimer transition state distance
                      'dimer_k': 20.0,  # dimer force constant
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
        self.nparticles = params['nsolvent'] + 2
        self.dim = int(2 * self.nparticles)

        # create mask matrix to help computing particle interactions
        self.mask_matrix = np.ones((self.nparticles, self.nparticles), dtype=np.float32)
        self.mask_matrix[0, 1] = 0.0
        self.mask_matrix[1, 0] = 0.0
        for i in range(self.nparticles):
            self.mask_matrix[i, i] = 0.0

        self.grid = self.init_positions(params['dimer_dmid'])

    def init_positions(self, dimer_distance, scaling_factor=1.05):
        pos = []
        pos.append(np.array([-0.5*dimer_distance, 0]))
        pos.append(np.array([0.5*dimer_distance, 0]))
        sqrtn = int(np.sqrt(self.params['nsolvent']))
        locs = np.linspace(-self.params['box_halfsize']-1, self.params['box_halfsize']+1, sqrtn+2)[1:-1]
        for i in range(0, sqrtn):
            for j in range(0, sqrtn):
                pos.append(np.array([locs[i], locs[j]]))
        pos = np.array(pos).reshape((1, 2*(self.params['nsolvent']+2)))
        return scaling_factor * pos

    def dimer_distance(self, x):
        return np.sqrt((x[:, 2] - x[:, 0])**2 + (x[:, 3] - x[:, 1])**2)

    def dimer_distance_tf(self, x):
        return tf.sqrt((x[:, 2] - x[:, 0])**2 + (x[:, 3] - x[:, 1])**2)

    def _distance_squared_matrix(self, crd1, crd2):
        return distance_matrix_squared(crd1, crd2, dim=2)

    def LJ_energy(self, x):
        batchsize = np.shape(x)[0]
        D2 = self._distance_squared_matrix(x, x)
        mmatrix = np.tile(np.expand_dims(self.mask_matrix, 0), (batchsize, 1, 1))
        D2 = D2 + (1.0 - mmatrix)
        D2rel = (self.params['rm']**2) / D2
        D2rel = D2rel * mmatrix
        E = 0.5*self.params['eps']*np.sum(D2rel**6, axis=(1, 2))
        return E

    def LJ_energy_tf(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        batchsize = tf.shape(x)[0]
        n = tf.shape(xcomp)[1]
        Xcomp = tf.tile(tf.expand_dims(xcomp, 2), [1, 1, n])
        Ycomp = tf.tile(tf.expand_dims(ycomp, 2), [1, 1, n])
        Dx = Xcomp - tf.transpose(Xcomp, perm=[0, 2, 1])
        Dy = Ycomp - tf.transpose(Ycomp, perm=[0, 2, 1])
        D2 = Dx**2 + Dy**2
        mmatrix = tf.tile(tf.expand_dims(self.mask_matrix, 0), [batchsize, 1, 1])
        D2 = D2 + (1.0 - mmatrix)
        D2rel = (self.params['rm']**2) / D2
        D2rel = D2rel * mmatrix
        E = 0.5*self.params['eps']*tf.reduce_sum(D2rel**6, axis=(1, 2))
        return E

    def dimer_energy(self, x):
        energy_dx = self.params['dimer_k']*(x[:, 0] + x[:, 2])**2
        energy_dy = self.params['dimer_k']*(x[:, 1])**2 + self.params['dimer_k']*(x[:, 3])**2
        d = np.sqrt((x[:, 0]-x[:, 2])**2 + (x[:, 1]-x[:, 3])**2)
        d0 = 2 * (d - self.params['dimer_dmid'])
        d2 = d0*d0
        d4 = d2*d2
        energy_interaction = self.params['dimer_slope']*d0 - self.params['dimer_a']*d2 + self.params['dimer_b']*d4

        return energy_dx + energy_dy + energy_interaction

    def dimer_energy_tf(self, x):
        energy_dx = self.params['dimer_k']*(x[:, 0] + x[:, 2])**2
        energy_dy = self.params['dimer_k']*(x[:, 1])**2 + self.params['dimer_k']*(x[:, 3])**2
        d = tf.sqrt((x[:, 0]-x[:, 2])**2 + (x[:, 1]-x[:, 3])**2)
        d0 = 2 * (d - self.params['dimer_dmid'])
        d2 = d0*d0
        d4 = d2*d2
        energy_interaction = self.params['dimer_slope']*d0 - self.params['dimer_a']*d2 + self.params['dimer_b']*d4

        return energy_dx + energy_dy + energy_interaction

    def box_energy(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        # indicator functions
        E = 0.0
        d_left = -(xcomp + self.params['box_halfsize'])
        E += np.sum((np.sign(d_left) + 1) * self.params['box_k'] * d_left**2, axis=1)
        d_right = (xcomp - self.params['box_halfsize'])
        E += np.sum((np.sign(d_right) + 1) * self.params['box_k'] * d_right**2, axis=1)
        d_down = -(ycomp + self.params['box_halfsize'])
        E += np.sum((np.sign(d_down) + 1) * self.params['box_k'] * d_down**2, axis=1)
        d_up = (ycomp - self.params['box_halfsize'])
        E += np.sum((np.sign(d_up) + 1) * self.params['box_k'] * d_up**2, axis=1)
        return E

    def box_energy_tf(self, x):
        xcomp = x[:, 0::2]
        ycomp = x[:, 1::2]
        # indicator functions
        E = 0.0
        d_left = -(xcomp + self.params['box_halfsize'])
        E += tf.reduce_sum((tf.sign(d_left) + 1) * self.params['box_k'] * d_left**2, axis=1)
        d_right = (xcomp - self.params['box_halfsize'])
        E += tf.reduce_sum((tf.sign(d_right) + 1) * self.params['box_k'] * d_right**2, axis=1)
        d_down = -(ycomp + self.params['box_halfsize'])
        E += tf.reduce_sum((tf.sign(d_down) + 1) * self.params['box_k'] * d_down**2, axis=1)
        d_up = (ycomp - self.params['box_halfsize'])
        E += tf.reduce_sum((tf.sign(d_up) + 1) * self.params['box_k'] * d_up**2, axis=1)
        return E

    def _energy(self, x):
        return self.LJ_energy(x) + self.dimer_energy(x) + self.box_energy(x)

    def energy(self, x):
        if x.shape[0] < 10000:
            return self._energy(x)
        else:
            energy_x = np.zeros(x.shape[0])
            for i in range(0, len(energy_x), 10000):
                i_from = i
                i_to = min(i_from + 10000, len(energy_x))
                energy_x[i_from:i_to] = self._energy(x[i_from:i_to])
            return energy_x

    def energy_tf(self, x):
        return self.LJ_energy_tf(x) + self.dimer_energy_tf(x) + self.box_energy_tf(x)

    def plot_dimer_energy(self, axis=None):
        """ Plots the dimer energy to the standard figure """
        x_scan = np.linspace(0.5, 2.5, 100)
        E_scan = self.dimer_energy(np.array([-0.5*x_scan, np.zeros(100), 0.5*x_scan, np.zeros(100)]).T)
        E_scan -= E_scan.min()

        import matplotlib.pyplot as plt
        if axis is None:
            axis = plt.gca()
        axis.plot(x_scan, E_scan, linewidth=2)
        axis.set_xlabel('x / a.u.')
        axis.set_ylabel('Energy / kT')
        axis.set_ylim(E_scan.min() - 2.0, E_scan[int(E_scan.size / 2)] + 2.0)

        return x_scan, E_scan
