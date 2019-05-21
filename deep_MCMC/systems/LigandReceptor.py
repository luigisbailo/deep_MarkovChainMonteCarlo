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
                      'eps_aa': 1.0,  # lennard jones epsilon ligand a - receptor a
                      'eps_bb': 1.0,  # lennard jones epsilon ligand b - receptor b
                      'eps_ac': 1.0,  # lennard jones epsilon ligand a - receptor c
                      'rm_aa': 1.0,  # lennard jones rm ligand a - receptor a
                      'rm_bb': 1.0,  # lennard jones rm ligand b - receptor b
                      'rm_ac': 1.0,  # lennard jones rm ligand a - receptor c
                      'k_rep': 20.0,  # harmonic repulsion all particles
                      'k_ligand': 10.0,  # harmonic attraction ligands
                      'k_receptor': 100.0,  # harmonic attraction receptor_site
                      'box_size': 4.0,
                      'box_k': 100.0,  # box repulsion force constant
                      # receptors always bind to the surface z=0
                      'x_a': 1,  # binding site receptor a
                      'y_a': 0,  # binding site receptor a
                      'x_b': 0,  # binding site receptor b
                      'y_b': 1,  # binding site receptor b
                      'x_c': -1,  # binding site receptor c
                      'y_c': 0,  # binding site receptor c
                      }

    def __init__(self, params=None):
        # set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables
        self.nparticles = params['nsolvent'] + 5
        self.dim = int(3 * self.nparticles)

        # create mask matrix to help computing particle interactions
        self.mask_matrix = np.ones((self.nparticles, self.nparticles), dtype=np.float32)
        for i in range(self.nparticles):
            self.mask_matrix[i, i] = 0.0

    #    [i,j,k]
    #    i - system configuration
    #    j - particle number -- ligand A(0), ligand B(1), receptor A(2), receptor B(3), receptor C(4)
    #    k - cartesian coordinate

    def init_positions(self, binding, batch_size=1):
        #  solvent position to be implemented
        conf = np.zeros([5, 3])
        if binding == 'aa':
            conf[0, 0] = self.params['x_a']
            conf[0, 1] = self.params['y_a']
            conf[0, 2] = self.params['rm_aa']
            conf[1, 0] = self.params['x_a']
            conf[1, 1] = self.params['y_a']
            conf[1, 2] = 2*self.params['rm_aa']
        if binding == 'bb':
            conf[0, 0] = self.params['x_b']
            conf[0, 1] = self.params['y_b']
            conf[0, 2] = 2*self.params['rm_bb']
            conf[1, 0] = self.params['x_b']
            conf[1, 1] = self.params['y_b']
            conf[1, 2] = self.params['rm_bb']
        if binding == 'ac':
            conf[0, 0] = self.params['x_c']
            conf[0, 1] = self.params['y_c']
            conf[0, 2] = self.params['rm_ac']
            conf[1, 0] = self.params['x_c']
            conf[1, 1] = self.params['y_c']
            conf[1, 2] = 2*self.params['rm_ac']

        conf[2, 0] = self.params['x_a']
        conf[2, 1] = self.params['y_a']
        conf[2, 2] = 0
        conf[3, 0] = self.params['x_b']
        conf[3, 1] = self.params['y_b']
        conf[3, 2] = 0
        conf[4, 0] = self.params['x_c']
        conf[4, 1] = self.params['y_c']
        conf[4, 2] = 0

        expanded_confs = np.tile(np.expand_dims(conf, axis=2), batch_size)
        expanded_confs = np.swapaxes(np.swapaxes(expanded_confs, 2, 1), 1, 0)

        return expanded_confs

    @staticmethod
    def ligand_ligand_ab_distance(conf):
        return np.sqrt((conf[:, 0, 0] - conf[:, 1, 0])**2 + (conf[:, 0, 1] - conf[:, 1, 1])**2
                       + (conf[:, 0, 2] - conf[:, 1, 2])**2)

    @staticmethod
    def ligand_ligand_ab_distance_tf(conf):
        return tf.sqrt((conf[:, 0, 0] - conf[:, 1, 0])**2 + (conf[:, 0, 1] - conf[:, 1, 1])**2
                       + (conf[:, 0, 2] - conf[:, 1, 2])**2)

    @staticmethod
    def ligand_receptor_aa_distance(conf):
        return np.sqrt((conf[:, 0, 0] - conf[:, 2, 0])**2 + (conf[:, 0, 1] - conf[:, 2, 1])**2
                       + (conf[:, 0, 2] - conf[:, 2, 2])**2)

    @staticmethod
    def ligand_receptor_aa_distance_tf(conf):
        return tf.sqrt((conf[:, 0, 0] - conf[:, 2, 0])**2 + (conf[:, 0, 1] - conf[:, 2, 1])**2
                       + (conf[:, 0, 2] - conf[:, 2, 2])**2)

    @staticmethod
    def ligand_receptor_bb_distance(conf):
        return np.sqrt((conf[:, 1, 0] - conf[:, 3, 0])**2 + (conf[:, 1, 1] - conf[:, 3, 1])**2
                       + (conf[:, 1, 2] - conf[:, 3, 2])**2)

    @staticmethod
    def ligand_receptor_bb_distance_tf(conf):
        return tf.sqrt((conf[:, 1, 0] - conf[:, 3, 0])**2 + (conf[:, 1, 1] - conf[:, 3, 1])**2
                       + (conf[:, 1, 2] - conf[:, 3, 2])**2)

    @staticmethod
    def ligand_receptor_ac_distance(conf):
        return np.sqrt((conf[:, 0, 0] - conf[:, 4, 0])**2 + (conf[:, 0, 1] - conf[:, 4, 1])**2
                       + (conf[:, 0, 2] - conf[:, 4, 2])**2)

    @staticmethod
    def ligand_receptor_ac_distance_tf(conf):
        return tf.sqrt((conf[:, 0, 0] - conf[:, 4, 0])**2 + (conf[:, 0, 1] - conf[:, 4, 1])**2
                       + (conf[:, 0, 2] - conf[:, 4, 2])**2)

    def receptor_asite_distance(self, conf):
        return np.sqrt((conf[:, 3, 0]-self.params['x_a'])**2 + (conf[:, 3, 1]-self.params['y_a'])**2 +
                       (conf[:, 3, 1]) ** 2)

    def receptor_bsite_distance(self, conf):
        return np.sqrt((conf[:, 4, 0]-self.params['x_b'])**2 + (conf[:, 4, 1]-self.params['y_b'])**2 +
                       (conf[:, 4, 1]) ** 2)

    def receptor_csite_distance(self, conf):
        return np.sqrt((conf[:, 5, 0]-self.params['x_c'])**2 + (conf[:, 5, 1]-self.params['y_c'])**2 +
                       (conf[:, 5, 1]) ** 2)

    @staticmethod
    def lennard_jones_energy(dist, eps, rm):
        return 4 * eps * ((rm/dist)**12 - (rm/dist)**6)

    @staticmethod
    def harmonic_energy(dist, k):
        return k * dist**2

    def receptor_site_energy(self, conf):
        dist_asite = self.receptor_asite_distance(conf)
        energy_asite = self.harmonic_energy(dist_asite, self.params['k_receptor'])
        dist_bsite = self.receptor_asite_distance(conf)
        energy_bsite = self.harmonic_energy(dist_bsite, self.params['k_receptor'])
        dist_csite = self.receptor_asite_distance(conf)
        energy_csite = self.harmonic_energy(dist_csite, self.params['k_receptor'])
        return energy_asite + energy_bsite + energy_csite

    def ligand_receptor_energy(self, conf):
        dist_aa = self.ligand_receptor_aa_distance(conf)
        energy_aa = self.lennard_jones_energy(dist_aa, self.params['eps_aa'], self.params['rm_aa'])
        dist_bb = self.ligand_receptor_bb_distance(conf)
        energy_bb = self.lennard_jones_energy(dist_bb, self.params['eps_bb'], self.params['rm_bb'])
        dist_ac = self.ligand_receptor_ac_distance(conf)
        energy_ac = self.lennard_jones_energy(dist_ac, self.params['eps_ac'], self.params['rm_ac'])
        return energy_aa + energy_bb + energy_ac

    def ligand_ligand_energy(self, conf):
        dist = self.ligand_ligand_ab_distance(conf)
        return self.harmonic_energy(dist, self.params['k_ligand'])

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

    def repulsive_energy_tf(self, conf):
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

    def box_energy(self, conf):
        x_comp = conf[:, :, 0]
        y_comp = conf[:, :, 1]
        z_comp = conf[:, :, 2]
        # indicator functions
        E = 0.0
        d = -(x_comp + self.params['box_size']/2)
        E += np.sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = (x_comp - self.params['box_size']/2)
        E += np.sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = -(y_comp + self.params['box_size']/2)
        E += np.sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = (y_comp - self.params['box_size']/2)
        E += np.sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = -(z_comp + self.params['box_size']/2)
        E += np.sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = (z_comp - self.params['box_size']/2)
        E += np.sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        return E

    def box_energy_tf(self, conf):
        x_comp = conf[:, :, 0]
        y_comp = conf[:, :, 1]
        z_comp = conf[:, :, 2]
        # indicator functions
        E = 0.0
        d = -(x_comp + self.params['box_size']/2)
        E += tf.reduce_sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = (x_comp - self.params['box_size']/2)
        E += tf.reduce_sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = -(y_comp + self.params['box_size']/2)
        E += tf.reduce_sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = (y_comp - self.params['box_size']/2)
        E += tf.reduce_sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = -(z_comp + self.params['box_size']/2)
        E += tf.reduce_sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        d = (z_comp - self.params['box_size']/2)
        E += tf.reduce_sum((np.sign(d) + 1) * self.params['box_k'] * d**2, axis=1)
        return E

    def energy(self, conf):
        return self.ligand_receptor_energy(conf) + self.ligand_ligand_energy(conf) + self.receptor_site_energy(conf) \
               + self.box_energy(conf)

    def draw_config (self, conf):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ligand
        ax.scatter(conf[0, 0], conf[0, 1], conf[0, 2], color='orange')
        ax.scatter(conf[1, 0], conf[1, 1], conf[1, 2], color='red')
        # receptor
        ax.scatter(conf[2, 0], conf[2, 1], conf[2, 2], color='royalblue')
        ax.scatter(conf[3, 0], conf[3, 1], conf[3, 2], color='royalblue')
        ax.scatter(conf[4, 0], conf[4, 1], conf[4, 2], color='royalblue')
        ax.set_xlim(-self.params['box_size']/2, self.params['box_size']/2)
        ax.set_ylim(-self.params['box_size']/2, self.params['box_size']/2)
        ax.set_zlim(-self.params['box_size']/2, self.params['box_size']/2)




