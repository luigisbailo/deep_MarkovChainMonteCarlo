import numpy as np
from tqdm import tqdm, tqdm_notebook


class MarkovChainLocalSteps:
    def __init__(self, system):
        self.system = system

    def run(self, init_conf, n_steps, noise=0.01, stride=1, temperature=1, reporter=None):
        beta = 1/temperature
        conf = init_conf
        energy = self.system.energy(conf)

        markov_chain = []
        acceptance_list = []

        steps = range(n_steps)
        if reporter is not None:
            if reporter == 'notebook':
                steps = tqdm_notebook(steps)
            elif reporter == 'script':
                steps = tqdm(steps)
            else:
                print('Invalid reporter. Using none.')
        for count in steps:
            conf_trial = conf + noise*np.random.rand(*conf.shape)
            energy_trial = self.system.energy(conf_trial)

            acceptance = np.exp(-beta*(energy_trial-energy))
            energy = np.where(acceptance > 1, energy_trial, energy)

            acceptance_exp = np.expand_dims(np.expand_dims(acceptance, 1), 2)
            conf = np.where(acceptance_exp > 1, conf_trial, conf)
            if count % stride == 0:
                markov_chain.append(conf)
                acceptance = np.where(acceptance<1, acceptance, 1)
                acceptance_list.append(acceptance)

        acceptance_list = np.transpose(np.array(acceptance_list), (1, 0))
        markov_chain = np.transpose(np.array(markov_chain), (1, 0, 2, 3))

        return markov_chain, acceptance_list

