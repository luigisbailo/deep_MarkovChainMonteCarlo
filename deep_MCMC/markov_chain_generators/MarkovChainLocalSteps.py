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
            conf_trial = conf + noise*np.random.normal(size=[*conf.shape])

            energy = self.system.energy(conf)
            energy_trial = self.system.energy(conf_trial)
            acceptance = np.exp(-beta*(energy_trial-energy))
            acceptance_exp = np.expand_dims(np.expand_dims(acceptance, 1), 2)
            rand_extractions = np.random.rand(*acceptance_exp.shape)
            conf = np.where(rand_extractions < acceptance_exp, conf_trial, conf)
            acceptance = np.where(acceptance < 1, acceptance, 1)
            acceptance_list.append(acceptance)

            if count % stride == 0:
                markov_chain.append(conf)

        acceptance_list = np.transpose(np.array(acceptance_list), (1, 0))
        markov_chain = np.transpose(np.array(markov_chain), (1, 0, 2, 3))

        return markov_chain, acceptance_list

