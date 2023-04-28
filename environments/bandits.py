import numpy as np


class Bandit:
    def __init__(self, config):
        self.arm_ids = np.arange(len(config['payoffs']))
        self.num_arms = len(config['payoffs'])
        self.payoffs = config['payoffs']
        self.sampling_distribution = config['sampling_distribution']
           #TODO set seed
        if self.sampling_distribution == 'uniform': 
            self.random_sample = np.random.random_sample # Uniform in [0,1)

    def sample(self, arm_id):
        '''
        inputs: arm_id: the arm being chosen
        return: 1 or 0 (1 is payoff percent of time for that arm, 0 otherwise)
        '''
        x = self.random_sample()
        return 1 if x < self.payoffs[arm_id] else 0
