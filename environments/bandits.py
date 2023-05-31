import numpy as np
from typeguard import typechecked

@typechecked
class Bandit():
    def __init__(self, config: dict) -> None:
        self.arm_ids = np.arange(len(config['payoffs']))
        self.num_arms = len(config['payoffs'])
        self.payoffs = config['payoffs']
        self.best_arm_id = np.argmax(self.payoffs).item() # only used for computing regret or for evaluation
        self.sampling_distribution = config['sampling_distribution']
           #TODO set seed
        if self.sampling_distribution == 'uniform': 
            self.random_sample = np.random.random_sample # Uniform in [0,1)
        

    def pull_arm(self, arm_id: int) -> float:
        '''
        inputs: arm_id: the arm being chosen
        return: 1 or 0 (1 is payoff percent of time for that arm, 0 otherwise)
        '''
        #TODO: Fix problem when arm_id=None
        x = self.random_sample()
        return 1. if x < self.payoffs[arm_id] else 0.

