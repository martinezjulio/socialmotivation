import numpy as np

class BaseBanditAgent():
    def __init__(self, config):
        self.num_iterations = config['num_iterations']
        self.num_initial_rounds = config['num_initial_rounds']
        self.decay = config['decay']
        self.epsilon = config['epsilon']
        self.optimistic = config['optimistic']
        self.solver = config['solver']

        self.arm_id_history = []
        self.reward_history = []
        self.best_arm_id = None
        self.best_arm_mean_payoff = None
        
        self.social_agent = config['social_agent']
        if self.social_agent:
            self.observe_simultaenously = config['observe_simultaneously'] # True or False (if False choice alternative)
            self.observe_action_only = config['observe_action_only'] # True or False (if False action and reward)
            self.observe_current_iteration = config['observe_current_iteration'] # 'current', 'best'

    def __call__(self):
        raise NotImplementedError

    def get_agent2_chosen_arm_id(self, agent2, iter):
        if self.observe_current_iteration:
            chosen_arm_id = agent2.arm_id_history[iter]
        else:
            raise NotImplementedError
        return chosen_arm_id

    def get_agent2_observed_reward(self, agent2, iter):
        if self.observe_current_iteration:
            observed_reward = agent2.reward_history[iter]
        else:
            #TODO: 
            raise NotImplementedError
        return observed_reward

    def observe_agent2(self, agent2, bandit, iter):
        '''
        return: this agent's chosen arm and observed reward (or simulated reward) at iter 
        '''
        chosen_arm_id = self.get_agent2_chosen_arm_id(agent2, iter)
        if self.observe_action_only:
            observed_reward = bandit.sample(chosen_arm_id)
        else:
            observed_reward = self.get_agent2_observed_reward(agent2, iter) 
        return chosen_arm_id, observed_reward

    @staticmethod
    def combine_rewards(init_rewards, agent2_observed_rewards):
        combined_init_rewards = {}
        for key in set(init_rewards).union(agent2_observed_rewards):
            val1 = []
            if key in init_rewards:
                val1 = init_rewards[key]
            val2 = []
            if key in agent2_observed_rewards:
                val2 = agent2_observed_rewards[key]
            val = val1 + val2
            combined_init_rewards[key] = val
        return combined_init_rewards

    @staticmethod
    def store_initial_rewards(initial_rewards, chosen_arm_id, observed_reward):
        if chosen_arm_id in initial_rewards:
            initial_rewards[chosen_arm_id].append(observed_reward)
        else:
            initial_rewards[chosen_arm_id] = [observed_reward]

class GreedyAgent(BaseBanditAgent):
    """
    Initially explore initial_rounds times and then stick to the best action.
    """
    def __init__(self, config):
        super().__init__(config)
        self.random_agent=config['random_agent']

    def __call__(self, bandit, agent2=None):
        assert((agent2 is not None and self.social_agent) or (agent2 is None and not self.social_agent)), "Agent2 must be a Bandit Agent if a social agent, otherwise should be type None"

        initial_rewards = {} # agent rewards from self exploration
        agent2_observed_rewards = {} # store rewards this agent observed for agent2

        if self.random_agent: 
            # ignore initial rounds when random agent 
            # (i.e. random agents will just choose arms randomly -- not the estiamte of the best arm)
            self.initial_rounds = self.num_iterations

        for iter in range(self.num_iterations):
            # choose a random arm in initial rounds
            # observe a reward of 1 or 0 depending on arm chosen
            if iter < self.num_initial_rounds:
                if self.social_agent:
                    if self.observe_simultaenously:
                        chosen_arm_id = np.random.choice(bandit.arm_ids) 
                        observed_reward = bandit.sample(chosen_arm_id) 
                        agent2_chosen_arm_id, agent2_observed_reward = self.observe_agent2(agent2, bandit, iter)
                        #self.store_initial_rewards(initial_rewards, chosen_arm_id, observed_reward)
                        self.store_initial_rewards(agent2_observed_rewards, agent2_chosen_arm_id, agent2_observed_reward)
                    else: # choose to observe or pull an arm
                        prob_observing = 1/(bandit.num_arm_ids+1)
                        observe_this_iter = np.random.choice(2,p=[1-prob_observing, prob_observing])
                        if observe_this_iter:
                            # agent gets zero immediate reward for observing (essentially pulling the nonexistent nth+1 arm)
                            chosen_arm_id = bandit.num_arms
                            observed_reward = 0
                            # but now observes an arm pull and reward from agent2
                            agent2_chosen_arm_id, agent2_observed_reward = self.observe_agent2(agent2, bandit, iter)
                            self.store_initial_rewards(agent2_observed_rewards, agent2_chosen_arm_id, agent2_observed_reward)
                        else:
                            chosen_arm_id = np.random.choice(bandit.arm_ids) 
                            observed_reward = bandit.sample(chosen_arm_id) 
                            #self.store_initial_rewards(initial_rewards, chosen_arm_id, observed_reward)
                else: # not a social agent i.e. does not observe other agents
                    chosen_arm_id = np.random.choice(bandit.arm_ids) 
                    observed_reward = bandit.sample(chosen_arm_id)    
                self.store_initial_rewards(initial_rewards, chosen_arm_id, observed_reward)        

            # if not in initial rounds
            else:
                if self.best_arm_id is None:
                    combined_initial_rewards = self.combine_rewards(initial_rewards, agent2_observed_rewards)
                    # now determine best arm with all information available
                    init_mean_payoffs = {} # average payoff using initial rewards
                    for key, value in combined_initial_rewards.items():
                        init_mean_payoffs[key] = np.mean(value) 
                    self.best_arm_id = max(init_mean_payoffs, key=init_mean_payoffs.get)
                    self.best_arm_mean_payoff = init_mean_payoffs[self.best_arm_id]
                chosen_arm_id = self.best_arm_id
                observed_reward = bandit.sample(chosen_arm_id)
            self.arm_id_history.append(chosen_arm_id)
            self.reward_history.append(observed_reward)
        return None
