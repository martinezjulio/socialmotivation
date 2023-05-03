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


class ContextualSocialBandit:
    """
    Contextual bandit with K arms and each arm has a d-dimensional feature vector.
    The arms are either modeled by a prob dist whose parameters changes during learning (last arm),
    or depended on arms selected by other agents.
    * cooking outcome is measured by a prob dist, and competency is just parameter of that dist
    """
    def __init__(self, config) -> None:
        self.rng = np.random.default_rng(config['seed'])
        self.num_arms = config['num_arms']
        self.arm_ids = np.arange(self.num_arms)        
        # values when each social arm is chosen
        self.social_arm_values = config['social_arm_values']        
        self.cooking_threshold = config['cooking_threshold']
        # keeps track of the recipricol social connections
        num_agents = int(self.social_arm_values.shape[0])
        self.total_reciprocal_connections = np.zeros((num_agents, num_agents))

    # share the cooking reward only with 
    def _prepare_cooking_rewards(self, agents):           
        for i, cooking_reward in enumerate(self.step_cooking_status):            
            agent_scores = agents[i].agent_scores
            reward_frac = cooking_reward / np.sum(agent_scores[agent_scores > 0])
            self.step_cooking_rewards += np.where(agent_scores > 0, reward_frac * agent_scores, 0)        

    def _convert_social_arm(self, arm_id, agent):
        arm_choice_arr = np.zeros(self.num_arms-1)
        arm_choice_arr[arm_id] = 1
        social_values = np.multiply(self.social_arm_values[agent.agent_id], arm_choice_arr)
        # because there are 3 values for each social actions (dislike, neutral and like)
        interact_agent_id = np.floor(arm_id/3).astype(int)
        agent.agent_scores[interact_agent_id] += social_values[arm_id]
        self.step_social_connections[agent.agent_id, interact_agent_id] += social_values[arm_id]
    
    def compute_social_reward(self, agent_id):   
        recipricol_connections = self.step_social_connections[agent_id] + self.step_social_connections[:, agent_id]
        # liking yourself does not count as a reward
        recipricol_connections[agent_id] = 0
        return (recipricol_connections).sum()

    def _prepare_reward(self, agent_id, arm_id, agent):
        print(f"Getting reward for agent {agent_id} and arm {arm_id} cooking competence {agent.cooking_competence}")        
        if arm_id == self.arm_ids[-1]:
            # increase competence if cooking arm is chosen
            agent.cooking_competence += agent.cooking_competence_delta
            cooking_outcome = self.rng.normal(agent.cooking_competence, 1)
            if cooking_outcome > self.cooking_threshold:
                print(f"agent {agent_id} cooks successfully")
                self.step_cooking_status[agent_id] = 1
            else:
                print(f"agent {agent_id} cooks unsuccessfully")
                self.step_cooking_status[agent_id] = -1                
        else:
            # choosing social arm
            self._convert_social_arm(arm_id, agent)        

    def step(self, actions, agents):
        self.step_cooking_status = np.zeros(len(agents))
        self.step_cooking_rewards = np.zeros(len(agents))
        self.step_social_rewards = np.zeros(len(agents))        
        self.step_social_connections = np.zeros((len(agents), len(agents)))        
        for agent_id, (action, agent) in enumerate(zip(actions, agents)):            
            self._prepare_reward(agent_id, action, agent)        
        self._prepare_cooking_rewards(agents)

        for agent_id, agent in enumerate(agents):
            # # get rewards
            social_reward = self.compute_social_reward(agent_id)
            self.step_social_rewards[agent_id] = social_reward
            cooking_reward = self.step_cooking_rewards[agent_id]
            self.step_cooking_rewards[agent_id] = cooking_reward 
            reward = social_reward + cooking_reward
            # update total recipricol connections
            recipricol_connections = self.step_social_connections[agent_id] + self.step_social_connections[:,agent_id]
            self.total_reciprocal_connections[agent_id] += recipricol_connections            
            #print(f"social reward {social_reward} cooking reward {cooking_reward} total reward {reward}")
        print(self.total_reciprocal_connections)
        step_total_rewards = self.step_social_rewards + self.step_cooking_rewards        
        return step_total_rewards
        