import numpy as np

from agents import banditagents
from environments import bandits


class Experiment():
    def __init__(self, agent_config, bandit_config):
        self.env = bandits.ContextualSocialBandit(bandit_config)
        self.agents = []
        for agent_id in range(agent_config['num_agents']):  
            agent_config['agent_id'] = agent_id
            self.agents.append(agent_config['agent_func'](agent_config))
            print(f"Initalized agent {agent_id} with scores {self.agents[agent_id].agent_scores}")
        self.all_total_rewards = np.zeros((agent_config['num_iterations'], 
                                            agent_config['num_agents']))
        self.all_social_rewards = np.zeros((agent_config['num_iterations'],
                                            agent_config['num_agents']))
        self.all_cooking_rewards = np.zeros((agent_config['num_iterations'],
                                            agent_config['num_agents']))
        self.all_total_reciprocal_connections = []    
        
    def run_exp(self):
        for step in range(self.agents[0].num_iterations):
            actions = []                        
            for agent in self.agents:
                actions.append(agent.act(self.env))
            step_total_rewards = self.env.step(actions, self.agents)
            # record data
            self.all_total_rewards[step] = step_total_rewards
            self.all_social_rewards[step] = self.env.step_social_rewards
            self.all_cooking_rewards[step] = self.env.step_cooking_rewards
            self.all_total_reciprocal_connections.append(self.env.total_reciprocal_connections)
            # update agents
            for agent_id, agent in enumerate(self.agents):
                agent.update(actions[agent_id], step_total_rewards[agent_id])