from urllib.request import proxy_bypass
import numpy as np
from typeguard import typechecked
from environments import bandits
from typing import List, Tuple, Dict, Optional
from abc import ABCMeta

@typechecked
class BaseBanditAgent(metaclass=ABCMeta):
    """
    A base class used to represent a social or non-social multiarm bandit agent 

    Attributes
    -----------
    payoffs : dict
    mean_payoffs : dict
    arm_id_history : list
    reward__history : list
    best_arm_id : None or int
    best_arm_mean_payoff : float

    Methods
    -------
    get_agent2_chosen_arm_id

    """
    
    def __init__(self, config: dict) -> None:
        self.num_iterations = config['num_iterations']
        self.social_agent = config['social_agent']
        self.payoffs = {} # dictionary: self.payoffs[arm_id] = [r1,r2,...]
        self.mean_payoffs = {} # dictionary self.mean_payoffs[arm_id] ~ mean([r1,r2,...]) if self.social_agent==False
        self.arm_id_history = []
        self.reward_history = []
        self.best_arm_id = None
        self.best_arm_mean_payoff = None
        if self.social_agent:
            self.agent2_payoffs = {}
            self.prob_observe = config['prob_observe']
            self.observe_simultaenously = config['observe_simultaneously'] # True or False (if False choice alternative)
            self.observe_action_only = config['observe_action_only'] # True or False (if False action and reward)
            self.observe_current_iteration = config['observe_current_iteration'] # 'current', 'best'

    def __call__(self):
        """
        Raises
        ------
        NotImplementedError
            If no sound is set for the animal or passed in as a
            parameter.
        """
        raise NotImplementedError

    @typechecked
    def get_agent2_chosen_arm_id(self, agent2, iter: int) -> int:
        """
        Retrieves the best arm id or chosen arm id on iteration iter by agent2

        Parameters
        ----------
        agent2 : BaseBanditAgent
            Another object of base type BaseBanditAgent
        
        iter : int
            the iteration at which this agent is interested in observing

        Return : int
            the agent2 observed arm index
        """
        #assert(isinstance(agent2, BaseBanditAgent)),"agent2 must of be of type BaseBanditAgent, instead got {}".format(type(BaseBanditAgent))
        if self.observe_current_iteration:
            chosen_arm_id = agent2.arm_id_history[iter]
        else:
            chosen_arm_id = agent2.best_arm_id
        return chosen_arm_id

    def get_agent2_observed_reward(self, agent2, iter: int) -> float:
        """
        Description: Returns agent2 observed reward at iteration iter

        Parameters
        ----------
        agent2 : BaseBanditAgent
            Another object of base type BaseBanditAgent
        iter : int
            the iteration at which this agent is interested in observing reward

        Returns
        -------
        float
            the reward this agent observes as a result of agent2 choosing the observed arm id
        """
        #assert(isinstance(agent2, BaseBanditAgent)),"agent2 must of be of type BaseBanditAgent, instead got {}".format(type(BaseBanditAgent))
        if self.observe_current_iteration:
            observed_reward = agent2.reward_history[iter]
        else:
            raise NotImplementedError
        return observed_reward

    def observe_agent2(self, agent2, bandit: bandits.Bandit, iter: int) -> Tuple[int, float]:
        """
        Description: Returns this agent's chosen arm and observed reward (or simulated reward) at iter 

        Parameters
        ----------
        agent2 : BaseBanditAgent
            Another object of base type BaseBanditAgent
        iter : int
            the iteration at which this agent is interested in observing reward

        Returns
        -------
        int
            the arm id it observed agent2 choose
        float
            the reward this agent observes as a result of agent2 choosing the observed arm id
        """
        chosen_arm_id = self.get_agent2_chosen_arm_id(agent2, iter)
        if self.observe_action_only:
            observed_reward = bandit.sample(chosen_arm_id)
        else:
            observed_reward = self.get_agent2_observed_reward(agent2, iter) 
        return (chosen_arm_id, observed_reward)

    def store_payoffs(self, chosen_arm_id: int, observed_reward: float, self_agent: bool = True) -> None:
        """
        Description: 

        Parameters
        ----------

        Returns
        -------

        """
        if self_agent:
            if chosen_arm_id in self.payoffs:
                self.payoffs[chosen_arm_id].append(observed_reward)
            else:
                self.payoffs[chosen_arm_id] = [observed_reward]
        else:
            if chosen_arm_id in self.agent2_payoffs:
                self.agent2_payoffs[chosen_arm_id].append(observed_reward)
            else:
                self.agent2_payoffs[chosen_arm_id] = [observed_reward]
        return None

    def combine_payoffs(self) -> dict:
        """
        Description: 

        Parameters
        ----------

        Returns
        -------

        """
        if self.social_agent:
            combined_payoffs = {}
            for key in set(self.payoffs).union(self.agent2_payoffs):
                val1 = []
                if key in self.payoffs:
                    val1 = self.payoffs[key]
                val2 = []
                if key in self.agent2_payoffs:
                    val2 = self.agent2_payoffs[key]
                val = val1 + val2
                combined_payoffs[key] = val
        else:
            combined_payoffs = self.payoffs.copy()
        return combined_payoffs

    def explore_socially(self, bandit: bandits.Bandit, agent2, iter: int) -> Tuple[int, float, Optional[int], Optional[float]]:
        """
        Description: 

        Parameters
        ----------

        Returns
        -------

        """
        agent2_chosen_arm_id, agent2_observed_reward = None, None # None if doesn't observe agent2
        if self.observe_simultaenously:
            chosen_arm_id = np.random.choice(bandit.arm_ids).item()
            observed_reward = bandit.sample(chosen_arm_id) 
            agent2_chosen_arm_id, agent2_observed_reward = self.observe_agent2(agent2, bandit, iter)
        else: # choose to observe or pull an arm
            if self.prob_observe is None:
                self.prob_observe = 1/(bandit.num_arms+1)
            observe = np.random.choice(a=[False, True],p=[1-self.prob_observe, self.prob_observe])
            if observe:
                # agent gets zero immediate reward for observing (essentially pulling the nonexistent nth+1 arm)
                chosen_arm_id = bandit.num_arms
                observed_reward = 0
                # but now observes an arm pull and reward from agent2
                agent2_chosen_arm_id, agent2_observed_reward = self.observe_agent2(agent2, bandit, iter)
            else: # agent chooses to randomly pull an arm instead of observe
                chosen_arm_id = np.random.choice(bandit.arm_ids).item()
                observed_reward = bandit.sample(chosen_arm_id) 
        return chosen_arm_id, observed_reward, agent2_chosen_arm_id, agent2_observed_reward

@typechecked
class GreedyAgent(BaseBanditAgent):
    """
    A BaseBanditAgent child class to define an agent that initially explores for self.num_initial_rounds times and then greedily just selects the self.best_arm_id found.

    Atributes
    ---------
    random_agent : bool
        If True, this agent is random and will always select a random arm
    num_initial_iterations : int
        The number of initial iterations the agent select arms randomly for (ignored if random_agent==True)
    solver: str
        Name of the solver, in this case 'GreedyAgent'
    """
    def __init__(self, config):
        super().__init__(config)
        self.random_agent=config['random_agent']
        self.num_initial_iterations=config['num_initial_iterations']
        self.solver = 'GreedyAgent'
     

    def __call__(self, bandit: bandits.Bandit, agent2=None) -> None:
        """
        Description: Runs the GreedyAgent algorithm

        Parameters
        ----------
        bandit : environment.bandits.Bandit
            The bandit problem/environment that the agent is trying to solve

        agent2 : agents.BaseBanditAgent
            The other agent that this agent is socializing with

        Returns
        ------
        None
        """
        assert((agent2 is not None and self.social_agent) or (agent2 is None and not self.social_agent)), "Agent2 must be a Bandit Agent if a social agent, otherwise should be type None"

        #initial_rewards = {} # agent rewards from self exploration
        #agent2_observed_rewards = {} # store rewards this agent observed for agent2

        if self.random_agent: 
            # ignore initial interations when random agent 
            # (i.e. random agents will just choose arms randomly -- not the estiamte of the best arm)
            self.initial_iterations = self.num_iterations

        for iter in range(self.num_iterations):
            # choose a random arm in initial iterations
            # observe a reward of 1 or 0 depending on arm chosen
            if iter < self.num_initial_iterations:
                if self.social_agent: # explore socially (i.e. observe other agents)
                    chosen_arm_id, observed_reward, agent2_chosen_arm_id, agent2_observed_reward = self.explore_socially(bandit, agent2, iter)
                    if agent2_chosen_arm_id is not None:
                        self.store_payoffs(agent2_chosen_arm_id, agent2_observed_reward, self_agent=False)
                else: # ignore other agents in exploration (i.e. just pull a random arm)
                    chosen_arm_id = np.random.choice(bandit.arm_ids).item()
                    observed_reward = bandit.sample(chosen_arm_id)    
                self.store_payoffs(chosen_arm_id, observed_reward)        
            else:
                if self.best_arm_id is None:
                    combined_payoffs = self.combine_payoffs()
                    # now determine best arm with all information available
                    #init_mean_payoffs = {} # average payoff using initial rewards
                    for key, value in combined_payoffs.items():
                        self.mean_payoffs[key] = np.mean(value) 
                    self.best_arm_id = max(self.mean_payoffs, key=self.mean_payoffs.get)
                    self.best_arm_mean_payoff = self.mean_payoffs[self.best_arm_id]
                chosen_arm_id = self.best_arm_id # pull over and over
                observed_reward = bandit.sample(chosen_arm_id)
            self.arm_id_history.append(chosen_arm_id)
            self.reward_history.append(observed_reward)
            #assert(self.best_arm_id is not None)," Failed to set best_arm_id"
        return None

@typechecked
class EpsilonGreedyAgent(BaseBanditAgent):
    """
    A BaseBanditAgent child class to define an agent that uses the epsilon-greedy algorithm by performing the action with the best average
    payoff with the probability (1-epsilon), otherwise picks a random action to keep
    exploring.

    Atributes
    ---------
    num_initial_iterations : int
        The number of initial iterations the agent select arms randomly for (ignored if random_agent==True)
    epsilon: float
        the probability of choosing a random action (0<= epsilon <=1), epsilon 0 means no random actions, 1 means always action
    decay: float
        How much we decay epsilon by after each iteration (0< decay <=1), decay=1 means no decay
    payoffs: dict
        a dict of a list for each arm in the bandit containing the observed rewards for pull that arm
    optimistic: bool
        Initializes all bandit arm payoffs to be 1
    solver: str
        Name of the solver, in this case 'GreedyAgent'
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_initial_iterations = config['num_initial_iterations']
        self.epsilon = config['epsilon']
        self.decay = config['decay']
        self.optimistic = config['optimistic']
        self.payoffs = {}
        self.solver = 'EpsilonGreedyAgent'

    def __call__(self, bandit: bandits.Bandit, agent2=None) -> None:      
        """
        Description: Runs the EpsilonGreedyAgent algorithm

        Parameters
        ----------
        bandit : environment.bandits.Bandit
            The bandit problem/environment that the agent is trying to solve

        agent2 : agents.BaseBanditAgent
            The other agent that this agent is socializing with

        Returns
        ------
        None
        """  
        #self.mean_payoffs = {arm_id:None for arm_id in bandit.arm_ids}
        if self.optimistic:
            self.payoffs = {arm_id:[1] for arm_id in bandit.arm_ids}

        for iter in range(self.num_iterations):
            # with prob epsilon do random action .... OR ..... if iter is within initial round
            if (np.random.random_sample() < self.epsilon) or (iter < self.num_initial_iterations):
                if self.social_agent:
                    chosen_arm_id, observed_reward, agent2_chosen_arm_id, agent2_observed_reward = self.explore_socially(bandit, agent2, iter)
                    if agent2_chosen_arm_id is not None:
                        self.store_payoffs(agent2_chosen_arm_id, agent2_observed_reward, self_agent=False)
                else: # not a social agent
                    chosen_arm_id = np.random.choice(bandit.arm_ids).item()
                    observed_reward = bandit.sample(chosen_arm_id)
            else: # choose the best arm so far
                combined_payoffs = self.combine_payoffs()
                for key,val in combined_payoffs.items():
                    self.mean_payoffs[key] = np.mean(val) 
                self.best_arm_id = max(self.mean_payoffs, key=self.mean_payoffs.get)
                self.best_arm_mean_payoff = self.mean_payoffs[self.best_arm_id]
                chosen_arm_id = self.best_arm_id
                observed_reward = bandit.sample(chosen_arm_id)
            self.store_payoffs(chosen_arm_id, observed_reward) 

            self.epsilon *= self.decay
            self.arm_id_history.append(chosen_arm_id)
            self.reward_history.append(observed_reward)
        return None

