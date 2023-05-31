# %%
from agents import banditagents
from environments import bandits
import utils

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
sns.set_theme(style="whitegrid", palette="pastel")
figures_dir = os.path.join(os.getcwd(), 'figures')

np.random.seed(seed=0)
alpha = 0.6
num_episodes = 40 #40
num_iterations = 500
observe_best_arm = True
save_all_iterations = True
configs_dir = '/Users/juliomartinez/Documents/PhD/socialmotivation/configs'
config_filename = os.path.join(configs_dir, 'VaryingTeachers.yaml')
config = utils.get_config(config_filename)

# %% 
def run_episode(demonstrator, learner, env, episode, group, 
                episodes_learner_arm_id_history, episodes_demonstrator_arm_id_history):
    # run demonstrator
    demonstrator(env)
    # run learner
    learner(env, demonstrator)

    # compute stats
    demonstrator_cumsum = np.cumsum(demonstrator.reward_history)
    learner_cumsum = np.cumsum(learner.reward_history)
    learner_delta_cumsum = learner_cumsum - demonstrator_cumsum
    demonstrator_delta_cumsum =  -learner_delta_cumsum 

    if episode == 0:
        episodes_learner_arm_id_history = np.array(learner.reward_history).reshape(-1,1)
        episodes_demonstrator_arm_id_history = np.array(demonstrator.reward_history).reshape(-1,1)
    else:

        #print('episodes_learner_arm_id_history.shape: {}'.format(episodes_learner_arm_id_history.shape))
        #print('learner.reward_history.shape: {}'.format(np.array(learner.reward_history).reshape(-1,1).shape))

        episodes_learner_arm_id_history = np.hstack((episodes_learner_arm_id_history, np.array(learner.reward_history).reshape(-1,1)))
        episodes_demonstrator_arm_id_history = np.hstack((episodes_demonstrator_arm_id_history, np.array(demonstrator.reward_history).reshape(-1,1)))

    # proportion of time best arm is chosen (given per iteration but across episodes)
    learner_prop_best_arm = np.mean(episodes_learner_arm_id_history == 0,axis=1)
    demonstrator_prop_best_arm = np.mean(episodes_demonstrator_arm_id_history == 0,axis=1)
    # proportion of time observe is chosen (given per iteration but across episodes)
    learner_prop_observe = np.mean(episodes_learner_arm_id_history == 2,axis=1)
    demonstrator_prop_observe = np.mean(episodes_demonstrator_arm_id_history == 2,axis=1)
    # proportion of time alternative arm is chosen (given per iteration but across episodes)
    learner_prop_other = 1 - learner_prop_best_arm - learner_prop_observe
    demonstrator_prop_other = 1 - demonstrator_prop_best_arm - demonstrator_prop_observe

    data = {
        'group': np.repeat(group,2*num_iterations),
        'solver': np.concatenate((np.repeat(learner.solver,num_iterations), np.repeat(demonstrator.solver,num_iterations))),
        'episode': np.repeat(episode,2*num_iterations),
        'iteration': np.tile(np.arange(num_iterations),2),
        'agent': np.concatenate((['learner']*num_iterations,['demonstrator']*num_iterations)),
        'reward':  np.concatenate((learner.reward_history,demonstrator.reward_history)),
        'cumulative_reward': np.concatenate((learner_cumsum, demonstrator_cumsum)),
        'delta_cumulative_reward': np.concatenate((learner_delta_cumsum,demonstrator_delta_cumsum)),
        'chosen_arm_id': np.concatenate((learner.arm_id_history,demonstrator.arm_id_history)),
        'alpha': np.repeat(alpha,2*num_iterations),
        'num_arms': np.repeat(env.num_arms,2*num_iterations),
        'trust_distr': np.repeat('p_trust: {}, p_obs: {}'.format(learner.p_greedytrust, learner.p_observe),2*num_iterations),
        'proportion_best_arm': np.concatenate((learner_prop_best_arm,demonstrator_prop_best_arm)),
        'proportion_observe': np.concatenate((learner_prop_observe,demonstrator_prop_observe)),
        'proportion_other': np.concatenate((learner_prop_other,demonstrator_prop_other)),
        }
    episode_df = pd.DataFrame.from_dict(data)
    return episode_df, episodes_learner_arm_id_history, episodes_demonstrator_arm_id_history


# %%
# for alpha iteration
# 
for i, alpha in enumerate(config['bandit']['alphas']):
    print(f'alpha: {alpha}')

    # setup environment
    bandit_config = {}
    bandit_config['payoffs'] = np.sort(np.array([alpha, 1.0-alpha]))[::-1] # largest payoff arm_id=0
    bandit_config['sampling_distribution'] = config['bandit']['sampling_distribution']
    env = bandits.Bandit(bandit_config)

    # solver iteration
    for j, (demonstratorSolver, learnerSolver) in enumerate(zip(config['demonstrators']['solvers'], config['learners']['solvers'])):
        print(f'demonstratorSolver: {demonstratorSolver}, ', f'learnerSolver: {learnerSolver}')

        # setup agent configs
        demonstratorAgentClass = getattr(banditagents, demonstratorSolver)
        learnerAgentClass = getattr(banditagents, learnerSolver)
        agents_config = {'demonstrator':{}, 'learner':{}}
        if learnerSolver in config['learners']:
            agents_config['learner'] = config['learners'][learnerSolver]
        if demonstratorSolver in config['demonstrators']:
            agents_config['demonstrator'] = config['demonstrators'][demonstratorSolver]
        agents_config['learner']['num_iterations'] = config['num_iterations']
        agents_config['demonstrator']['num_iterations'] = config['num_iterations']

        # p_trust and p_obs iteration``
        for k, p_trust in enumerate(config['ThompsonTrust_distributions']['p_trust']):
            p_obs = config['ThompsonTrust_distributions']['p_obs'][k]
            if j==0:
                print('p_trust:', p_trust, 'p_obs:', p_obs)

            agents_config['learner']['prob_trust'] = p_trust
            agents_config['learner']['prob_observe'] = p_obs
        
            # episode iteration
            for l in range(config['num_episodes']): 
                #print(f'episode: {episode}')
                if l == 0:
                     episodes_learner_arm_id_history, episodes_demonstrator_arm_id_history = None, None

                # initialize agents
                demonstrator = demonstratorAgentClass(agents_config['demonstrator'])   
                learner = learnerAgentClass(agents_config['learner'])

                episode_df, episodes_learner_arm_id_history, episodes_demonstrator_arm_id_history = run_episode(
                     demonstrator, learner, env, l, j,
                     episodes_learner_arm_id_history, episodes_demonstrator_arm_id_history,
                     )

                if i < 1 and j < 1 and k < 1 and l < 1:
                    varying_alpha_results_long_df = episode_df.copy()
                else:
                    varying_alpha_results_long_df = pd.concat([varying_alpha_results_long_df,episode_df],join='inner', ignore_index=True)

varying_alpha_results_long_df = varying_alpha_results_long_df.sort_values(by=['solver', 'episode', 'iteration'], ignore_index=True)
varying_alpha_results_long_df.to_csv('/Users/juliomartinez/Documents/PhD/socialmotivation/varying_alpha_results_long.csv')
learner_varying_alpha_results_long_df = varying_alpha_results_long_df.groupby('agent').get_group('learner')


# %%
# Plot Results
#fig, axs = plt.subplots(nrows=1, ncols=len(config['bandit']['alphas'][::2]), sharex=False, sharey=True, figsize=(20,5))
#for i, alpha in enumerate(config['bandit']['alphas'][::2]):
#    legend = None
#    if i==0:
#        legend = 'auto'
#    alpha_learner_results_long_df = learner_varying_alpha_results_long_df_df.groupby('alpha').get_group(alpha)
#    sns.lineplot(data=alpha_learner_results_long_df, x="iteration", y="delta_cumulative_reward", hue="group", kind='trust_distr', ax=axs[i], legend=legend)
#    axs[i].set_title('alpha = ' + str(alpha))
#plt.tight_layout()
#plt.savefig(os.path.join(figures_dir,'delta_cumulative_reward_per_alpha.png'),bbox_inches="tight")
#plt.show()

for name, group_df in learner_varying_alpha_results_long_df.groupby('group'):
    print(name)
    fig = plt.figure(figsize=(20,5))
    sns.lineplot(data=group_df[group_df['iteration'] == num_iterations-1], x="alpha", y="delta_cumulative_reward", hue='trust_distr')
    #plt.title('Varying payoff probability alpha, $(p_{arm1} = \\alpha, p_{arm2} = 1-\\alpha)$')
    plt.ylabel('Delta Total Reward')
    plt.title('Demonstrator: {}'.format(config['demonstrators']['solvers'][name]))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir,'delta_total_reward_per_group_{}.png'.format(name)), bbox_inches="tight")
    plt.show()







# %%
for i, num_arms in enumerate(config['bandit']['num_arms']):
    print(f'num_arms: {num_arms}')

    # setup environment
    bandit_config = {}
    bandit_config['payoffs'] =np.sort(np.squeeze(np.random.dirichlet(np.ones(num_arms),size=1)))[::-1] # largest payoff arm_id=0
    bandit_config['sampling_distribution'] = config['bandit']['sampling_distribution']
    env = bandits.Bandit(bandit_config)

    # solver iteration
    for j, (demonstratorSolver, learnerSolver) in enumerate(zip(config['demonstrators']['solvers'], config['learners']['solvers'])):
        print(f'demonstratorSolver: {demonstratorSolver}, ', f'learnerSolver: {learnerSolver}')

        # setup agent configs
        demonstratorAgentClass = getattr(banditagents, demonstratorSolver)
        learnerAgentClass = getattr(banditagents, learnerSolver)
        agents_config = {'demonstrator':{}, 'learner':{}}
        if learnerSolver in config['learners']:
            agents_config['learner'] = config['learners'][learnerSolver]
        if demonstratorSolver in config['demonstrators']:
            agents_config['demonstrator'] = config['demonstrators'][demonstratorSolver]
        agents_config['learner']['num_iterations'] = config['num_iterations']
        agents_config['demonstrator']['num_iterations'] = config['num_iterations']

        # p_trust and p_obs iteration``
        for k, p_trust in enumerate(config['ThompsonTrust_distributions']['p_trust']):
            p_obs = config['ThompsonTrust_distributions']['p_obs'][k]
            if j==0:
                print('p_trust:', p_trust, 'p_obs:', p_obs)

            agents_config['learner']['prob_trust'] = p_trust
            agents_config['learner']['prob_observe'] = p_obs
        
            # episode iteration
            for l in range(config['num_episodes']): 
                #print(f'episode: {episode}')
                if l == 0:
                     episodes_learner_arm_id_history, episodes_demonstrator_arm_id_history = None, None

                # initialize agents
                demonstrator = demonstratorAgentClass(agents_config['demonstrator'])   
                learner = learnerAgentClass(agents_config['learner'])

                episode_df, episodes_learner_arm_id_history, episodes_demonstrator_arm_id_history = run_episode(
                     demonstrator, learner, env, l, j,
                     episodes_learner_arm_id_history, episodes_demonstrator_arm_id_history,
                     )

                if i < 1 and j < 1 and k < 1 and l < 1:
                    varying_num_arms_results_long_df = episode_df.copy()
                else:
                    varying_num_arms_results_long_df = pd.concat([varying_num_arms_results_long_df,episode_df],join='inner', ignore_index=True)

varying_num_arms_results_long_df = varying_num_arms_results_long_df.sort_values(by=['solver', 'episode', 'iteration'], ignore_index=True)
varying_num_arms_results_long_df.to_csv('/Users/juliomartinez/Documents/PhD/socialmotivation/varying_num_arms_results_long.csv')
learner_varying_num_arms_results_long_df = varying_num_arms_results_long_df.groupby('agent').get_group('learner')

# %%
varying_num_arms_results_long_df = pd.read_csv('/Users/juliomartinez/Documents/PhD/socialmotivation/varying_num_arms_results_long.csv', index_col=0)
learner_varying_num_arms_results_long_df = varying_num_arms_results_long_df.groupby('agent').get_group('learner')

# %%

# %%
for name, group_df in learner_varying_num_arms_results_long_df.groupby('group'):
    print(name)
    fig = plt.figure(figsize=(20,5))
    sns.lineplot(data=group_df[group_df['iteration'] == num_iterations-1], x="num_arms", y="delta_cumulative_reward", hue='trust_distr')
    #plt.title('Varying payoff probability alpha, $(p_{arm1} = \\alpha, p_{arm2} = 1-\\alpha)$')
    plt.ylabel('Delta Total Reward')
    plt.title('Demonstrator: {}'.format(config['demonstrators']['solvers'][name]))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir,'delta_total_reward_per_group_{}.png'.format(name)), bbox_inches="tight")
    plt.show()

# %%
for name, group_df in varying_num_arms_results_long_df.groupby('group'):
    print(name)
    fig = plt.figure(figsize=(20,5))
    sns.lineplot(data=group_df[group_df['iteration'] == num_iterations-1], x="num_arms", y="cumulative_reward", hue='trust_distr', style='solver')
    #plt.title('Varying payoff probability alpha, $(p_{arm1} = \\alpha, p_{arm2} = 1-\\alpha)$')
    plt.ylabel('Cumulative Reward')
    plt.title('Demonstrator: {}'.format(config['demonstrators']['solvers'][name]))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir,'delta_total_reward_per_group_{}.png'.format(name)), bbox_inches="tight")
    plt.show()

# %%
episode_df = varying_num_arms_results_long_df.groupby(['episode', 'trust_distr']).get_group((39, 'p_trust: 1.0, p_obs: 0.0'))

for name, group_df in episode_df.groupby(['group', 'num_arms']):
    print(name)
    fig = plt.figure(figsize=(20,5))
    sns.stripplot(data=group_df, x="iteration", y="chosen_arm_id", hue='solver')
    #plt.title('Varying payoff probability alpha, $(p_{arm1} = \\alpha, p_{arm2} = 1-\\alpha)$')
    plt.ylabel('Arm ID')
    plt.title('Demonstrator: {}, Num Arms: {}'.format(config['demonstrators']['solvers'][name[0]], name[1] ))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir,'Examples_{}.png'.format(name)), bbox_inches="tight")
    plt.show()
# %%
