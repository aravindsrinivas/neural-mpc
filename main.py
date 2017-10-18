import numpy as np
import sys
sys_paths = ['/Users/aravind/drl/mjc/lib/python3.6/site-packages',
 '',
 '/Users/aravind/anaconda/bin',
 '/Users/aravind/anaconda/lib/python3.6/site-packages',
 '/usr/local/lib/python3.6/site-packages',
 '/usr/local/Cellar/python3/3.6.2/bin/python3.6',
 '/Users/aravind/rllab',
 '/Users/aravind/homework/hw4',
 '/Users/aravind/Lasagne',
 '/Users/aravind/anaconda/lib/python36.zip',
 '/Users/aravind/anaconda/lib/python3.6',
 '/Users/aravind/anaconda/lib/python3.6/lib-dynload',
 '/Users/aravind/anaconda/lib/python3.6/site-packages/Sphinx-1.5.6-py3.6.egg',
 '/Users/aravind/anaconda/lib/python3.6/site-packages/aeosa',
 '/Users/aravind/Theano',
 '/Users/aravind/anaconda/lib/python3.6/site-packages/gym-0.9.3-py3.6.egg',
 '/Users/aravind/anaconda/lib/python3.6/site-packages/setuptools-27.2.0-py3.6.egg',
 '/Users/aravind/anaconda/lib/python3.6/site-packages/IPython/extensions',
 '/Users/aravind/.ipython',
 '/Users/aravind/gym']

for syspath in sys_paths:
    sys.path.append(sys_paths)
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew



gamma = 1.

def sample(env, 
           controller, 
           num_paths=10, 
           horizon=1000, 
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller), 
        and returns rollouts by running on the env. 
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    """ YOUR CODE HERE """
    for _ in range(num_paths):
        path = {}
        path_obs = []
        path_next_obs = []
        path_rewards = []
        path_returns = []
        path_actions = []
        path_return = 0
        obs = env.reset()
        for step in range(horizon):
            action = controller.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            path_return += reward
            path_obs.append(obs)
            path_next_obs.append(next_obs)
            path_rewards.append(reward)
            path_actions.append(action)
            path_rewards.append(reward)
            #path_returns.append(path_return)
            obs = next_obs
            if done:
                break
        path['observations'] = path_obs
        path['next_observations'] = path_next_obs
        path['rewards'] = path_rewards
        path['returns'] = path_returns
        path['actions'] = path_actions
        path['return'] = path_return
        #path['returns'] = scipy.signal.lfilter([1],[1,-gamma], path['rewards'][::-1], axis = 0)[::-1]
        paths.append(path)
    return paths

#path is a dictionary and path is a list of paths and the value for each key is a list/array.

# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])

def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
    """

    """ YOUR CODE HERE """
    obs = []
    next_obs = []
    deltas = []
    actions = []
    for path in data:
        obs += path['observations']
        next_obs += path['next_observations']
        actions += path['actions']
    
    obs = np.array(obs)
    next_obs = np.array(next_obs)
    actions = np.array(actions)
    deltas = next_obs - obs
    mean_obs = np.mean(obs, axis=0)
    std_obs = np.std(obs, axis=0)
    mean_deltas = np.mean(deltas, axis=0)
    std_deltas = np.std(deltas, axis=0)
    mean_action = np.mean(actions, axis=0)
    std_action = np.std(actions, axis=0)

    return (mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action)


def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions. 
    """
    """ YOUR CODE HERE """
    pass

def train(env, 
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=10,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10, 
         num_paths_onpol=10, 
         num_simulated_paths=10000,
         env_horizon=1000, 
         mpc_horizon=15,
         n_layers=2,
         size=500,
         activation=tf.nn.relu,
         output_activation=None
         ):

    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run. 

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated 
    |                           by a random agent. We use these to train our 
    |_                          initial dynamics model.
    
    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining 
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments. 

    """

    logz.configure_output_dir(logdir)

    #========================================================
    # 
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.

    random_controller = RandomController(env)

    """ YOUR CODE HERE """
    paths = sample(env,
           random_controller,
           num_paths=num_paths_random,
           horizon=mpc_horizon,
           render=False,
           verbose=False)
    

    #========================================================
    # 
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network. 
    # 
    #normalization = """ YOUR CODE HERE """
    normalization = compute_normalization(paths)

    #========================================================
    # 
    # Build dynamics model and MPC controllers.
    # 
    sess = tf.Session()

    dyn_model = NNDynamicsModel(env=env, 
                                n_layers=n_layers, 
                                size=size, 
                                activation=activation, 
                                output_activation=output_activation, 
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env, 
                                   dyn_model=dyn_model, 
                                   horizon=mpc_horizon, 
                                   cost_fn=cost_fn, 
                                   num_simulated_paths=num_simulated_paths)


    #========================================================
    # 
    # Tensorflow session building.
    # 
    sess.__enter__()
    tf.global_variables_initializer().run()

    #========================================================
    # 
    # Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset. 
    # Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
    # 
    for itr in range(onpol_iters):
        """ YOUR CODE HERE """

        dyn_model.fit(paths)
        new_paths = sample(env,mpc_controller, num_paths=num_paths_mpc,horizon=mpc_horizon,render=False,verbose=False)
        paths = paths + new_paths # Aggregation
        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))

        logz.dump_tabular()

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=1)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=60)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=10)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name is "HalfCheetah-v1":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    train(env=env, 
                 cost_fn=cost_fn,
                 logdir=logdir,
                 render=args.render,
                 learning_rate=args.learning_rate,
                 onpol_iters=args.onpol_iters,
                 dynamics_iters=args.dyn_iters,
                 batch_size=args.batch_size,
                 num_paths_random=args.random_paths, 
                 num_paths_onpol=args.onpol_paths, 
                 num_simulated_paths=args.simulated_paths,
                 env_horizon=args.ep_len, 
                 mpc_horizon=args.mpc_horizon,
                 n_layers = args.n_layers,
                 size=args.size,
                 activation=tf.nn.relu,
                 output_activation=None,
                 )

if __name__ == "__main__":
    main()
