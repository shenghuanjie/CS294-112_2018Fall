#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import load_policy
import tf_util
from matplotlib import pyplot as plt
import os

import builtins as __builtin__
import pandas as pd
import argparse


savedir = './report'


def print(*args, **kwargs):
    """My custom print() function."""
    # Adding new arguments to the print function signature
    # is probably a bad idea.
    # Instead consider testing if custom argument keywords
    # are present in kwargs
    tempargs = list(args)
    for iarg, arg in enumerate(tempargs):
        if type(arg).__module__ == np.__name__:
            tempargs[iarg] = bmatrix(arg)
        elif isinstance(arg, pd.DataFrame):
            tempargs[iarg] = btabu(arg)
        elif isinstance(arg, argparse.Namespace):
            tempargs[iarg] = bargs(arg)
        elif isinstance(arg, str):
            #if '\\' in arg:
            #    arg = arg.replace('\\', r' \textbackslash ')
            if '_' in arg:
                arg = arg.replace('_', r'\_')
            if '<' in arg:
                arg = arg.replace('<', r'\textless ')
            if '>' in arg:
                arg = arg.replace('>', r'\textgreater ')
            if '<=' in arg:
                arg = arg.replace('<=', r'\le ')
            if '>=' in arg:
                arg = arg.replace('>=', r'\ge ')
            tempargs[iarg] = arg
        else:
            tempargs[iarg] = str(arg).replace('_', r'\_')
    tempargs = tuple(tempargs)
    __builtin__.print(*tempargs, **kwargs, end='')
    __builtin__.print(r' \\', **kwargs)


def bargs(a):
    a_dict = vars(a)
    a_keys = list(a_dict.keys())
    rv = [r'\\']
    for iKey in range(len(a_keys)):
        rv += [a_keys[iKey] + ': ' + str(a_dict[a_keys[iKey]]) + r' \\']
    rv += [r'\\']
    return '\n'.join(rv)


def bmatrix(a):
    """Returns a LaTeX bmatrix
    Retrieved from https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    a = np.array(a)
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\[']
    rv += [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    rv += [r'\]']
    return '\n'.join(rv)


def btabu(a):
    nCol = len(a.columns)
    rv = [r'\begin{tabu} to 1.0\textwidth {  ' + '|X[c] ' * (nCol + 1) + '| }']
    rv += [r'\hline']
    currentRow = ' '
    for idx, column in enumerate(a.columns):
        currentRow += ' & ' + str(column)
    rv += [currentRow + '\\\\']
    for idx, row in a.iterrows():
        currentRow = str(idx) + ' '
        for _, column in enumerate(a.columns):
            currentRow += ' & ' + str(row[column])
        rv += [r'\hline']
        rv += [currentRow + '\\\\']
    rv += [r'\hline']
    rv += [r'\end{tabu}\\']
    return '\n'.join(rv)


class Imitator(object):
    """
        This class is used for behavior cloning
        Basically, it runs supervised learning algorithm with a simple
        neural network with two hidden layers
    """
    def __init__(self, input_shape):
        self.env_shape = input_shape
        self.model = self.build_model()
        self.history = []

    def build_model(self, hidden_layers=2, layer_size=64, activation_func=tf.nn.relu):
        model = keras.Sequential()
        model.add(keras.layers.Dense(layer_size, activation=activation_func,
                                     input_shape=(self.env_shape[0],)))
        for iLayer in range(hidden_layers-1):
            model.add(keras.layers.Dense(layer_size, activation=activation_func))
        model.add(keras.layers.Dense(self.env_shape[1]))
        optimizer = tf.train.RMSPropOptimizer(0.001)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae'])
        return model

    def fit(self, train_data, train_labels, epochs=200, validation_split=0):
        self.history = self.model.fit(train_data, train_labels, epochs=epochs,
                                      validation_split=validation_split, verbose=0)

    def predict(self, test_data):
        test_predictions = np.atleast_2d(self.model.predict(test_data))
        return test_predictions

    def plot_history(self):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [1000$]')
        if 'mean_absolute_error' in self.history:
            plt.plot(self.history.epoch, np.array(self.history.history['mean_absolute_error']),
                     label='Train Loss')
        if 'val_mean_absolute_error' in self.history:
            plt.plot(self.history.epoch, np.array(self.history.history['val_mean_absolute_error']),
                     label='Val loss')
        plt.legend()
        plt.ylim([0, 5])
        plt.show()


class DAgger(Imitator):

    # Override
    def __init__(self, input_shape, policy_fn, args):
        super().__init__(input_shape)
        self.policy_fn = policy_fn
        self.args = args
        self.histories = []

    # Override
    def fit(self, train_data, train_labels, epochs=200, validation_split=0, runs=1, track=False):
        plot_data = []
        for iRun in range(runs):
            if self.args.verbose:
                print('DAgger Run', iRun)
            # fit behavior cloning
            self.model = self.build_model()
            super().fit(train_data, train_labels, epochs, validation_split)
            self.histories.append(self.history)
            # get expert opinions
            expert_data, expert_labels = self._get_expert_opinions()
            # data augmentation
            train_data = np.vstack((train_data, expert_data))
            train_labels = np.vstack((train_labels, expert_labels))
            if self.args.verbose:
                print('train_data.shape', train_data.shape)
            if track:
                _, dagger_returns = run_policy(self.args, self.predict)
                plot_data.append(dagger_returns)
        return plot_data

    # Override
    def plot_history(self, history=True):
        if history:
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Abs Error [1000$]')
            all_epoches = []
            last_epoch = 0
            for ih in range(len(self.histories)):
                all_epoches.append(np.array(self.histories[ih].epoch) + last_epoch)
                last_epoch += self.histories[ih].epoch[-1]
            all_epoches = np.array(all_epoches).flatten()

            if 'mean_absolute_error' in self.histories[ih].history:
                plt.plot(all_epoches,
                         np.array([np.array(self.histories[ih].history['mean_absolute_error'])
                                    for ih in range(len(self.histories))]).flatten(),
                         label='Train Loss')

            if 'val_mean_absolute_error' in self.histories[ih].history:
                plt.plot(all_epoches,
                         np.array([np.array(self.histories[ih].history['val_mean_absolute_error'])
                                   for ih in range(len(self.histories))]).flatten(),
                         label='Val loss')

            plt.legend()
            plt.ylim([0, 5])
            plt.show()
        else:
            super().plot_history()

    # Private
    def _get_expert_opinions(self):
        # run behavior cloning
        behavior_data, _ = run_policy(self.args, self.predict)
        observations = behavior_data['observations']
        # get expert actions
        expert_actions = []
        with tf.Session():
            tf_util.initialize()
            for iObs in range(observations.shape[0]):
                obs = observations[iObs, :]
                expert_action = self.policy_fn(obs[None, :])
                expert_actions.append(expert_action)
        expert_actions = np.squeeze(np.array(expert_actions))
        return observations, expert_actions


def generate_rollout_data(args, policy_fn):
    with tf.Session():
        tf_util.initialize()
        expert_data, returns = run_policy(args, policy_fn)
    return expert_data, returns


def run_policy(args, policy_fn):
    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    for i in range(args.num_rollouts):
        if args.verbose:
            print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0 and args.verbose:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    if args.verbose:
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    '''
        The original code returns a 3D array for actions
        Here I convert it to a 2D array

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}
    '''
    expert_data = {'observations': np.array(observations),
                   'actions': np.squeeze(np.array(actions))}

    '''
    if not os.path.exists(os.path.join('.', 'expert_data')):
        os.makedirs(os.path.join('.', 'expert_data'))
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
    '''
    return expert_data, returns


def setup_mujoco():
    #import sys
    mujoco_path = os.path.join(os.path.expanduser('~'), r'.mujoco\mjpro150\bin')
    #sys.path.append(mujoco_path)
    os.environ["PATH"] += r';' + mujoco_path + ';'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def modify_args(parser, args, **kwargs):
    dict_args = vars(args)
    for key, value in kwargs.items():
        dict_args[key] = value
    return dict2args(parser, dict_args)


def dict2args(parser, dict_args):
    str_args = []
    for key in list(dict_args.keys()):
        if isinstance(dict_args[key], bool):
            if dict_args[key]:
                str_args.append('--' + key)
        else:
            str_args.append('--' + key)
            str_args.append(str(dict_args[key]))
    return parser.parse_args(str_args)


def setup_inputs():
    """
    I don't use terminal but PyCharm on Windows
    Therefore I need to parse everything here before returning it
    :return: parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, required=True)
    parser.add_argument('--envname', type=str, required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--epochs', type=int, default=200,
                        help='The number of epoches for behavioral cloning')
    parser.add_argument('--dagger_runs', type=int, default=5,
                        help='The number of dagger cycles')

    # custom arguments
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--show_result", action='store_true')

    # select section to run
    parser_algorithm = parser.add_mutually_exclusive_group(required=True)
    parser_algorithm.add_argument('--demo', action='store_true')
    parser_algorithm.add_argument('--section22', action='store_true')
    parser_algorithm.add_argument('--section23', action='store_true')
    parser_algorithm.add_argument('--section32', action='store_true')
    parser_algorithm.add_argument('--section41', action='store_true')

    return parser


def show_results(parser, args, behavior):
    args = modify_args(parser, args, render=True)
    generate_rollout_data(args, behavior.predict)


def load_expert_policy(args):
    if args.verbose:
        print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    if args.verbose:
        print('loaded and built')
    return policy_fn


def get_expert_data_dim(expert_data):
    input_shape = (expert_data['observations'].shape[1], expert_data['actions'].shape[1])
    return input_shape


def demo(parser, inputs):
    args = modify_args(parser, inputs, render=True)
    policy_fn = load_expert_policy(args)
    generate_rollout_data(args, policy_fn)


def section2_2(parser, inputs):
    """
    run behavioral cloning on all tasks and save the results
    as text file which can be directly copied into latex
    :return:
    """
    envnames = os.listdir('./experts')
    print_envnames = []
    print_returns = []
    for iEnv in range(len(envnames)):
        # get the current environmental name
        envname1 = os.path.splitext(envnames[iEnv])[0]
        print_envnames.append(envname1)
        args = modify_args(parser, inputs,
                           expert_policy_file='./experts/' + envnames[iEnv],
                           envname=envname1)
        if inputs.verbose:
            print('Running ' + envname1)

        # get expert policy
        policy_fn = load_expert_policy(args)
        # run expert policy and save returns
        expert_data, expert_returns = generate_rollout_data(args, policy_fn)

        # get expert data dimensions
        dims = get_expert_data_dim(expert_data)
        # use behavioral cloning to mimic expert policy
        behavioral_cloner = Imitator(dims)
        behavioral_cloner.fit(expert_data['observations'], expert_data['actions'], epochs=args.epochs)
        # run behavioral cloner and save returns
        _, cloner_returns = run_policy(args, behavioral_cloner.predict)

        print_returns.append([np.mean(expert_returns), np.std(expert_returns),
                              np.mean(cloner_returns), np.std(cloner_returns)])

    df = pd.DataFrame(np.array(print_returns),
                      columns=['exp(mean)', 'exp(std)', 'bc(mean)', 'bc(std)'],
                      index=print_envnames)

    with open(os.path.join(savedir, '/section22.txt'), 'w') as f:
        print(args, file=f)
        print(df, file=f)


def section2_3(parser, inputs, hyper='epochs', spectrum=range(50, 500, 50)):
    """
    run behavioral cloning on one task from above
    test the effect of a hyperparameter and report the figure
    :return:
    """
    # get expert policy
    policy_fn = load_expert_policy(inputs)
    # run expert policy and save returns
    expert_data, expert_returns = generate_rollout_data(inputs, policy_fn)

    # get expert data dimensions
    dims = get_expert_data_dim(expert_data)

    # store data for plotting
    plot_data = []

    for num in spectrum:
        args = modify_args(parser, inputs,
                           **{hyper: num})

        # use behavioral cloning to mimic expert policy
        if args.verbose:
            print("Testing: {}({})".format(hyper, num))
        behavioral_cloner = Imitator(dims)
        behavioral_cloner.fit(expert_data['observations'], expert_data['actions'], epochs=args.epochs)
        # run behavioral cloner and save returns
        _, cloner_returns = run_policy(args, behavioral_cloner.predict)
        plot_data.append(cloner_returns)

    plot_data = np.array(plot_data)
    plt.figure()
    plt.errorbar(spectrum, np.mean(plot_data, axis=1), yerr=np.std(plot_data, axis=1))
    plt.title("Section 2.3: Hyperparameter tuning of behavioral cloning")
    plt.xlabel(hyper)
    plt.ylabel(r"policy's returns")
    plt.savefig(os.path.join(savedir, 'section23.png'))


def section3_2(parser, inputs):
    """
    run DAgger and plot the policy's mean return
    with behavioral cloning and expert policy as well
    :return:
    """
    # get expert policy
    policy_fn = load_expert_policy(inputs)
    # run expert policy and save returns
    expert_data, expert_returns = generate_rollout_data(inputs, policy_fn)
    # get expert data dimensions
    dims = get_expert_data_dim(expert_data)

    # run behavioral cloning as a comparison
    behavioral_cloner = Imitator(dims)
    behavioral_cloner.fit(expert_data['observations'], expert_data['actions'])
    _, cloner_returns = run_policy(inputs, behavioral_cloner.predict)

    # store returns
    # run DAgger
    dagger = DAgger(dims, policy_fn, inputs)
    plot_data = dagger.fit(expert_data['observations'], expert_data['actions'],
                           epochs=inputs.epochs, runs=inputs.dagger_runs, track=True)
    # dagger.plot_history(history=True)
    '''
    _, dagger_returns = run_policy(inputs, dagger.predict)
    plot_data.append(dagger_returns)
    for iRun in range(inputs.dagger_runs - 1):
        dagger.fit([], [], epochs=inputs.epochs, runs=1)
        if inputs.verbose:
            print('Current DAgger Dataset Shape: {}'.format(dagger.train_data.shape))
        # run behavioral cloner and save returns
        _, dagger_returns = run_policy(inputs, dagger.predict)
        plot_data.append(dagger_returns)
    '''
    plot_data = np.array(plot_data)
    plt.figure()
    plot_x = np.linspace(1, inputs.dagger_runs, inputs.dagger_runs)
    plt.errorbar(plot_x, np.mean(plot_data, axis=1), yerr=np.std(plot_data, axis=1), label='DAgger')
    plt.plot([1, inputs.dagger_runs], [np.mean(expert_returns), np.mean(expert_returns)],
             color='k', linestyle='-', linewidth=2, label='Expert')
    plt.plot([1, inputs.dagger_runs], [np.mean(cloner_returns), np.mean(cloner_returns)],
             color='r', linestyle='-', linewidth=2, label='Behavioral Cloning')
    plt.fill_between(plot_x,
                     np.mean(expert_returns) - np.std(expert_returns),
                     np.mean(expert_returns) + np.std(expert_returns),
                     alpha=0.3, color='k')
    plt.fill_between(plot_x,
                     np.mean(cloner_returns) - np.std(cloner_returns),
                     np.mean(cloner_returns) + np.std(cloner_returns),
                     alpha=0.3, color='r')
    plt.xlim(0, inputs.dagger_runs + 1)
    plt.title("Section 3.2: Comparing DAgger, Behavioral cloning, and Expert")
    plt.xlabel('DAgger iterations')
    plt.ylabel(r"policy's returns")
    plt.legend(loc='best')
    plt.savefig(os.path.join(savedir, 'section32.png'))


def section4_1(parser, inputs):
    """
    run behavioral cloning on one task from above
    test the effect of a different policy architecture
    by changing the number of layers, the
    :return:
    """
    # get expert policy
    policy_fn = load_expert_policy(inputs)
    # run expert policy and save returns
    expert_data, expert_returns = generate_rollout_data(inputs, policy_fn)

    # get expert data dimensions
    dims = get_expert_data_dim(expert_data)

    # store returns
    print_returns = []
    print_returns.append([np.mean(expert_returns), np.std(expert_returns)])

    # use behavioral cloning to mimic expert policy
    behavioral_cloner = Imitator(dims)

    # fit model 1 using default settings
    # hidden_layers=2, layer_size=64, activation_func=tf.nn.relu
    behavioral_cloner.fit(expert_data['observations'], expert_data['actions'], epochs=inputs.epochs)
    # run behavioral cloner and save returns
    _, cloner_returns = run_policy(inputs, behavioral_cloner.predict)
    print_returns.append([np.mean(cloner_returns), np.std(cloner_returns)])

    # fit model 2 with the following settings
    # hidden_layers=6, layer_size=64, activation_func=tf.nn.relu
    behavioral_cloner.model = behavioral_cloner.build_model(hidden_layers=6)
    behavioral_cloner.fit(expert_data['observations'], expert_data['actions'], epochs=inputs.epochs)
    # run behavioral cloner and save returns
    _, cloner_returns = run_policy(inputs, behavioral_cloner.predict)
    print_returns.append([np.mean(cloner_returns), np.std(cloner_returns)])

    # fit model 3 with the following settings
    # hidden_layers=2, layer_size=16, activation_func=tf.nn.relu
    behavioral_cloner.model = behavioral_cloner.build_model(layer_size=16)
    behavioral_cloner.fit(expert_data['observations'], expert_data['actions'], epochs=inputs.epochs)
    # run behavioral cloner and save returns
    _, cloner_returns = run_policy(inputs, behavioral_cloner.predict)
    print_returns.append([np.mean(cloner_returns), np.std(cloner_returns)])

    # fit model 4 with the following settings
    # hidden_layers=2, layer_size=64, activation_func=tf.nn.tanh
    behavioral_cloner.model = behavioral_cloner.build_model(activation_func=tf.nn.tanh)
    behavioral_cloner.fit(expert_data['observations'], expert_data['actions'], epochs=inputs.epochs)
    # run behavioral cloner and save returns
    _, cloner_returns = run_policy(inputs, behavioral_cloner.predict)
    print_returns.append([np.mean(cloner_returns), np.std(cloner_returns)])

    # convert and save data frame
    print_returns = np.array(print_returns)
    df = pd.DataFrame(print_returns,
                      columns=['mean', 'std'],
                      index=['expert', 'original', 'hidden_layers=6',
                             'layer_size=16', 'activation_func=tt.nn.tanh'])

    with open(os.path.join(savedir, 'section41.txt'), 'w') as f:
        print(inputs, file=f)
        print(df, file=f)

    plt.figure()
    plt.bar(range(1, print_returns.shape[0] + 1), print_returns[:, 0],
            yerr=print_returns[:, 1],
            tick_label=['expert', 'original', 'hidden_layers=6',
                        'layer_size=16', 'activation_func=tt.nn.tanh'])
    plt.xlim([0, print_returns.shape[0] + 1])
    plt.xticks(range(1, print_returns.shape[0] + 1),
               ['expert', 'original', 'hidden_layers=6',
                'layer_size=16', 'activation_func=tt.nn.tanh'], rotation=45)
    plt.title("Section 4.1: The effect of policy architecture on behavioral cloning")
    plt.xlabel(r'policy architecture')
    plt.ylabel(r"policy's returns")
    plt.savefig(os.path.join(savedir, 'section41.png'))


def main():
    setup_mujoco()
    parser = setup_inputs()
    args = parser.parse_args(['--expert_policy_file', './experts/Humanoid-v2.pkl',
                              '--envname', 'Humanoid-v2',
                              '--num_rollouts', '20',
                              '--max_timesteps', '1000',
                              '--epochs', '500',
                              '--dagger_runs', '10',
                              '--section32'])

    if args.demo:
        demo(parser, args)

    if args.section22:
        section2_2(parser, args)

    if args.section23:
        section2_3(parser, args)

    if args.section32:
        section3_2(parser, args)

    if args.section41:
        section4_1(parser, args)

    '''
    # This tells you how to run behavioral cloner
    behavioral_cloner = Imitator(dims)
    behavioral_cloner.fit(expert_data['observations'], expert_data['actions'])
    behavioral_cloner.plot_history()
    behavior = behavior_cloner
    '''
    '''
    # This tells you how to run DAgger
    dagger = DAgger(dims, policy_fn, args)
    dagger.fit(expert_data['observations'], expert_data['actions'])
    dagger.plot_history()
    behavior = dagger
    '''
    '''
    # This is how you run your learned policy
    run_policy(args, behavior.predict)
    '''
    # show_results(parser, args, behavior)


def test():
    """
    This file is here to show the journey of my debugging
    :return:
    """
    '''
    a = None
    b = np.ones((10, 10))
    if a is not None:
        b = np.vstack((b, np.atleast_2d(a)))
    print(b)
    '''
    '''
    print(os.path.abspath(os.path.join(savedir, '/section222.txt')))
    print(os.path.join(savedir, 'section222.txt'))
    with open(os.path.join(savedir, 'section222.txt'), 'w') as f:
        print("sadas", file=f)
    '''

    # print("Testing: {}({})".format("ewew", 5))

    plt.figure()
    h1 = plt.errorbar(range(10), range(10), yerr=range(10), label='a')
    h2 = plt.plot(range(10), label='b')
    h3 = plt.plot(range(10, 20), label='c')
    plt.fill(range(10, 20), range(10, 20))
    plt.legend()
    plt.show()
    '''
        a = []
        for i in range(5):
            a.append([1, 2, 3])

        print(np.mean(np.array(a), axis=1))
        '''
    '''
    parser = setup_inputs()
    args = parser.parse_args(['--expert_policy_file', './experts/Humanoid-v2.pkl',
                              '--envname', 'Humanoid-v2',
                              '--num_rollouts', '3',
                              '--max_timesteps', '5',
                              '--epochs', '5',
                              '--dagger_runs', '5',
                              '--section41'
                              ])
    #print(args)
    print(modify_args(parser, args, epochs=20))
    '''
    '''
    df = pd.DataFrame(np.random.randint(1, 10, (5, 5)),
                      columns=['a', 'b', 'c', 'd', 'e'])
    print(df, file=f)
    '''


if __name__ == '__main__':
    main()
    # test()



