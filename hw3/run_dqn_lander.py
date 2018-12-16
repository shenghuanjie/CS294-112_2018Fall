import os.path as osp

import tensorflow.contrib.layers as layers
from gym import wrappers

import dqn
from dqn_utils import *


def lander_model(obs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def lander_optimizer():
    return dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(1e-3),
        kwargs={}
    )


def lander_stopping_criterion(num_timesteps):
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    return stopping_criterion


# YOUR OWN CODE
def lander_exploration_schedule(num_timesteps, schedule='PiecewiseSchedule'):
    if schedule == 'ConstantSchedule':
        return ConstantSchedule(0.02)
    elif schedule == 'LinearSchedule':
        return LinearSchedule(
            num_timesteps,
            0.02
        )
    else:
        return PiecewiseSchedule(
            [
                (0, 1),
                (num_timesteps * 0.1, 0.02),
            ], outside_value=0.02
        )


def lander_kwargs():
    return {
        'optimizer_spec': lander_optimizer(),
        'q_func': lander_model,
        'replay_buffer_size': 50000,
        'batch_size': 32,
        'gamma': 1.00,
        'learning_starts': 1000,
        'learning_freq': 1,
        'frame_history_len': 1,
        'target_update_freq': 3000,
        'grad_norm_clipping': 10,
        'lander': True
    }


def lander_learn(env,
                 session,
                 num_timesteps,
                 # YOUR OWN CODE
                 seed,
                 doubleQ=True,
                 exp_name='doubleQ',
                 schedule='PiecewiseSchedule',
                 rew_file='lander_test.pk1'):
    # optimizer = lander_optimizer()
    # stopping_criterion = lander_stopping_criterion(num_timesteps)
    # exploration_schedule = lander_exploration_schedule(num_timesteps)

    dqn.learn(
        env=env,
        session=session,
        exploration=lander_exploration_schedule(num_timesteps, schedule),
        stopping_criterion=lander_stopping_criterion(num_timesteps),
        double_q=doubleQ,
        # YOUR OWN CODE
        rew_file=rew_file,
        seed=seed,
        env_name='LunarLander-v2',
        exp_name=exp_name,
        **lander_kwargs()
    )
    env.close()


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count={'GPU': 0})
    # GPUs don't significantly speed up deep Q-learning for lunar lander,
    # since the observations are low-dimensional
    session = tf.Session(config=tf_config)
    return session


def get_env(seed):
    env = gym.make('LunarLander-v2')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

    return env


def setup_inputs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='DQN')
    parser.add_argument('--doubleQ', action='store_true')
    parser.add_argument('--schedule', type=str, default='PiecewiseSchedule')
    return parser.parse_args()


def main():
    # YOUR OWN CODE
    args = setup_inputs()
    # Run training
    seed = 86252 #4565 # you may want to randomize this
    print('random seed = %d' % seed)
    env = get_env(seed)
    session = get_session()
    set_global_seeds(seed)
    # YOUR OWN CODE
    # Q2,3
    lander_learn(env, session, num_timesteps=500000,
                 seed=seed, doubleQ=args.doubleQ, exp_name=args.exp_name, schedule=args.schedule,
                 rew_file='lander_' + args.exp_name + '.pk1')


if __name__ == "__main__":
    main()
