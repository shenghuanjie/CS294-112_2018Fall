# dagger
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
import os
import load_policy
from tensorflow.keras import backend as K
import tf_util

import gym
import argparse


mujoco_path = os.path.join(os.path.expanduser('~'), r'.mujoco\mjpro150\bin')
#sys.path.append(mujoco_path)
os.environ["PATH"] += r';' + mujoco_path + ';'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# parser = argparse.ArgumentParser()
# parser.add_argument('data', type=str)
# args = parser.parse_args()

data_list = os.listdir("expert_data")
data_dict = {i[:-4]:pickle.load(open("expert_data/"+i, "rb")) for i in data_list}

def build_model(data):
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(data["observations"].shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(data["actions"][0].shape[1])
        ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss="mse",
                    optimizer=optimizer,
                    metrics=["mae"])
    return model

def fit_model(data, EPOCHS):
    model = build_model(data)
    history = model.fit(data["observations"], data["actions"].reshape(
        data["actions"].shape[0], data["actions"].shape[2]), 
        epochs=EPOCHS, validation_split=0, verbose=0)
    return model

def run_policy(env, model, rollout=3, max_steps=100):
    returns = []
    observations = []
    env = gym.make(env)
    for i in range(rollout):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs[None,:])
            observations.append(obs)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    return returns, observations

Humanoid_expert = "experts/Humanoid-v2.pkl"

def generate_rollout(num_rollouts=5, envname='Humanoid-v2', max_timesteps=100):
    policy_fn = load_policy.load_policy(Humanoid_expert)
    with tf.Session() as sess:
        tf_util.initialize()
        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []

        for i in range(num_rollouts):
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
                #if steps % 100 == 0:
                #    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

    return expert_data, returns

#print(data_dict["Humanoid-v2"])

Humanoid_data, expert_returns = generate_rollout()
model = fit_model(Humanoid_data, 200)
cloner_returns, _ = run_policy("Humanoid-v2", model)


# Humanoid
res = []
with tf.Session() as sess:
    K.set_session(sess)
    tf_util.initialize()
    policy_fn = load_policy.load_policy(Humanoid_expert)
    for i in range(10):
        # train policy from human data D
        model = fit_model(Humanoid_data, 200)   # epoch
        # run policy to get dataset (observations)
        performance, observations = run_policy("Humanoid-v2", model)
        print(np.mean(performance), np.std(performance))
        res.append((i, np.mean(performance), np.std(performance)))

        # ask human to label D with Action A
        actions = []
        for obs in observations:
            actions.append(policy_fn(obs[None,:]))
        # aggregate
        
        obs = np.concatenate((Humanoid_data["observations"], np.array(observations)))
        acts = np.concatenate((Humanoid_data["actions"], np.array(actions)))
        Humanoid_data = {"observations": obs, "actions": acts}
        print(Humanoid_data["observations"].shape)
        print(Humanoid_data["actions"].shape)


#print(np.array(range(1, len(res)+1)).shape)
#print(np.array([i[2] for i in res]).shape)

plt.errorbar(range(1, len(res)+1), [i[1] for i in res], [i[2] for i in res], label="Dagger")
plt.plot(range(1, len(res)+1), [np.mean(cloner_returns)]*len(res), label="Behavioral Cloning")
plt.plot(range(1, len(res)+1), [np.mean(expert_returns)]*len(res), label="Expert")
plt.title("Avg Rewards against Iterations for Humanoid-v2")
plt.xlabel("iteration")
plt.ylabel("rewards")
plt.legend()
plt.show()






