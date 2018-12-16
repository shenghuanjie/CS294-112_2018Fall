## Deep Q-learning
# Question 1
# To run the analysis, use the following line
python run_dqn_atari.py

# To plot the result, use the following line
python multiplot.py data\qlearn_exp_PongNoFrameskip-v4_24-09-2018_10-10-51 --value best_mean_reward mean_reward_(100_episodes) --legend best_mean_reward mean_reward_(100_episodes)


# Question 2
# To run the analysis, use the following line
python run_dqn_lander.py
python run_dqn_lander.py --doubleQ --exp_name doubleQ

# To plot the result, use the following line
python plot.py data\qlearn_DQN_LunarLander-v2_25-09-2018_22-47-04 data\qlearn_doubleQ_LunarLander-v2_27-09-2018_18-56-47 --value best_mean_reward


# Question 3
# To run the analysis, use the following line
python run_dqn_lander.py
python run_dqn_lander.py --schedule ConstantSchedule --exp_name ConstantDQN
python run_dqn_lander.py --schedule LinearSchedule --exp_name LinearDQN

# To plot the result, use the following line
python plot.py data\qlearn_DQN_LunarLander-v2_25-09-2018_22-47-04 data\qlearn_ConstantDQN_LunarLander-v2_25-09-2018_23-52-17 data\qlearn_LinearDQN_LunarLander-v2_26-09-2018_00-15-31 --value best_mean_reward



## Actor-critic
# Question 1.
# To run the analysis, use the following line
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_1 -ntu 1 -ngsptu 1
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 100_1 -ntu 100 -ngsptu 1
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_100 -ntu 1 -ngsptu 100
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 10_10 -ntu 10 -ngsptu 10

# To plot the result, use the following line
python plot.py data\ac_1_1_CartPole-v0_27-09-2018_19-36-12 data\ac_1_100_CartPole-v0_27-09-2018_19-42-41 data\ac_10_10_CartPole-v0_27-09-2018_19-45-46 data\ac_100_1_CartPole-v0_27-09-2018_19-39-10


# Question 2.
# To run the analysis, use the following line
python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10
python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10

# To plot the result, use the following line
python plot.py data\ac_10_10_InvertedPendulum-v2_27-09-2018_19-56-45
python plot.py data\ac_10_10_HalfCheetah-v2_27-09-2018_20-09-29


# Bonus
# To run the analysis, use the following line
python train_ac_f18_bonus.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name 10_10_v_7_48_0005 -ntu 10 -ngsptu 10 -vl 7 -vs 48 -vlr 0.005
python train_ac_f18_bonus.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10_v_7_48_0005 -ntu 10 -ngsptu 10 -vl 7 -vs 48 -vlr 0.005

# To plot the result, use the following line
python plot.py ..\hw2\data\HalfCheetah-rtg-nn_HalfCheetah-v2_18-09-2018_10-43-21 data\ac_10_10_HalfCheetah-v2_28-09-2018_13-44-24 data\ac_10_10_v_7_48_0005_HalfCheetah-v2_28-09-2018_22-07-19
python plot.py ..\hw2\data\vpg_InvertedPendulum-v2_16-09-2018_17-32-43 data\ac_10_10_InvertedPendulum-v2_27-09-2018_19-56-45 data\ac_10_10_v_7_48_0005_InvertedPendulum-v2_29-09-2018_00-15-23
