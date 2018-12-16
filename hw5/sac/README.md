# To run the experiment, use the following commands:

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf -e 3
python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam -e 3 --reparam

python train_mujoco.py --env_name Ant-v2 --exp_name reparam -e 3 --reparam
python train_mujoco.py --env_name Ant-v2 --exp_name reparam_2qf -e 3 --two_qf --reparam


# To plot the results, run the following commands:

python plot.py data/sac_HalfCheetah-v2_reinf_03-11-2018_20-12-35 data/sac_HalfCheetah-v2_reparam_04-11-2018_20-28-40 --value MaxEpReturn

python plot.py data/sac_Ant-v2_reparam_04-11-2018_23-31-59 data/sac_Ant-v2_reparam_2qf_05-11-2018_03-05-52 --value MaxEpReturn


