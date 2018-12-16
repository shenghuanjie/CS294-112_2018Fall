# Problem 4:

# For small batch size simulation:

python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na

# For small batch size plot:

python plot.py data/sb_no_rtg_dna_CartPole-v0_16-09-2018_16-29-21 data/sb_rtg_dna_CartPole-v0_16-09-2018_16-39-41 data/sb_rtg_na_CartPole-v0_16-09-2018_16-44-01 --value AverageReturn

# For large batch size simulation:

python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na

# For large batch size plot:

python plot.py data/lb_no_rtg_dna_CartPole-v0_16-09-2018_18-52-20 data/lb_rtg_dna_CartPole-v0_16-09-2018_18-54-25 data/lb_rtg_na_CartPole-v0_16-09-2018_18-56-55 --value AverageReturn


# Problem 5: 
# 1. comment out main() and uncomment prob5()
# 2. run the following command:

python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -rtg

# 3. record the best batch size and learning rate:
# max_score:1000.0, batch_size:10000, learning rate:0.01

# To answer the question, use the following commands:

python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -rtg -lr 0.01 -b 10000
python plot.py data/vpg_InvertedPendulum-v2_16-09-2018_17-32-43 --value AverageReturn

# Problem 7.
python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005

python plot.py data\ll_b40000_r0.005_LunarLanderContinuous-v2_16-09-2018_17-40-36

# Problem 8:
# 0. In get_log_prob(), I switched to MultivariateNormalDiag at this point
# 1. comment out main() and uncomment prob8()
# 2. run the following command:

python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -rtg --nn_baseline

# 3. record the best batch size and learning rate:
# max_score:10., batch_size:50000, learning rate:0.01

python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --exp_name HalfCheetah
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --exp_name HalfCheetah-rtg
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 --nn_baseline --exp_name HalfCheetah-nn
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name HalfCheetah-rtg-nn

python plot.py data/HalfCheetah-nn_HalfCheetah-v2_18-09-2018_00-50-50 data\HalfCheetah-rtg-nn_HalfCheetah-v2_18-09-2018_10-43-21 data\HalfCheetah-rtg_HalfCheetah-v2_18-09-2018_00-19-41 data\HalfCheetah_HalfCheetah-v2_17-09-2018_23-52-16


# Bonus:

python train_pg_f18.py CartPole-v0 -n 100 -b 100 -e 3 -dna -gs 10 --exp_name sb_no_rtg_dna_gs10_b100_n100
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna -gs 1 --exp_name sb_no_rtg_dna_gs1_b1000_n100
python train_pg_f18.py CartPole-v0 -n 10 -b 1000 -e 3 -dna -gs 10 --exp_name sb_no_rtg_dna_gs10_b1000_n10

python plot.py data\sb_no_rtg_dna_gs10_b100_n100_CartPole-v0_18-09-2018_09-36-13 data\sb_no_rtg_dna_gs1_b1000_n100_CartPole-v0_18-09-2018_09-43-13 data\sb_no_rtg_dna_gs10_b1000_n10_CartPole-v0_18-09-2018_09-37-17
