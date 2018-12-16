Instruction:
I use PyCharm on Windows 10 without terminal.  Therefore, I parse everything in my code.
If you want to run it from terminal, make sure you comment out the third statement in the main function.  However, due to technical difficulty, I never tested it in the terminal.
Besides, additional keys should be provided ("--expert_policy_file", "-envname").  Some keys are not used in certain sections.

Demo:
python run_expert.py --expert_policy_file experts/Humanoid-v2.pkl --envname Humanoid-v2 --num_rollouts 20 --max_timesteps 1000 --render --demo

Section 2.2
python run_expert.py --expert_policy_file experts/Humanoid-v2.pkl --envname Humanoid-v2 --num_rollouts 20 --max_timesteps 1000 --epochs 500 --section22

Section 2.3
python run_expert.py --expert_policy_file experts/Humanoid-v2.pkl --envname Humanoid-v2 --num_rollouts 20 --max_timesteps 1000 --epochs 500 --section23

Section 3.2
python run_expert.py --expert_policy_file experts/Humanoid-v2.pkl --envname Humanoid-v2 --num_rollouts 20 --max_timesteps 1000 --epochs 500 --section32

Section 4.1
python run_expert.py --expert_policy_file experts/Humanoid-v2.pkl --envname Humanoid-v2 --num_rollouts 20 --max_timesteps 1000 --epochs 300 --section41