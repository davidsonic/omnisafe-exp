# source /workspace/miniconda3/bin/activate && conda activate omnisafe && cd /workspace/code/omnisafe/examples
# python train_policy.py --algo PPOLag --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo TRPOLag --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo FOCOPS --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NaturalPG --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo PolicyGradient --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo CUP --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NPGPD --env-id SafetyPointGoal1-v0 --total-steps 10000000 --device cpu --torch-threads 1 


# python train_policy.py --algo CUP --env-id SafetyHopperVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo TRPOLag --env-id SafetyHopperVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo PPOLag --env-id SafetyHopperVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo FOCOPS --env-id SafetyHopperVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NaturalPG --env-id SafetyHopperVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NPGPD --env-id SafetyHopperVelocity-v1  --total-steps 10000000 --device cpu --torch-threads 1 


# python train_policy.py --algo CUP --env-id SafetyHalfCheetahVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo TRPOLag --env-id SafetyHalfCheetahVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo PPOLag --env-id SafetyHalfCheetahVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo FOCOPS --env-id SafetyHalfCheetahVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NaturalPG --env-id SafetyHalfCheetahVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NPGPD --env-id SafetyHalfCheetahVelocity-v1  --total-steps 10000000 --device cuda:0  --torch-threads 1 


# python train_policy.py --algo CUP --env-id SafetyHumanoidVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo TRPOLag --env-id SafetyHumanoidVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo PPOLag --env-id SafetyHumanoidVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo FOCOPS --env-id SafetyHumanoidVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NaturalPG --env-id SafetyHumanoidVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NPGPD --env-id SafetyHumanoidVelocity-v1 --total-steps 10000000 --device cpu  --torch-threads 1 


# python train_policy.py --algo CUP --env-id SafetySwimmerVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo TRPOLag --env-id SafetySwimmerVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo PPOLag --env-id SafetySwimmerVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo FOCOPS --env-id SafetySwimmerVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NaturalPG --env-id SafetySwimmerVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NPGPD --env-id SafetySwimmerVelocity-v1  --total-steps 10000000 --device cpu  --torch-threads 1 


# python train_policy.py --algo CUP --env-id SafetyWalker2dVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo TRPOLag --env-id SafetyWalker2dVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo PPOLag --env-id SafetyWalker2dVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo FOCOPS --env-id SafetyWalker2dVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NaturalPG --env-id SafetyWalker2dVelocity-v1 --parallel 1 --total-steps 10000000 --device cpu --vector-env-nums 1 --torch-threads 1
# python train_policy.py --algo NPGPD --env-id SafetyWalker2dVelocity-v1  --total-steps 10000000 --device cpu  --torch-threads 1 