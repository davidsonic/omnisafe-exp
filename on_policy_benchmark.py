# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of training a policy from exp-x config with OmniSafe."""

import warnings

import torch

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='On-Policy-Benchmarks')

    # set up the algorithms.
    base_policy = ['PolicyGradient', 'NaturalPG', 'PPO']
    naive_lagrange_policy = []
    first_order_policy = ['CUP', 'FOCOPS']
    second_order_policy = ['CPO', 'PCPO']
    saute_policy = []
    simmer_policy = []
    pid_policy = []
    early_mdp_policy = []

    eg.add(
        'algo',
        base_policy +
        naive_lagrange_policy +
        first_order_policy +
        second_order_policy +
        saute_policy +
        simmer_policy +
        pid_policy +
        early_mdp_policy
    )

    # you can use wandb to monitor the experiment.
    eg.add('logger_cfgs:use_wandb', [False])
    # you can use tensorboard to monitor the experiment.
    eg.add('logger_cfgs:use_tensorboard', [True])

    # the default configs here are as follows:
    # eg.add('algo_cfgs:steps_per_epoch', [20000])
    # eg.add('train_cfgs:total_steps', [20000 * 500])
    # which can reproduce results of 1e7 steps.

    # if you want to reproduce results of 1e6 steps, using
    # eg.add('algo_cfgs:steps_per_epoch', [2048])
    # eg.add('train_cfgs:total_steps', [2048 * 500])

    # set the device.
    avaliable_gpus = list(range(torch.cuda.device_count()))
    # if you want to use GPU, please set gpu_id like follows:
    # gpu_id = [0, 1, 2, 3]
    # if you want to use CPU, please set gpu_id = None
    # we recommends using CPU to obtain results as consistent
    # as possible with our publicly available results,
    # since the performance of all on-policy algorithms
    # in OmniSafe is tested on CPU.
    gpu_id = None

    if gpu_id and not set(gpu_id).issubset(avaliable_gpus):
        warnings.warn('The GPU ID is not available, use CPU instead.', stacklevel=1)
        gpu_id = None

    # set up the environment.
    eg.add('env_id', [
        'SafetyHopperVelocity-v1',
        ])
    eg.add('seed', [0, 5, 10, 15, 20])

    # total experiment num must can be divided by num_pool.
    # meanwhile, users should decide this value according to their machine.
    eg.run(train, num_pool=5, gpu_id=gpu_id)