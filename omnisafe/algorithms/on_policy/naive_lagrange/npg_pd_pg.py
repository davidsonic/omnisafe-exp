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
"""Implementation of the NPGPD algorithm."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)



@registry.register
class NPGPD(TRPO):
    """The Lagrange version of the TRPO algorithm.

    A simple combination of the Lagrange method and the Trust Region Policy Optimization algorithm.
    """

    def _init(self) -> None:
        """Initialize the TRPOLag specific model.

        The TRPOLag algorithm uses a Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the TRPOLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)


    def _sgd_step(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor: 
        """Return grad_log respect to theta
        Take a SGD step but do not update the actor
        """
        self._actor_critic.actor.zero_grad()
        distribution = self._actor_critic.actor(obs)
        logp_  = self._actor_critic.actor.log_prob(act)
        loss = -logp_.mean()
        self._logger.store(
            {
                'Train/GradientStep': loss.mean().item(),
            }
        )
        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)
        grads = get_flat_gradients_from(self._actor_critic.actor)
        return grads
        


    def _proj_ball(self, x: torch.Tensor): 
        """Return the projection of x onto unit ball
        """
        assert x.ndim == 1, 'x must be a flattened tensor'
        return x if x.norm().item() <=1 else x / x.norm()
        


    def _update_actor(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network.

            - Compute the gradient of the policy.
            - Compute the step direction using NPG-PD estimate
            - Search for a step size that satisfies the constraint.
            - Update the policy network.

            Args:
                obs (torch.Tensor): The observation tensor.
                act (torch.Tensor): The action tensor.
                logp (torch.Tensor): The log probability of the action.
                adv_r (torch.Tensor): The reward advantage tensor.
                adv_c (torch.Tensor): The cost advantage tensor.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)
        grads = -get_flat_gradients_from(self._actor_critic.actor)

        theta_new = theta_old + grads
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        # self._logger.store(
        #     {
        #         'Loss': loss.item(),
        #     }
        # )

        

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old}(a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        
        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        self._update_actor(obs, act, logp, adv_r, adv_c)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, target_value_r, target_value_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                target_value_r,
                target_value_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)

        self._logger.store(
            {
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
                'Value/Adv': adv_r.mean().item(),
            },
        )

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})



    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        TRPOLag uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
