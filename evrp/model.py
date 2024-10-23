from typing import Union

from rl4co.models import DeepACOPolicy
from rl4co.models.rl import REINFORCE
from rl4co.models.zoo import AttentionModelPolicy
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline

from evrp.policy import Policy

class Model(REINFORCE):
    """Attention Model based on REINFORCE: https://arxiv.org/abs/1803.08475.
    Check :class:`REINFORCE` and :class:`rl4co.models.RL4COLitModule` for more details such as additional parameters  including batch size.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """    

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: AttentionModelPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = Policy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)