from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from jump_lab.variants import DEFAULT_VARIANT, Variant


@dataclass
class SymmetryCfg:
  """RSL-RL symmetry extension.

  The mirror loss pulls the mean action for a left-right mirrored observation
  towards the mirror of the mean action. Data augmentation is off: RSL-RL reuses
  the original sample's old log-probability for the mirrored sample, so until the
  policy is already symmetric the mirrored half has probability ratios near zero.
  In pilots it left the surrogate loss positive and training stalled.
  """

  data_augmentation_func: str = "jump_lab.training.mdp.symmetry:mirror"
  use_data_augmentation: bool = False
  use_mirror_loss: bool = True
  mirror_loss_coeff: float = 1.0


@dataclass
class JumpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  symmetry_cfg: SymmetryCfg | None = field(default_factory=SymmetryCfg)


def jump_ppo_runner_cfg(
  variant: Variant = DEFAULT_VARIANT,
) -> RslRlOnPolicyRunnerCfg:
  cfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=JumpPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=f"{variant}",
    wandb_project="jump-lab",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
  if variant == "jump_no_symmetry":
    cfg.algorithm.symmetry_cfg = None
  return cfg
