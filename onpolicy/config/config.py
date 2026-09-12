import argparse
import json

from onpolicy.utils.training_stage import normalize_training_stage


def get_config():
    """ 
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default False; if set, use running mean and std to normalize value targets.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters:
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='gnn_mappo', choices=["rmappo", "mappo", "happo", "gnn_mappo", "hatrpo", "mat", "mat_dec"])

    parser.add_argument("--experiment_name", type=str, default="train-medium-ppo3", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--device", type=str, default='cuda:0', help="by default None, will use cuda if available; or else will use cpu. If set, use the device specified.")
    parser.add_argument(
        "--cuda_memory_fraction",
        type=float,
        default=0.0,
        help=(
            "optional per-process CUDA allocator limit as a fraction of the "
            "visible device; 0 leaves the PyTorch default unlimited"
        ),
    )
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=100,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=20,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument(
        "--shared_eval_socket",
        type=str,
        default="",
        help=(
            "optional AF_UNIX socket for a per-GPU shared HKBZ evaluator; "
            "an empty value keeps in-process validation"
        ),
    )
    parser.add_argument(
        "--shared_eval_cpu_set",
        type=str,
        default="",
        help=(
            "logical CPU set temporarily assigned to the shared evaluator "
            "while this trainer is blocked on validation"
        ),
    )
    parser.add_argument(
        "--shared_eval_timeout_seconds",
        type=float,
        default=7200.0,
        help="maximum queue plus execution time for one shared validation",
    )
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--num_episodes", type=int, default=200,
                        help='Number of episodes to train (default: 40)')
    parser.add_argument("--max_train_cases", type=int, default=0,
                        help="limit HKBZ training cases for smoke tests; 0 uses the full split")
    parser.add_argument("--max_eval_cases", type=int, default=0,
                        help="limit HKBZ validation cases for smoke tests; 0 uses the full split")
    parser.add_argument(
        "--eval_dataset_dir",
        type=str,
        default="",
        help=(
            "optional HKBZ evaluation-dataset override; an empty value uses "
            "eval_dataset_dir from the environment YAML"
        ),
    )
    parser.add_argument(
        "--eval_case_offset",
        type=int,
        default=0,
        help=(
            "number of cases to skip in the fixed evaluation partition order; "
            "combine with --max_eval_cases to isolate tune/select subsets"
        ),
    )
    parser.add_argument(
        "--eval_partition_seed",
        type=int,
        default=20260803,
        help=(
            "seed for the fixed evaluation partition order; deliberately "
            "independent of the model-training seed"
        ),
    )
    parser.add_argument(
        "--eval_partition_stratify_by",
        choices=["", "distribution", "profile"],
        default="",
        help=(
            "optional metadata key used to stratify fixed tune/select "
            "partitions; distribution is recommended for Stage-1 v2"
        ),
    )
    parser.add_argument(
        "--train_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "distribution_balanced", "profile_balanced"],
        help=(
            "HKBZ case sampler; balanced modes oversample either metadata "
            "distributions or profiles to the requested mixture without "
            "discarding unique training cases when no explicit sample-size "
            "cap is set"
        ),
    )
    parser.add_argument(
        "--train_sampling_weights",
        type=str,
        default="iid=0.50,ood_stress=0.45,ood_scale=0.05",
        help=(
            "comma-separated distribution weights used by "
            "--train_sampling_mode distribution_balanced"
        ),
    )
    parser.add_argument(
        "--train_sampling_size",
        type=int,
        default=0,
        help=(
            "number of sampled HKBZ cases per coverage cycle; 0 chooses the "
            "smallest balanced expansion that retains every unique case"
        ),
    )
    parser.add_argument(
        "--train_sampling_pool_size",
        type=int,
        default=0,
        help=(
            "optional balanced case pool retained by every rollout worker; "
            "when this exceeds --train_sampling_size, only the latter number "
            "of cases is consumed per epoch and successive epochs rotate "
            "through the larger deterministic pool"
        ),
    )
    parser.add_argument(
        "--device_policy_head_mode",
        type=str,
        default="shared",
        choices=["shared", "type_adapter", "per_type"],
        help=(
            "ordinary-mobile-device actor backend: the checkpoint-compatible "
            "shared head, a shared head with a categorical type adapter, or "
            "one independently recurrent actor head per ordinary device type"
        ),
    )
    parser.add_argument(
        "--ordinary_device_type_count",
        type=int,
        default=10,
        help="number of non-R014 mobile resource types routed by Stage2 heads",
    )
    parser.add_argument(
        "--device_timing_head",
        action="store_true",
        default=False,
        help=(
            "factor device dispatch-vs-defer timing through an explicit "
            "scalar gate while retaining the request-ranking pointer"
        ),
    )
    parser.add_argument(
        "--device_global_matching",
        action="store_true",
        default=False,
        help=(
            "decode deterministic device actions with a global one-to-one "
            "request matching instead of fixed device-order greedy claims"
        ),
    )
    parser.add_argument(
        "--request_ready_prediction",
        action="store_true",
        default=False,
        help=(
            "enable the request-level ready-time head; Stage2 supervised "
            "training requires it and Stage3 must match the checkpoint"
        ),
    )
    parser.add_argument(
        "--request_ready_policy_injection",
        type=str,
        default="learned",
        choices=["none", "dag", "learned"],
        help=(
            "how request-ready information enters the resource policy: none "
            "trains the predictor without policy gradients, dag injects only "
            "the deterministic dependency lower bound, and learned also "
            "injects the predicted intrinsic ready time"
        ),
    )
    parser.add_argument(
        "--device_resource_adapter",
        action="store_true",
        default=False,
        help=(
            "add a zero-initialized resource-only residual adapter after the "
            "shared encoder; the protected aircraft path remains unchanged"
        ),
    )
    parser.add_argument(
        "--request_ready_time_scale",
        type=float,
        default=3600.0,
        help="seconds used to log-normalize request ready-time regression",
    )
    parser.add_argument(
        "--request_ready_hard_blocking",
        action="store_true",
        default=False,
        help=(
            "route already-blocking requests to their exact zero lead time "
            "instead of asking the non-negative residual head to approximate zero"
        ),
    )
    parser.add_argument(
        "--request_ready_context_features",
        action="store_true",
        default=False,
        help=(
            "give the predictor explicit request-kind, dependency-depth and "
            "predecessor timing context in addition to frozen GNN embeddings"
        ),
    )
    parser.add_argument(
        "--request_ready_head_mode",
        type=str,
        default="shared",
        choices=["shared", "horizon_split"],
        help=(
            "use one checkpoint-compatible ready-time output head or independent "
            "H1/H2/departure/other output projections over a shared predictor trunk"
        ),
    )
    parser.add_argument(
        "--request_ready_quantile_head",
        action="store_true",
        default=False,
        help=(
            "predict ordered q20/q50/q80 ready-time residuals; q50 remains the "
            "point prediction exposed to existing callers"
        ),
    )
    parser.add_argument("--user_name", type=str, default='marl', help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='IA', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=150, help="Max length for any episode")
    parser.add_argument("--rollout_until_done", dest="rollout_until_done", action='store_true',
                        default=True, help="for HKBZ, collect each rollout until the schedule naturally terminates")
    parser.add_argument("--no_rollout_until_done", dest="rollout_until_done", action='store_false',
                        help="for HKBZ, fall back to fixed-length rollout collection")
    parser.add_argument("--rollout_max_steps", type=int, default=2000,
                        help="safety cap for natural HKBZ rollout collection")
    parser.add_argument("--allow_incomplete_rollout", action='store_true', default=False,
                        help="allow PPO update when a natural HKBZ rollout hits rollout_max_steps before done")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", dest='use_valuenorm', action='store_true',
                        help="use running mean and std to normalize value targets")
    parser.add_argument("--no_valuenorm", dest='use_valuenorm', action='store_false',
                        help="disable value-target normalization")
    parser.set_defaults(use_valuenorm=False)
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=30,
                        help="Time length of chunks used to train a recurrent_policy")
    parser.add_argument("--max_graphs_per_forward", type=int, default=0,
                        help="fail fast when mini_batch_size * data_chunk_length exceeds this limit; 0 disables the guard")
    parser.add_argument(
        "--shared_encoder_activation_checkpoint",
        action="store_true",
        default=False,
        help=(
            "recompute the trainable shared graph encoder during backward to "
            "bound Stage3 PPO activation memory"
        ),
    )
    parser.add_argument("--actor_warmup_shards", type=int, default=0,
                        help="critic-only rollout shards before the first actor PPO update")
    parser.add_argument("--clear_cuda_cache_after_update", action="store_true", default=False,
                        help="release unused CUDA allocator blocks after each rollout update")
    parser.add_argument(
        "--shared_gpu_phase_lock",
        type=str,
        default="",
        help=(
            "optional absolute advisory-lock path that serializes memory-heavy "
            "PPO updates across trainers sharing one GPU"
        ),
    )

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=0.0001,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=0.0001,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--shared_actor_lr_scale", type=float, default=1.0,
                        help="learning-rate multiplier for the shared GNN encoder")
    parser.add_argument(
        "--shared_actor_lr_scale_schedule",
        type=str,
        default="",
        help=(
            "optional comma-separated per-epoch shared-encoder LR scales; "
            "the final value is held for later epochs"
        ),
    )
    parser.add_argument(
        "--shared_actor_lr_max_multiplier",
        type=float,
        default=0.0,
        help=(
            "optional cap applied only to the shared-encoder portion of the "
            "adaptive Actor-LR multiplier; 0 inherits the global cap"
        ),
    )
    parser.add_argument("--plane_actor_lr_scale", type=float, default=1.0,
                        help="learning-rate multiplier for the pretrained plane actor backend")
    parser.add_argument("--device_actor_lr_scale", type=float, default=1.0,
                        help="learning-rate multiplier for the ordinary device actor backend")
    parser.add_argument("--transporter_actor_lr_scale", type=float, default=1.0,
                        help="learning-rate multiplier for the transporter actor backend")

    # trpo parameters
    parser.add_argument("--kl_threshold", type=float, 
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--ls_step", type=int, 
                        default=10, help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float, 
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=3,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--target_kl", type=float, default=0.0,
                        help="stop actor PPO updates when approximate KL exceeds this value; 0 disables")
    parser.add_argument(
        "--actor_kl_backtrack",
        action="store_true",
        default=False,
        help=(
            "retry an Actor optimizer group at smaller step scales when the "
            "post-update old-policy KL gate rejects it; failed groups are "
            "rolled back without aborting the remaining PPO shard"
        ),
    )
    parser.add_argument(
        "--actor_kl_backtrack_scales",
        type=str,
        default="0.5,0.25,0.125",
        help=(
            "comma-separated retry scales for --actor_kl_backtrack; values "
            "must be strictly decreasing and in (0, 1)"
        ),
    )
    parser.add_argument(
        "--adaptive_actor_kl",
        action="store_true",
        default=False,
        help=(
            "adapt actor learning rates after each PPO update so measured "
            "post-update joint KL remains in the configured band"
        ),
    )
    parser.add_argument("--adaptive_actor_kl_low", type=float, default=1e-4,
                        help="lower edge of the desired post-update joint-KL band")
    parser.add_argument("--adaptive_actor_kl_high", type=float, default=5e-4,
                        help="upper edge of the desired post-update joint-KL band")
    parser.add_argument("--adaptive_actor_lr_min_scale", type=float, default=0.25,
                        help="minimum actor-LR scale relative to its initial value")
    parser.add_argument("--adaptive_actor_lr_max_scale", type=float, default=8.0,
                        help="maximum actor-LR scale relative to its initial value")
    parser.add_argument("--adaptive_actor_lr_up", type=float, default=1.5,
                        help="actor-LR multiplier when measured KL is below target")
    parser.add_argument("--adaptive_actor_lr_down", type=float, default=0.5,
                        help="actor-LR multiplier when measured KL is above target")
    parser.add_argument(
        "--adaptive_actor_min_step_completion",
        type=float,
        default=0.9,
        help=(
            "minimum actual/planned Actor optimizer-step ratio required before "
            "the KL controller may increase learning rates"
        ),
    )
    parser.add_argument(
        "--bc_reference_kl_coef",
        type=float,
        default=0.0,
        help=(
            "coefficient for the sampled KL penalty from the current policy "
            "to the frozen post-BC reference policy; 0 disables"
        ),
    )
    parser.add_argument(
        "--bc_reference_kl_coef_schedule",
        type=str,
        default="",
        help=(
            "optional comma-separated per-PPO-epoch BC-reference KL "
            "coefficients; the final value is held for later epochs and an "
            "empty schedule preserves --bc_reference_kl_coef"
        ),
    )
    parser.add_argument(
        "--bc_reference_target_kl",
        type=float,
        default=0.0,
        help=(
            "monitoring target for sampled KL to the frozen post-BC reference; "
            "0 disables target monitoring"
        ),
    )
    parser.add_argument(
        "--bc_reference_hard_gate",
        action="store_true",
        default=False,
        help=(
            "stop Actor updates when BC-reference KL exceeds "
            "--bc_reference_target_kl; disabled by default so the KL term is "
            "a soft regularizer"
        ),
    )
    parser.add_argument(
        "--adaptive_bc_reference_kl",
        action="store_true",
        default=False,
        help=(
            "adapt the soft BC-reference KL coefficient after each PPO shard "
            "instead of using the BC-reference target as a hard stop"
        ),
    )
    parser.add_argument("--adaptive_bc_reference_target_kl", type=float, default=0.03,
                        help="target sampled KL used by the adaptive soft BC controller")
    parser.add_argument("--adaptive_bc_reference_coef_min", type=float, default=0.02,
                        help="minimum adaptive soft BC-reference coefficient")
    parser.add_argument("--adaptive_bc_reference_coef_max", type=float, default=1.0,
                        help="maximum adaptive soft BC-reference coefficient")
    parser.add_argument("--adaptive_bc_reference_coef_up", type=float, default=1.5,
                        help="coefficient multiplier when BC-reference KL is above target")
    parser.add_argument("--adaptive_bc_reference_coef_down", type=float, default=0.8,
                        help="coefficient multiplier when BC-reference KL is below half target")
    parser.add_argument("--mini_batch_size", type=int, default=10,
                        help='size of training batch for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1.0, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=1.00,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=False, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--no_normalize_advantages", dest='normalize_advantages',
                        action='store_false',
                        help="disable per-role advantage normalization")
    parser.add_argument("--no_role_balanced_loss", dest='role_balanced_loss',
                        action='store_false',
                        help="disable role-balanced PPO and value losses")
    parser.add_argument("--no_case_balanced_loss", dest='case_balanced_loss',
                        action='store_false',
                        help="disable equal-case PPO/value weighting")
    parser.set_defaults(
        normalize_advantages=True,
        role_balanced_loss=True,
        case_balanced_loss=True,
    )
    parser.add_argument("--plane_loss_coef", type=float, default=1.0,
                        help="relative PPO/value weight for plane decisions")
    parser.add_argument("--device_loss_coef", type=float, default=0.5,
                        help="relative PPO/value weight for ordinary device decisions")
    parser.add_argument("--transporter_loss_coef", type=float, default=0.5,
                        help="relative PPO/value weight for transporter decisions")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    parser.add_argument("--gnn_pretrain", action='store_false', 
                        default=True, help="by default True, use pretrain gnn model. If set, do not use pretrain gnn model.")
    parser.add_argument("--gnn_freeze_epochs", type=int, default=0,
                        help="number of initial PPO epochs that freeze the shared GNN")
    parser.add_argument("--plane_freeze_epochs", type=int, default=0,
                        help="number of initial joint-training epochs that freeze the pretrained plane actor")
    parser.add_argument(
        '--stage2_allow_shared_unfreeze',
        action='store_true',
        default=False,
        help=(
            'experimental Stage2 arm: keep the plane decoder frozen while '
            'allowing the unsplit shared encoder to adapt after '
            '--gnn_freeze_epochs'
        ),
    )
    parser.add_argument(
        "--plane_order_freeze_epochs",
        type=int,
        default=0,
        help="number of initial Stage-1 PPO epochs that freeze only the learned plane-order head",
    )
    parser.add_argument(
        "--training_stage",
        type=normalize_training_stage,
        default="auto",
        choices=["auto", "plane_pretrain", "resource_joint", "joint_finetune"],
        help="explicit HKBZ training stage; auto preserves legacy single-process behavior",
    )
    parser.add_argument(
        '--stage2_frozen_manifest',
        type=str,
        default='',
        help='explicit portable frozen B0 manifest for fresh Stage3 initialization; not a resume checkpoint',
    )
    parser.add_argument(
        "--resume_stage1",
        action="store_true",
        default=False,
        help="explicitly allow plane_pretrain to restore a same-stage recovery checkpoint",
    )
    parser.add_argument(
        "--resume_stage2",
        action="store_true",
        default=False,
        help=(
            "explicitly resume resource_joint from a compatible recovery "
            "checkpoint (supervised Stage2 uses the DeviceBC cursor options)"
        ),
    )
    parser.add_argument(
        "--device_bc_resume_epoch",
        type=int,
        default=0,
        help=(
            "zero-based DeviceBC epoch containing the supervised recovery "
            "cursor; valid only with --resume_stage2"
        ),
    )
    parser.add_argument(
        "--device_bc_resume_completed_rollouts",
        type=int,
        default=0,
        help=(
            "number of fully completed DeviceBC rollouts in the recovery "
            "epoch; valid only with --resume_stage2"
        ),
    )
    parser.add_argument(
        "--reset_optimizers_on_resume",
        action="store_true",
        default=False,
        help="restore model weights but start with fresh actor and critic optimizers",
    )
    parser.add_argument(
        "--reset_value_normalizer_on_resume",
        action="store_true",
        default=False,
        help=(
            "restore model weights but discard checkpoint ValueNorm statistics; "
            "required when the PPO reward/return semantics change"
        ),
    )
    parser.add_argument(
        "--reset_value_normalizer_before_ppo",
        action="store_true",
        default=False,
        help=(
            "discard inherited ValueNorm statistics after Stage hand-off/BC "
            "restore and calibrate the Critic on the current return semantics"
        ),
    )
    parser.add_argument(
        "--strict_checkpoint_contract",
        action="store_true",
        default=False,
        help=(
            "require versioned observation and environment semantics metadata "
            "before restoring a Stage-1 checkpoint"
        ),
    )
    parser.add_argument(
        "--strict_stage1_reward_contract",
        action="store_true",
        default=False,
        help=(
            "require gamma=1 and an exactly equivalent scaled -Cmax objective "
            "for Stage-1 reward-credit ablations"
        ),
    )
    parser.add_argument(
        "--selection_checkpoint_dir",
        type=str,
        default=None,
        help="optional prior global-Best checkpoint preserved across a resumed Stage1 run",
    )
    parser.add_argument("--use_anneal", action='store_false', 
                        default=True, help="by default True, use anneal to adjust hyperparameters. If set, do not use anneal.")
    parser.add_argument("--anneal_original", type=float, default=1.0, help="the original value of anneal, default 1.0")
    parser.add_argument("--anneal_final", type=float, default=0.1, help="the final value of anneal, default 0.1")
    parser.add_argument(
        "--tau_anneal_epochs",
        type=int,
        default=0,
        help=(
            "number of training epochs used to move tau from "
            "--anneal_original to --anneal_final, including both endpoints; "
            "later epochs hold the final tau. 0 preserves the legacy linear "
            "schedule across the complete run"
        ),
    )

    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1, help="time duration between contiunous twice log printing.")
    parser.add_argument("--status_heartbeat_seconds", type=float, default=60.0,
                        help="seconds between atomic run-status heartbeats; <=0 disables")
    parser.add_argument(
        "--torch_mp_sharing_strategy",
        choices=["file_descriptor", "file_system"],
        default="file_descriptor",
        help=(
            "PyTorch CPU tensor IPC strategy; file_descriptor avoids one "
            "torch_shm_manager process per environment worker"
        ),
    )
    parser.add_argument(
        "--ipc_timeout_seconds",
        type=float,
        default=300.0,
        help="hard timeout for one vector-environment IPC response batch",
    )
    parser.add_argument(
        "--safe_async_graph_clone_workers",
        type=int,
        default=0,
        help=(
            "number of parent-side threads used to clone graph observations "
            "while other ordered IPC replies arrive; 0 keeps the synchronous "
            "path"
        ),
    )
    parser.add_argument(
        "--safe_graph_batch_pipeline",
        action="store_true",
        default=False,
        help=(
            "prefetch deterministic PyG Batch construction and reuse each "
            "prepared batch across equivalent policy evaluations"
        ),
    )
    parser.add_argument(
        "--safe_dagger_teacher_overlap",
        action="store_true",
        default=False,
        help=(
            "overlap read-only IGA-teacher RPCs with deterministic student "
            "inference while preserving environment result order"
        ),
    )
    parser.add_argument("--recovery_checkpoint_interval_shards", type=int, default=1,
                        help="overwrite checkpoint_Recovery.pt every N completed shards; <=0 disables")

    # eval parameters
    parser.add_argument("--use_eval", dest='use_eval', action='store_true',
                        help="run deterministic C_max validation during training")
    parser.add_argument("--no_eval", dest='use_eval', action='store_false',
                        help="disable validation during training")
    parser.set_defaults(use_eval=False)
    parser.add_argument(
        "--skip_pre_ppo_eval",
        action="store_true",
        default=False,
        help=(
            "skip only the pre-PPO baseline evaluation; intended for short "
            "resource-joint infrastructure canaries, never formal selection"
        ),
    )
    parser.add_argument(
        "--skip_epoch_eval",
        action="store_true",
        default=False,
        help=(
            "skip PPO epoch validation; intended only for throughput/OOM "
            "canaries, never formal checkpoint selection"
        ),
    )
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="number of training epochs between C_max validations")
    parser.add_argument("--canary_eval_interval_shards", type=int, default=0,
                        help="run an in-epoch deterministic validation every N shards; 0 disables")
    parser.add_argument(
        "--canary_eval_max_per_epoch",
        type=int,
        default=0,
        help=(
            "maximum in-epoch validations per epoch; 0 keeps the legacy "
            "unlimited behavior"
        ),
    )
    parser.add_argument(
        "--canary_eval_max_cases",
        type=int,
        default=0,
        help=(
            "case count used only by shard canaries served through the shared "
            "validator; 0 uses the full configured validation partition"
        ),
    )
    parser.add_argument("--canary_max_regression", type=float, default=0.0,
                        help="maximum relative C_max regression versus current Best accepted by the shard canary")
    parser.add_argument(
        "--canary_stop_on_regression",
        action="store_true",
        default=False,
        help="stop training cleanly when an in-epoch canary exceeds the regression limit",
    )
    parser.add_argument("--eval_canary_rounds", type=int, default=1,
                        help="minimum validation rounds before an invalid checkpoint can abort full evaluation")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")
    parser.add_argument(
        "--evaluation_tau",
        type=float,
        default=0.3,
        help=(
            "fixed policy temperature for every deterministic validation, "
            "including the pre-PPO baseline"
        ),
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="iid",
        choices=["raw", "iid", "composite", "composite_tail"],
        help=(
            "checkpoint selection by the raw validation mean, IID mean, "
            "weighted IID/OOD validation, or a convex combination of that "
            "composite and the validation worst tail"
        ),
    )
    parser.add_argument("--selection_iid_weight", type=float, default=0.50)
    parser.add_argument("--selection_ood_stress_weight", type=float, default=0.45)
    parser.add_argument("--selection_ood_scale_weight", type=float, default=0.05)
    parser.add_argument(
        "--selection_tail_fraction",
        type=float,
        default=0.10,
        help="largest validation fraction used by composite_tail selection",
    )
    parser.add_argument(
        "--selection_tail_weight",
        type=float,
        default=0.25,
        help="weight of the worst-tail mean in composite_tail selection",
    )
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="stop after this many non-improving validations; 0 disables early stopping")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="by default None. set the path to pretrained model.")
    parser.add_argument("--gnn_checkpoint", type=str, default=None, help="the path to gnn checkpoint, default None")
    parser.add_argument("--device_bc_pretrain_epochs", type=int, default=0,
                        help="supervised Stage2 epochs for the ready-time and mobile-resource policy heads")
    parser.add_argument(
        "--device_bc_training_scope",
        type=str,
        default="policy_and_ready",
        choices=["policy_and_ready", "ready_only", "policy_frozen_ready"],
        help=(
            "parameters optimized by supervised Stage2: the historical "
            "resource-policy plus ready-time heads, or only the intrinsic "
            "request-ready predictor, or resource policy with a frozen predictor"
        ),
    )
    parser.add_argument('--request_ready_checkpoint', type=str, default='',
                        help='predictor-only source for policy_frozen_ready BC')
    parser.add_argument('--request_ready_checkpoint_sha256', type=str, default='',
                        help='required immutable digest of the frozen ready source')
    parser.add_argument('--stage2_policy_warmstart_checkpoint', type=str, default='',
                        help='fork selected Stage2 policy weights with fresh optimizer/RNG (not resume)')
    parser.add_argument('--stage2_policy_warmstart_sha256', type=str, default='',
                        help='immutable digest of the selected policy fork source')
    parser.add_argument('--stage2_policy_warmstart_evaluation', type=str, default='',
                        help='case-level source evaluation that must replay exactly before BC')
    parser.add_argument('--stage2_policy_warmstart_evaluation_sha256', type=str, default='',
                        help='immutable digest of the warm-start reference evaluation')
    parser.add_argument('--stage2_bc_deterministic', action='store_true', default=False,
                        help='require deterministic CUDA algorithms in BC training and shared evaluation')
    parser.add_argument('--device_bc_eval_each_epoch', action='store_true', default=False,
                        help='save complete BC epoch boundaries and select by validation Cmax')
    parser.add_argument('--train_sampling_seed', type=int, default=None,
                        help='optional data-order seed independent of model/training seed')
    parser.add_argument(
        "--device_bc_only",
        action="store_true",
        default=False,
        help=(
            "legacy compatibility switch; canonical supervised Stage2 always "
            "stops after its post-supervision validation"
        ),
    )
    parser.add_argument(
        '--resource_bc_checkpoint',
        type=str,
        default='',
        help=(
            'legacy resource-BC boundary used only by noncanonical research '
            'launchers; canonical Stage2 rejects this option'
        ),
    )
    parser.add_argument("--device_bc_lr", type=float, default=0.0,
                        help="learning rate for device behavior cloning warmup; <=0 uses --lr")
    parser.add_argument(
        "--device_bc_teacher",
        type=str,
        default="heuristic",
        choices=["heuristic", "iga", "joint_iga"],
        help=(
            "live-mask resource teacher used during BC/DAgger; joint_iga "
            "decodes the strictly rebound Stage3 full-joint chromosome"
        ),
    )
    parser.add_argument(
        "--resource_iga_teacher_dir",
        type=str,
        default="",
        help="directory containing verified per-case Stage2 IGA teacher JSON files",
    )
    parser.add_argument(
        "--resource_iga_teacher_index",
        type=str,
        default="",
        help="strict SHA256/fingerprint sidecar index for Stage2 IGA teachers",
    )
    parser.add_argument(
        "--joint_iga_teacher_dir",
        type=str,
        default="",
        help="directory containing verified Stage3 full-joint IGA teacher JSON files",
    )
    parser.add_argument(
        "--joint_iga_teacher_index",
        type=str,
        default="",
        help=(
            "strict per-case SHA256 and target-planning-contract index for "
            "Stage3 joint IGA teachers"
        ),
    )
    parser.add_argument(
        "--device_bc_role_balanced",
        action="store_true",
        default=False,
        help="normalize ordinary-device and R014 BC losses independently",
    )
    parser.add_argument(
        "--device_bc_timing_balanced",
        action="store_true",
        default=False,
        help=(
            "normalize DeviceBC over blocking-dispatch, lookahead-dispatch, "
            "and intentional-defer timing strata (crossed with role strata "
            "when --device_bc_role_balanced is enabled)"
        ),
    )
    parser.add_argument(
        "--device_bc_legacy_noop_timing",
        action="store_true",
        default=False,
        help=(
            "comparison-only compatibility mode that classifies every "
            "teacher no-op as a defer timing label, reproducing the Stage2 "
            "supervision semantics used before no-op causes were separated"
        ),
    )
    parser.add_argument(
        "--device_bc_min_teacher_score_margin",
        type=float,
        default=-1.0,
        help=(
            "drop ambiguous non-noop IGA labels whose selected genome-score "
            "distance to the nearest legal alternative is below this value; "
            "a negative value disables filtering"
        ),
    )
    parser.add_argument(
        "--device_bc_ranking_loss_coef",
        type=float,
        default=0.0,
        help=(
            "coefficient for listwise distillation of all legal IGA "
            "candidate scores; 0 preserves categorical DeviceBC"
        ),
    )
    parser.add_argument(
        "--device_bc_categorical_loss_coef",
        type=float,
        default=1.0,
        help=(
            "coefficient for serialized per-device categorical BC; set to 0 "
            "when the permutation-invariant final-matching target is used"
        ),
    )
    parser.add_argument(
        "--device_bc_min_ranking_labels_per_epoch",
        type=int,
        default=64,
        help=(
            "minimum legal-candidate listwise labels per supervised Stage2 "
            "epoch; canonical Stage2 fails closed below this value"
        ),
    )
    parser.add_argument(
        "--device_bc_ranking_temperature",
        type=float,
        default=0.10,
        help="softmax temperature for normalized IGA candidate-score targets",
    )
    parser.add_argument(
        "--device_bc_assignment_loss_coef",
        type=float,
        default=0.0,
        help=(
            "coefficient for permutation-invariant supervision of the final "
            "teacher matching; this replaces chromosome-score distillation"
        ),
    )
    parser.add_argument(
        "--device_bc_assignment_margin_loss_coef",
        type=float,
        default=0.0,
        help=(
            "coefficient for a set-level margin between teacher-selected and "
            "non-selected requests"
        ),
    )
    parser.add_argument(
        "--device_bc_assignment_margin",
        type=float,
        default=0.20,
        help="log-probability margin used by structured assignment supervision",
    )
    parser.add_argument(
        "--device_bc_min_assignment_labels_per_epoch",
        type=int,
        default=0,
        help=(
            "minimum permutation-invariant device-type assignment groups per "
            "Stage2 epoch; zero disables this gate for categorical controls"
        ),
    )
    parser.add_argument('--device_bc_matching_audit', action='store_true', default=False,
                        help='Audit full teacher matching and report actual deployment-decoder agreement')
    parser.add_argument('--device_bc_teacher_deployment_projection', action='store_true', default=False,
                        help='Explicit BC-only IGA-derived maximum-Blocking teacher; requires audit and no factual Ready labels')
    parser.add_argument('--device_bc_full_edge_loss_coef', type=float, default=0.0,
                        help='Additional identity-preserving structured matching margin; 0 keeps legacy loss')
    parser.add_argument('--device_bc_empty_wait_loss_coef', type=float, default=0.0,
                        help='Typed temporal-defer NLL for resource groups with no teacher dispatch')
    parser.add_argument('--device_bc_min_wait_groups_per_epoch', type=int, default=0,
                        help='Fail closed if an enabled wait arm has insufficient meaningful empty groups')
    parser.add_argument(
        "--device_bc_timing_loss_coef",
        type=float,
        default=0.0,
        help=(
            "coefficient for an explicit dispatch-vs-defer binary BC loss "
            "computed from the normalized request distribution"
        ),
    )
    parser.add_argument(
        "--request_ready_loss_coef",
        type=float,
        default=1.0,
        help=(
            "coefficient for request-level ready-time regression in Stage2 "
            "supervised learning"
        ),
    )
    parser.add_argument(
        "--request_ready_seconds_loss_coef",
        type=float,
        default=0.0,
        help=(
            "coefficient for an additional Smooth-L1 loss on absolute "
            "ready-time error normalized by --request_ready_seconds_loss_scale; "
            "zero preserves the historical log-time objective"
        ),
    )
    parser.add_argument(
        "--request_ready_seconds_loss_scale",
        type=float,
        default=600.0,
        help="seconds used to normalize the optional absolute-time Smooth-L1 loss",
    )
    parser.add_argument(
        "--request_ready_h1_weight",
        type=float,
        default=1.0,
        help="loss weight for bounded_mobile_frontier_h1 ready-time labels",
    )
    parser.add_argument(
        "--request_ready_h2_weight",
        type=float,
        default=1.0,
        help="loss weight for bounded_mobile_frontier_h2 ready-time labels",
    )
    parser.add_argument(
        "--request_ready_departure_weight",
        type=float,
        default=1.0,
        help="loss weight for departure pickup ready-time labels",
    )
    parser.add_argument(
        "--request_ready_underprediction_weight",
        type=float,
        default=1.0,
        help=(
            "multiplicative ready-time loss weight when the prediction is later "
            "than the model estimate (underprediction is operationally riskier)"
        ),
    )
    parser.add_argument(
        "--request_ready_blocking_weight",
        type=float,
        default=1.0,
        help="extra ready-time loss weight for already-blocking requests",
    )
    parser.add_argument(
        "--request_ready_exclude_blocking_loss",
        action="store_true",
        default=False,
        help=(
            "exclude deterministic blocking_wait targets from neural regression; "
            "use together with --request_ready_hard_blocking"
        ),
    )
    parser.add_argument(
        "--request_ready_kind_balanced_loss",
        action="store_true",
        default=False,
        help=(
            "average ready-time loss within request kinds before combining them "
            "so the frequent H1 frontier cannot dominate H2"
        ),
    )
    parser.add_argument(
        "--request_ready_quantile_loss_coef",
        type=float,
        default=0.0,
        help=(
            "coefficient for ordered q20/q50/q80 pinball loss; requires "
            "--request_ready_quantile_head"
        ),
    )
    parser.add_argument(
        "--request_ready_holdout_folds",
        type=int,
        default=0,
        help=(
            "number of deterministic case-hash folds for predictor-only holdout "
            "validation; zero disables the extra frozen validation pass"
        ),
    )
    parser.add_argument(
        "--request_ready_holdout_fold",
        type=int,
        default=0,
        help="zero-based holdout fold selected by case-path SHA256",
    )
    parser.add_argument(
        "--request_ready_min_labels_per_epoch",
        type=int,
        default=64,
        help=(
            "minimum finite request ready-time labels per Stage2 epoch; the "
            "canonical supervised pipeline fails closed below this value"
        ),
    )
    parser.add_argument(
        "--device_bc_dagger_schedule",
        type=str,
        default="1.0",
        help="comma-separated per-BC-epoch teacher execution probabilities",
    )
    parser.add_argument('--stage2_policy_improvement_protocol', action='store_true', default=False,
                        help='Common N0-N3 deterministic learner and exact selected-state prefix replay')
    parser.add_argument('--stage2_cost_improvement', action='store_true', default=False,
                        help='Bounded complete-joint Cmax improvement with frozen GPU student continuations')
    parser.add_argument('--stage2_cost_scale_seconds', type=float, default=120.)
    parser.add_argument('--stage2_cost_weight_clip', type=float, default=4.)
    parser.add_argument('--stage2_cost_tie_seconds', type=float, default=1.)
    parser.add_argument('--stage2_cost_loss_coef', type=float, default=1.)
    parser.add_argument('--stage2_cost_snapshot_horizon', type=int, default=256)
    parser.add_argument('--stage2_cost_branch_timeout_seconds', type=float, default=600.)
    parser.add_argument('--stage2_research_train_cases', type=str, default='',
                        help='Hash-bound train-only ordered cases for diagnostics/canary; never evaluation cases')
    parser.add_argument(
        "--device_bc_dagger_seed",
        type=int,
        default=1701,
        help="seed for environment-level Stage2 DAgger teacher execution",
    )
    parser.add_argument("--device_bc_min_labels_per_epoch", type=int, default=64,
                        help="minimum supervised device labels to collect per BC epoch before moving on")
    parser.add_argument(
        "--stage2_max_raw_regression_seconds",
        type=float,
        default=float("inf"),
        help=(
            "maximum post-supervised raw validation regression versus the "
            "frozen pre-supervised checkpoint before Best is rejected"
        ),
    )
    parser.add_argument(
        "--stage2_max_stress_regression_seconds",
        type=float,
        default=float("inf"),
        help=(
            "maximum OOD-stress validation regression versus pre-supervised "
            "before the Stage2 candidate is scientifically rejected"
        ),
    )
    parser.add_argument("--device_bc_min_rollouts_per_epoch", type=int, default=1,
                        help="minimum full vector rollouts to collect per BC epoch")
    parser.add_argument("--device_bc_max_rollouts_per_epoch", type=int, default=20,
                        help="maximum full rollouts to collect per BC epoch")
    parser.add_argument("--device_bc_train_gnn", action='store_true', default=False,
                        help="also update the shared GNN encoder during device behavior cloning")
    parser.add_argument("--device_bc_stochastic_plane", dest='device_bc_plane_deterministic',
                        action='store_false', default=True,
                        help="use stochastic loaded plane policy while collecting BC labels")
    parser.add_argument("--device_bc_skip_ready_targets", action='store_true', default=False,
                        help="frozen-ready IGA policy training only: do not query or score trajectory-bound ready labels")
    parser.add_argument("--no_device_bc_reset_optim", dest='device_bc_reset_optim',
                        action='store_false', default=True,
                        help="do not create/reset a PPO optimizer after supervised resource training")
    parser.add_argument("--no_device_bc_save", dest='device_bc_save',
                        action='store_false', default=True,
                        help="do not save a checkpoint after device BC warmup")
    parser.add_argument(
        "--resource_ppo_update_schedule",
        type=str,
        default="joint",
        choices=["joint", "ordinary_then_joint", "r014_then_joint"],
        help="legacy joint-RL role schedule; canonical Stage2 never consumes it",
    )
    parser.add_argument(
        "--resource_ppo_warmup_epochs",
        type=int,
        default=1,
        help="legacy joint-RL warmup length; canonical Stage2 never consumes it",
    )

    # specific for IA environment
    parser.add_argument("--obj", type=str, default='s', help="the object to train, default t(time)")
    parser.add_argument('--reward_coef', type=float, default=0.01, help="distance coefficient")
    parser.add_argument('--reward_coef_t', type=float, default=0.01, help="time coefficient")
    parser.add_argument('--reward_coef_c', type=float, default=1.0, help="fuel coefficient")
    parser.add_argument("--fuse_s", action='store_false', default=True, help="whether to fuse s(distance) into reward")
    parser.add_argument("--auto_fuse", action='store_false', default=True, help="whether to fuse s(distance) into reward")
    parser.add_argument('--start_epoch', type=int, default=60, help="the epoch to start fusing reward, default 10")
    parser.add_argument('--fuse_epoch', type=int, default=20, help="the epoch to start fusing reward, default 10")
    parser.add_argument('--max_agent_num', type=int, default=24, help="the max number of plane agents, default 6")
    parser.add_argument('--max_device_num', type=int, default=0, help="the max number of mobile device agents")
    parser.add_argument('--resource_policy', type=str, default='heuristic', choices=['heuristic', 'drl'],
                        help="mobile resource dispatch policy: heuristic or drl")
    parser.add_argument(
        '--device_lookahead_dispatch',
        action='store_true',
        default=False,
        help=(
            "expose deferrable mobile-resource requests after a plane commits "
            "to an operation/site pair, before the plane starts waiting"
        ),
    )
    parser.add_argument(
        '--device_lookahead_safety_margin',
        type=float,
        default=60.0,
        help=(
            "maximum seconds a mobile device may arrive before a lookahead "
            "request becomes blocking"
        ),
    )
    parser.add_argument(
        '--device_deadline_aware_dispatch',
        action='store_true',
        default=False,
        help=(
            'mask a lookahead dispatch until travel time enters its JIT '
            'deadline window and wake the event loop at that boundary'
        ),
    )
    parser.add_argument(
        '--device_future_intent_horizon',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help=(
            'number of committed dependency edges exposed as mobile-resource '
            'intents; horizons above one use a bounded, soft-reservation '
            'dependency frontier at the currently committed stand'
        ),
    )
    parser.add_argument(
        '--device_future_intent_mode',
        type=str,
        default='legacy_one',
        choices=['legacy_one', 'bounded_frontier'],
        help=(
            'legacy_one exposes one direct successor; bounded_frontier '
            'exposes the highest-priority direct mobile-resource successors '
            'without changing the fixed request tensor width'
        ),
    )
    parser.add_argument(
        '--device_frontier_max_requests',
        type=int,
        default=2,
        help='maximum dependency-frontier intents exposed per aircraft',
    )
    parser.add_argument(
        '--device_request_capacity_per_plane',
        type=int,
        default=0,
        help=(
            'fixed padded request slots per aircraft; zero derives the '
            'smallest safe width from the configured lookahead frontier. '
            'Causal arms sharing one validator must use the same nonzero width'
        ),
    )
    parser.add_argument(
        '--resource_release_aware_eta',
        action='store_true',
        default=False,
        help=(
            'include in-service release time in device ETA/features and treat '
            'a job resource list as alternatives instead of simultaneous needs'
        ),
    )
    parser.add_argument(
        '--device_lookahead_reservation_mode',
        type=str,
        default='none',
        choices=['none', 'soft', 'hard'],
        help=(
            'lease a device pre-positioned for a future ordinary-resource '
            'request; soft leases may be stolen by a blocking request'
        ),
    )
    parser.add_argument(
        '--device_reservation_grace_seconds',
        type=float,
        default=300.0,
        help='bounded grace period after predicted need before a lease expires',
    )
    parser.add_argument(
        '--device_departure_lookahead',
        action='store_true',
        default=False,
        help='expose an R014 pickup intent during the final service operation',
    )
    parser.add_argument('--train_domain_rand', action='store_true', default=False,
                        help="enable domain randomization during joint-policy fine-tuning")
    parser.add_argument(
        '--plane_order_mode',
        type=str,
        default='fixed',
        choices=['fixed', 'learned'],
        help=(
            'fixed plane-id traversal without an order head; learned is kept '
            'only for legacy checkpoint compatibility'
        ),
    )
    parser.add_argument(
        '--plane_pair_decoder',
        type=str,
        default='joint_pair',
        choices=['cascade', 'joint_pair'],
        help='joint operation-site pair scorer; cascade is kept for legacy compatibility',
    )
    parser.add_argument(
        '--stage1_baseline',
        type=str,
        default='proposed',
        choices=['proposed', 'l2d', 'multi_ppo', 'fjsp_drl', 'daniel'],
        help=(
            'Stage-1 scheduling architecture. Non-proposed choices use the '
            'same HKBZ environment, masks, PPO budget and heuristic Hungarian '
            'resource backend.'
        ),
    )
    parser.add_argument(
        '--plane_bc_pretrain_epochs',
        type=int,
        default=0,
        help='epochs of IGA-teacher behavior cloning before Stage-1 PPO',
    )
    parser.add_argument(
        '--plane_bc_only', action='store_true', default=False,
        help=(
            'stop after PlaneBC and the deterministic Pre-PPO validation; '
            'used for causally clean BC screening'
        ),
    )
    parser.add_argument('--plane_bc_teacher_dir', type=str, default='',
                        help='directory containing replayable IGA teacher JSON files')
    parser.add_argument('--plane_bc_lr', type=float, default=0.0,
                        help='plane IGA behavior-cloning LR; <=0 uses --lr')
    parser.add_argument(
        '--plane_bc_shared_lr_scale',
        type=float,
        default=0.3,
        help='shared-GNN learning-rate multiplier during plane IGA BC',
    )
    parser.add_argument(
        '--plane_bc_freeze_shared_epochs',
        type=int,
        default=0,
        help='initial plane-BC epochs that train only the plane backend',
    )
    parser.add_argument(
        '--plane_bc_pair_loss_coef',
        type=float,
        default=1.0,
        help='weight of operation-site imitation loss during plane BC',
    )
    parser.add_argument(
        '--plane_bc_order_loss_coef',
        type=float,
        default=0.0,
        help=(
            'legacy learned-order imitation weight; must be zero when '
            '--plane_order_mode=fixed'
        ),
    )
    parser.add_argument(
        '--plane_bc_rollouts_per_epoch', type=int, default=0,
        help='teacher rollouts per BC epoch; <=0 covers every case shard once',
    )
    parser.add_argument(
        '--plane_bc_dagger_schedule',
        type=str,
        default='',
        help=(
            'comma-separated whole-environment teacher execution rates per BC '
            'epoch, e.g. 1,0.75,0.5,0.25,0.1,0; empty keeps teacher forcing'
        ),
    )
    parser.add_argument(
        '--plane_bc_dagger_seed',
        type=int,
        default=0,
        help='independent seed for reproducible DAgger teacher/student mixing',
    )
    parser.add_argument(
        '--plane_bc_per_agent_dagger', action='store_true', default=False,
        help=(
            'sample teacher/student execution independently for each active '
            'plane and repair site conflicts with a feasible assignment'
        ),
    )
    parser.add_argument(
        '--plane_bc_staging_dagger_schedule', type=str, default='',
        help=(
            'optional per-epoch teacher rates for ZY-T staging decisions; an '
            'empty value inherits --plane_bc_dagger_schedule'
        ),
    )
    parser.add_argument(
        '--plane_bc_dagger_tail_start_fraction', type=float, default=1.0,
        help=(
            'teacher-trajectory progress where risk-triggered DAgger begins; '
            '1.0 disables tail-specific teacher execution'
        ),
    )
    parser.add_argument(
        '--plane_bc_dagger_tail_teacher_rate', type=float, default=0.0,
        help=(
            'minimum teacher execution rate after the tail-risk threshold; '
            'the per-epoch DAgger rate remains the lower bound elsewhere'
        ),
    )
    parser.add_argument(
        '--plane_bc_phase_aware', action='store_true', default=False,
        help=(
            'stratify BC by service, ZY-T hold/move, and forced departure '
            'phases; legacy weighting remains the default'
        ),
    )
    parser.add_argument(
        '--plane_bc_service_weight', type=float, default=1.0,
        help='base BC multiplier for service-phase operation-site decisions',
    )
    parser.add_argument(
        '--plane_bc_staging_hold_weight', type=float, default=1.0,
        help='BC multiplier for a ZY-T decision that keeps the current site',
    )
    parser.add_argument(
        '--plane_bc_staging_move_weight', type=float, default=2.0,
        help='BC multiplier for a ZY-T decision that vacates the current site',
    )
    parser.add_argument(
        '--plane_bc_service_tail_start_fraction', type=float, default=1.0,
        help=(
            'per-plane service-completion fraction where service BC emphasis '
            'starts; 1.0 disables this phase-local ramp'
        ),
    )
    parser.add_argument(
        '--plane_bc_service_tail_weight', type=float, default=1.0,
        help='maximum service BC multiplier at 100 percent service progress',
    )
    parser.add_argument(
        '--global_feature_mode',
        type=str,
        default='none',
        choices=['none', 'f1', 'f1f2', 'f1f2_departure'],
        help=(
            'global scheduling context: none is the legacy control, f1 adds '
            'workload/arrival summaries, f1f2 adds resource summaries, and '
            'f1f2_departure substitutes explicit R014/runway/queue summaries '
            'without changing the 24-dimensional encoder input'
        ),
    )
    parser.add_argument(
        '--plane_bc_initial_weight', type=float, default=5.0,
        help='BC multiplier for the first operation-site choice of each plane',
    )
    parser.add_argument(
        '--plane_bc_relocation_weight', type=float, default=2.0,
        help='BC multiplier for teacher actions that move to another site',
    )
    parser.add_argument(
        '--plane_bc_critical_op_weight', type=float, default=2.0,
        help='BC multiplier for configured critical operations (ZY10/ZY12)',
    )
    parser.add_argument(
        '--plane_bc_tail_start_fraction', type=float, default=1.0,
        help=(
            'teacher-trajectory progress where smooth tail emphasis begins; '
            '1.0 disables temporal tail weighting'
        ),
    )
    parser.add_argument(
        '--plane_bc_tail_weight', type=float, default=1.0,
        help=(
            'maximum BC state weight at the end of the IGA teacher trajectory; '
            'must be at least 1'
        ),
    )
    parser.add_argument(
        '--plane_bc_tail_final_start_fraction', type=float, default=1.0,
        help=(
            'progress where a second, steeper final-tail BC ramp begins; 1.0 '
            'keeps the legacy one-ramp weighting'
        ),
    )
    parser.add_argument(
        '--plane_bc_tail_final_weight', type=float, default=-1.0,
        help=(
            'maximum BC weight at trajectory completion for the second ramp; '
            'a negative value inherits --plane_bc_tail_weight'
        ),
    )
    parser.add_argument(
        '--joint_team_ppo', action='store_true', default=False,
        help='form one PPO ratio from the simultaneous plane action tuple',
    )
    parser.add_argument(
        '--joint_team_ppo_scope',
        type=str,
        default='plane',
        choices=['plane', 'all'],
        help=(
            'roles included in the joint PPO ratio; Stage1 uses plane and '
            'Stage3 joint_finetune requires all'
        ),
    )
    parser.add_argument(
        '--role_atomic_ppo', action='store_true', default=False,
        help=(
            'form a separate joint-event PPO ratio for plane, ordinary-device, '
            'and transporter roles, combine their scalar losses, and apply one '
            'atomic optimizer step'
        ),
    )
    parser.add_argument(
        '--role_event_returns', action='store_true', default=False,
        help=(
            'construct team-time returns on each role\'s own physical decision '
            'event sequence instead of broadcasting through the mixed timeline'
        ),
    )
    parser.add_argument(
        '--role_event_credit_mode', type=str, default='elapsed',
        choices=['elapsed', 'critical_path'],
        help=(
            'elapsed keeps the exact physical-time reward decomposition; '
            'critical_path redistributes the same role return over audited '
            'post-episode Cmax-frontier and resource-lateness event scores'
        ),
    )
    parser.add_argument(
        '--role_event_credit_uniform_mix', type=float, default=0.15,
        help=(
            'fraction of the physical elapsed-time decomposition retained '
            'when --role_event_credit_mode critical_path is active'
        ),
    )
    parser.add_argument(
        '--counterfactual_q_baseline', action='store_true', default=False,
        help=(
            'interpret each existing role StepCritic as Q(s,a), train it on '
            'the replayed action return, and use the policy-weighted legal '
            'action expectation as the rollout control variate'
        ),
    )
    parser.add_argument(
        '--counterfactual_q_topk', type=int, default=8,
        help='maximum legal policy actions used in the Q expectation',
    )
    parser.add_argument(
        '--counterfactual_q_min_mass', type=float, default=0.90,
        help=(
            'diagnostic target for probability mass represented by the '
            'counterfactual top-k action set'
        ),
    )
    parser.add_argument(
        '--role_sequential_ppo', action='store_true', default=False,
        help=(
            'apply role heads in a rotating HAPPO-style sequence while '
            'retaining one macro rollback boundary'
        ),
    )
    parser.add_argument(
        '--role_sequential_factor_clip', type=float, default=2.0,
        help='symmetric cap for preceding-role importance products',
    )
    parser.add_argument(
        '--role_sequential_min_ess', type=float, default=0.50,
        help='minimum normalized ESS accepted for a sequential macro update',
    )
    parser.add_argument(
        '--role_event_gae_lambda', type=float, default=1.0,
        help=(
            'GAE lambda on each role-specific physical event clock; 1.0 '
            'exactly reproduces the role Monte-Carlo target'
        ),
    )
    for role_name in ('plane', 'device', 'transporter'):
        parser.add_argument(
            f'--{role_name}_role_gae_lambda',
            type=float,
            default=-1.0,
            help=(
                f'{role_name}-clock GAE lambda; a negative value inherits '
                '--role_event_gae_lambda'
            ),
        )
    parser.add_argument(
        '--role_loss_weighting', type=str, default='fixed',
        choices=['fixed', 'sqrt_event'],
        help=(
            'fixed uses plane/device/R014 coefficients; sqrt_event assigns '
            'each case role mass proportional to the square root of its role '
            'event count'
        ),
    )
    parser.add_argument(
        '--role_loss_min_share', type=float, default=0.15,
        help='minimum per-case role mass in sqrt_event weighting',
    )
    parser.add_argument(
        '--role_loss_max_share', type=float, default=0.60,
        help='maximum per-case role mass in sqrt_event weighting',
    )
    parser.add_argument(
        '--role_valuenorm', action='store_true', default=False,
        help='maintain independent ValueNorm statistics for the three roles',
    )
    parser.add_argument(
        '--shared_gradient_diagnostics', action='store_true', default=False,
        help=(
            'measure per-role shared-encoder gradient norms/cosines on one '
            'probe replay sample per PPO shard'
        ),
    )
    parser.add_argument(
        '--shared_encoder_pcgrad', action='store_true', default=False,
        help=(
            'project conflicting plane/device/R014 policy gradients only on '
            'the unsplit shared encoder before the atomic Actor step'
        ),
    )
    parser.add_argument(
        '--shared_gradient_method', type=str, default='sum',
        choices=['sum', 'norm_balance', 'norm_pcgrad', 'cagrad'],
        help=(
            'shared-encoder role-gradient combiner. sum preserves the ordinary '
            'joint loss; norm_balance applies bounded EMA norm balancing; '
            'norm_pcgrad additionally projects severe conflicts symmetrically; '
            'cagrad uses a conflict-averse common direction. The legacy '
            '--shared_encoder_pcgrad switch retains its historical projection.'
        ),
    )
    parser.add_argument(
        '--shared_grad_ema_beta', type=float, default=0.97,
        help='EMA decay used by shared role-gradient norm balancing',
    )
    parser.add_argument(
        '--shared_grad_norm_power', type=float, default=0.5,
        help='power applied to target/EMA norm ratios',
    )
    parser.add_argument(
        '--shared_grad_min_scale', type=float, default=0.5,
        help='minimum role-gradient scale after EMA norm balancing',
    )
    parser.add_argument(
        '--shared_grad_max_scale', type=float, default=2.0,
        help='maximum role-gradient scale after EMA norm balancing',
    )
    parser.add_argument(
        '--shared_grad_conflict_threshold', type=float, default=-0.05,
        help=(
            'norm_pcgrad projects only role pairs whose current cosine is '
            'below this threshold'
        ),
    )
    parser.add_argument(
        '--shared_cagrad_c', type=float, default=0.2,
        help='CAGrad conflict-aversion coefficient in [0, 1)',
    )
    parser.add_argument(
        '--stage3_allow_shared_frozen', action='store_true', default=False,
        help=(
            'explicit Stage3 research arm that keeps the shared GNN frozen for '
            'the complete run while training all role-specific heads'
        ),
    )
    parser.add_argument(
        '--stage3_handoff_mode',
        type=str,
        default='strict',
        choices=['strict', 'critical_path_wave1', 'ppo_gain_wave'],
        help=(
            'Stage2-to-Stage3 contract mode. critical_path_wave1 permits only '
            'the preregistered horizon/frontier and policy-invariant '
            'critical-resource-potential transitions; ppo_gain_wave permits '
            'the same horizon=3/frontier=4 planning transition and either the '
            'exact Stage2 reward or a fully unshaped pure-Cmax reward, while '
            'keeping the model handoff bit exact'
        ),
    )
    parser.add_argument(
        '--plane_target_kl', type=float, default=0.0025,
        help='post-update joint-event KL limit for the plane role',
    )
    parser.add_argument(
        '--device_target_kl', type=float, default=0.005,
        help='post-update joint-event KL limit for ordinary mobile devices',
    )
    parser.add_argument(
        '--transporter_target_kl', type=float, default=0.005,
        help='post-update joint-event KL limit for R014 transporters',
    )
    parser.add_argument(
        '--central_team_critic', action='store_true', default=False,
        help='use one graph-level value for all simultaneous plane decisions',
    )
    parser.add_argument(
        '--hindsight_reward_mode', type=str, default='team_cmax',
        choices=[
            'shaping',
            'cmax_delta',
            'potential_cmax',
            'iga_potential',
            'hybrid_cmax',
            'team_cmax',
            'team_time',
            'team_time_potential',
            'team_time_resource_potential',
            'team_time_resource_fitted_potential',
        ],
        help=(
            'HKBZ reward mode; team_cmax broadcasts one terminal target; '
            'team_time uses global negative remaining makespan at each '
            'decision; team_time_potential adds a versioned departure-aware '
            'IGA-calibrated telescoping potential; '
            'team_time_resource_potential instead uses an online critical '
            'resource-slack potential without changing the terminal objective; '
            'team_time_resource_fitted_potential uses the same online resource '
            'features with weights fitted only from matched Stage2 IGA replays'
        ),
    )
    parser.add_argument('--hindsight_cmax_coef', type=float, default=0.0,
                        help="coefficient for cmax_delta hindsight reward")
    parser.add_argument('--hindsight_shaping_coef', type=float, default=0.0,
                        help="coefficient for shaping component in non-team hindsight modes")
    parser.add_argument('--hindsight_terminal_cmax_coef', type=float, default=1.0,
                        help="coefficient for the single case-level team Cmax return")
    parser.add_argument(
        '--resource_lateness_coef', type=float, default=0.0,
        help='auxiliary team-return penalty per second of total aircraft resource wait',
    )
    parser.add_argument(
        '--resource_critical_lateness_coef', type=float, default=0.0,
        help='auxiliary team-return penalty per second of worst-aircraft resource wait',
    )
    parser.add_argument(
        '--resource_earliness_coef', type=float, default=0.0,
        help='anti-hoarding team-return penalty per second of predicted early arrival',
    )
    parser.add_argument(
        '--resource_wait_constraint_target', type=float, default=0.0,
        help=(
            'target mean per-case total mobile-resource wait in seconds; zero '
            'disables the adaptive Lagrange multiplier'
        ),
    )
    parser.add_argument(
        '--resource_wait_dual_lr', type=float, default=0.0,
        help='epoch-level normalized dual-ascent rate for the wait constraint',
    )
    parser.add_argument(
        '--resource_wait_dual_max', type=float, default=0.05,
        help='upper bound for the adaptive resource-wait penalty coefficient',
    )
    parser.add_argument(
        '--iga_potential_weights_path', type=str, default='',
        help='JSON weights calibrated from exact IGA teacher trajectory replays',
    )
    parser.add_argument(
        '--iga_potential_beta', type=float, default=0.0,
        help=(
            'scale of team-time potential shaping (IGA-calibrated or online '
            'resource-slack, according to --hindsight_reward_mode)'
        ),
    )
    parser.add_argument(
        '--iga_potential_beta_schedule', type=str, default='',
        help=(
            'optional comma-separated per-PPO-epoch potential beta values; '
            'the final value is held for later epochs and an empty schedule '
            'preserves --iga_potential_beta'
        ),
    )
    parser.add_argument(
        '--iga_potential_gamma', type=float, default=0.99,
        help='discount used in gamma * Phi(next_state) - Phi(state)',
    )
    parser.add_argument(
        '--resource_slack_criticality_seconds',
        type=float,
        default=1800.0,
        help=(
            'remaining-work slack scale used to emphasize mobile-resource '
            'requests on the estimated Cmax-critical aircraft'
        ),
    )
    parser.add_argument(
        '--resource_slack_min_weight',
        type=float,
        default=0.25,
        help=(
            'minimum request weight in the online critical-path potential; '
            'zero fully suppresses large-slack aircraft and one disables '
            'slack discrimination'
        ),
    )
    parser.add_argument(
        '--resource_slack_forecast_seconds',
        type=float,
        default=0.0,
        help=(
            'seconds of state cost per excess arrival within the existing '
            '1800-second lookahead window; zero disables forecast pressure'
        ),
    )
    parser.add_argument(
        '--tail_policy_start_fraction', type=float, default=1.0,
        help=(
            'episode-time fraction where smooth late-decision PPO emphasis '
            'starts; 1.0 disables it'
        ),
    )
    parser.add_argument(
        '--tail_policy_weight', type=float, default=1.0,
        help=(
            'maximum late-decision policy weight at Cmax; weights are '
            'renormalized per case so every case keeps the same total mass'
        ),
    )
    parser.add_argument(
        '--cvar_policy_fraction', type=float, default=1.0,
        help=(
            'fraction of rollout cases in the high-Cmax tail that receive '
            'extra PPO mass; 1.0 disables cross-case CVaR reweighting'
        ),
    )
    parser.add_argument(
        '--cvar_policy_weight', type=float, default=1.0,
        help=(
            'relative policy/value mass assigned to the high-Cmax tail; '
            'the rollout-wide total mass is conserved'
        ),
    )
    parser.add_argument(
        '--cvar_case_metric', type=str, default='cmax',
        choices=['cmax', 'paired_delta'],
        help=(
            'rank CVaR cases by raw Cmax or by the paired Cmax excess over '
            'the configured per-case teacher baseline'
        ),
    )
    parser.add_argument(
        '--paired_case_baseline_dir', type=str, default='',
        help=(
            'directory of verified per-case IGA JSON files used as a '
            'variance-reducing paired terminal baseline'
        ),
    )
    parser.add_argument(
        '--paired_case_baseline_coef', type=float, default=0.0,
        help=(
            'add this fraction of the matched reference Cmax to the negative '
            'terminal return; 1.0 trains directly on -delta_Cmax'
        ),
    )
    parser.add_argument(
        '--paired_case_baseline_scope', type=str, default='returns',
        choices=['returns', 'actor'],
        help=(
            "apply the paired case baseline to both Actor/Critic return "
            "targets (legacy 'returns') or only as an action-independent "
            "Actor advantage offset ('actor')"
        ),
    )
    parser.add_argument('--device_deadlock_repeat_limit', type=int, default=300,
                        help="raise a diagnostic error after the same active device-request state repeats this many times")
    parser.add_argument('--plane_cycle_repeat_limit', type=int, default=8,
                        help="terminate after the same plane post-decision state repeats this many times")
    parser.add_argument('--plane_no_progress_limit', type=int, default=120,
                        help="terminate a rollout after this many plane decisions without irreversible progress")
    parser.add_argument('--plane_relocation_limit', type=int, default=40,
                        help="terminate after this many relocations without irreversible progress")
    parser.add_argument('--plane_first_completion_bonus', type=float, default=120.0,
                        help="hindsight reward for each first-time job completion")
    parser.add_argument('--plane_repeat_relocation_penalty', type=float, default=300.0,
                        help="penalty for returning to the previously departed stand without progress")
    parser.add_argument('--plane_reset_job_penalty', type=float, default=120.0,
                        help="penalty per long-occupancy job reset by relocation")
    parser.add_argument('--plane_no_progress_penalty', type=float, default=60.0,
                        help="penalty per relocation action without irreversible progress")
    parser.add_argument('--plane_cycle_penalty', type=float, default=20000.0,
                        help="terminal hindsight penalty when the plane cycle guard fires")
    parser.add_argument('--grad_accumulation_steps', type=int, default=5, help="the number of gradient accumulation steps, default 5")    
    parser.add_argument(
        '--grad_accumulation_target_graphs', type=int, default=0,
        help=(
            "critic optimizer target graphs per accumulation group; >0 "
            "balances whole microbatches by graph count instead of using a "
            "fixed number of accumulation steps"
        ),
    )
    parser.add_argument(
        '--actor_grad_accumulation_steps', type=int, default=0,
        help="actor-only accumulation steps; <=0 reuses --grad_accumulation_steps",
    )
    parser.add_argument(
        '--actor_grad_accumulation_target_graphs', type=int, default=0,
        help=(
            "actor optimizer target graphs per accumulation group; >0 "
            "preserves effective optimizer batch mass when the physical "
            "graph forward size changes"
        ),
    )
    return parser
