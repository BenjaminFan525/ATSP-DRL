import argparse
import json


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
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=100,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=20,
                        help="Number of parallel envs for evaluating rollouts")
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
        "--train_sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "distribution_balanced"],
        help=(
            "HKBZ case sampler; distribution_balanced oversamples metadata "
            "distributions to the requested mixture without discarding unique "
            "training cases when no explicit sample-size cap is set"
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
    parser.add_argument("--actor_warmup_shards", type=int, default=0,
                        help="critic-only rollout shards before the first actor PPO update")
    parser.add_argument("--clear_cuda_cache_after_update", action="store_true", default=False,
                        help="release unused CUDA allocator blocks after each rollout update")

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
        "--plane_order_freeze_epochs",
        type=int,
        default=0,
        help="number of initial Stage-1 PPO epochs that freeze only the learned plane-order head",
    )
    parser.add_argument(
        "--training_stage",
        type=str,
        default="auto",
        choices=["auto", "plane_pretrain", "device_bc", "frozen_joint", "full_joint"],
        help="explicit HKBZ training stage; auto preserves legacy single-process behavior",
    )
    parser.add_argument(
        "--resume_stage1",
        action="store_true",
        default=False,
        help="explicitly allow plane_pretrain to restore a same-stage recovery checkpoint",
    )
    parser.add_argument(
        "--reset_optimizers_on_resume",
        action="store_true",
        default=False,
        help="restore model weights but start with fresh actor and critic optimizers",
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
    parser.add_argument("--recovery_checkpoint_interval_shards", type=int, default=1,
                        help="overwrite checkpoint_Recovery.pt every N completed shards; <=0 disables")

    # eval parameters
    parser.add_argument("--use_eval", dest='use_eval', action='store_true',
                        help="run deterministic C_max validation during training")
    parser.add_argument("--no_eval", dest='use_eval', action='store_false',
                        help="disable validation during training")
    parser.set_defaults(use_eval=False)
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="number of training epochs between C_max validations")
    parser.add_argument("--canary_eval_interval_shards", type=int, default=0,
                        help="run an in-epoch deterministic validation every N shards; 0 disables")
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
        choices=["iid", "composite"],
        help="checkpoint selection by IID mean or weighted IID/OOD validation",
    )
    parser.add_argument("--selection_iid_weight", type=float, default=0.50)
    parser.add_argument("--selection_ood_stress_weight", type=float, default=0.45)
    parser.add_argument("--selection_ood_scale_weight", type=float, default=0.05)
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
                        help="supervised warmup epochs for mobile device and transporter policy heads before PPO")
    parser.add_argument("--device_bc_lr", type=float, default=0.0,
                        help="learning rate for device behavior cloning warmup; <=0 uses --lr")
    parser.add_argument("--device_bc_min_labels_per_epoch", type=int, default=64,
                        help="minimum supervised device labels to collect per BC epoch before moving on")
    parser.add_argument("--device_bc_min_rollouts_per_epoch", type=int, default=1,
                        help="minimum full vector rollouts to collect per BC epoch")
    parser.add_argument("--device_bc_max_rollouts_per_epoch", type=int, default=20,
                        help="maximum full rollouts to collect per BC epoch")
    parser.add_argument("--device_bc_train_gnn", action='store_true', default=False,
                        help="also update the shared GNN encoder during device behavior cloning")
    parser.add_argument("--device_bc_stochastic_plane", dest='device_bc_plane_deterministic',
                        action='store_false', default=True,
                        help="use stochastic loaded plane policy while collecting BC labels")
    parser.add_argument("--no_device_bc_reset_optim", dest='device_bc_reset_optim',
                        action='store_false', default=True,
                        help="do not reset PPO optimizers after device BC warmup")
    parser.add_argument("--no_device_bc_save", dest='device_bc_save',
                        action='store_false', default=True,
                        help="do not save a checkpoint after device BC warmup")

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
        '--plane_bc_pretrain_epochs',
        type=int,
        default=0,
        help='epochs of IGA-teacher behavior cloning before Stage-1 PPO',
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
        '--global_feature_mode',
        type=str,
        default='none',
        choices=['none', 'f1', 'f1f2'],
        help=(
            'global scheduling context: none is the legacy control, f1 adds '
            'workload/arrival summaries, f1f2 adds resource summaries'
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
        '--joint_team_ppo', action='store_true', default=False,
        help='form one PPO ratio from the simultaneous plane action tuple',
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
        ],
        help=(
            'HKBZ reward mode; team_cmax broadcasts one terminal target; '
            'team_time uses global negative remaining makespan at each decision'
        ),
    )
    parser.add_argument('--hindsight_cmax_coef', type=float, default=0.0,
                        help="coefficient for cmax_delta hindsight reward")
    parser.add_argument('--hindsight_shaping_coef', type=float, default=0.0,
                        help="coefficient for shaping component in non-team hindsight modes")
    parser.add_argument('--hindsight_terminal_cmax_coef', type=float, default=1.0,
                        help="coefficient for the single case-level team Cmax return")
    parser.add_argument(
        '--iga_potential_weights_path', type=str, default='',
        help='JSON weights calibrated from exact IGA teacher trajectory replays',
    )
    parser.add_argument(
        '--iga_potential_beta', type=float, default=0.0,
        help='scale of IGA-calibrated potential-based reward shaping',
    )
    parser.add_argument(
        '--iga_potential_gamma', type=float, default=0.99,
        help='discount used in gamma * Phi(next_state) - Phi(state)',
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
        '--actor_grad_accumulation_steps', type=int, default=0,
        help="actor-only accumulation steps; <=0 reuses --grad_accumulation_steps",
    )
    return parser
