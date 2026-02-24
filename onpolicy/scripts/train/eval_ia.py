import gym
import imageio
from PIL import Image, ImageDraw
import numpy as np
import torch
import sys
import os
from gym import spaces

# os.environ["MUJOCO_GL"] = "egl"

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)
# sys.path.append(parent_dir+"/onpolicy/algorithm")
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.IA.IA_env import IAEnv
from onpolicy.config.config import get_config

def _t2n(x):
    return x.detach().cpu().numpy()

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    parser.add_argument("--harv_vmin", type=float, default=1.0)
    parser.add_argument("--harv_vmax", type=float, default=2.0)
    parser.add_argument("--harv_capmin", type=int, default=14)
    parser.add_argument("--harv_capmax", type=int, default=20)
    parser.add_argument("--trans_vmin", type=float, default=5.0)
    parser.add_argument("--trans_vmax", type=float, default=8.0)
    parser.add_argument("--trans_capmin", type=int, default=70)
    parser.add_argument("--trans_capmax", type=int, default=100)
    
    # environment settings
    parser.add_argument('--dt', type=float, default=0.1, help="simulation interval")
    parser.add_argument('--decision_dt', type=float, default=5.0, help="decision interval")
    # Other settings
    parser.add_argument("--shared_reward", action='store_true', default=False, help='Whether agent share the same rewadr')
    parser.add_argument('--wait_time_factor', type=float, default=10.0, help="wait time factor")
    parser.add_argument('--distance_factor', type=float, default=0.01, help="distanc factor")
    parser.add_argument('--trans_times_factor', type=float, default=10.0, help="trans times factor")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # np.random.seed(7)
    env = IAEnv(all_args)
    eval_episode_rewards = []
    actors = []
    obs = env.reset()

    eval_rnn_states = np.zeros((1),dtype=np.float32)
    eval_masks = np.ones((1), dtype=np.float32)

    for i in range(all_args.num_transporter):
        act = R_Actor(all_args,env.observation_space[i],env.action_space[i])
        act.load_state_dict(torch.load("./onpolicy/scripts/results/IA/ia_simple/mappo/check/run3/models/actor.pt"))
        actors.append(act)
        # print(act.act.action_out.logstd._bias)

    frames = []

    with torch.no_grad():
        # print(env.env.initial_gripper1_pos,env.env.initial_gripper1_rot,env.env.initial_gripper2_pos,env.env.initial_gripper2_rot)
        for eval_step in range(all_args.episode_length):
            # print("step: ", eval_step, "\n")
            # env.render()
            # img = env.render("rgb_array")[0]
            # frames.append(img)
            action = []
            for agent_id in range(all_args.num_transporter):
                actor = actors[agent_id]
                actor.eval()
                # print(actor.act.action_out.log_std)
                # print("agent id: ", agent_id, "observation: ", obs[agent_id], "\n")
                eval_action,_,rnn_states_actor = actor(
                    obs[agent_id],
                    eval_rnn_states,
                    eval_masks,
                    deterministic=True,
                )
                eval_action = eval_action.detach().cpu().numpy()
                # print("action: ",eval_action)
                action.append(eval_action)

            # print("action: ", action)
            obs, eval_rewards, done, infos = env.step(np.stack(action).squeeze())
            # print("reward: ",eval_rewards)
            # print("action: ",np.stack(action).squeeze().reshape(all_args.num_agents,3))
            eval_episode_rewards.append(eval_rewards)

            if np.all(done):
                break
            # if i % 100 == 0:
            #     imageio.imsave(f"env_auto_trans_mode{i}.jpg", img)

        for h in range(all_args.num_harvester):
            print(f"Harvester{h} total wait time: ", env.world.harvesters[h].total_wait_time)
        for t in range(all_args.num_transporter):
            print(f"Transporter{t} total trip: ", env.world.transporters[t].total_trip)
            print(f"Transporter{t} trans times: ", env.world.transporters[t].trans_times)

        print("episode reward: ",np.array(eval_episode_rewards).sum())
        
        # imageio.mimsave("./render/trained_policy_3_2.mp4", frames, fps=10)  

if __name__ == "__main__":
    main(sys.argv[1:])