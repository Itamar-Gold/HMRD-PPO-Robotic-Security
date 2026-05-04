import torch
import json
import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.simulation.engine import DroneSimulation
from src.algorithms.ppo import CDFC_Actor, CFC_Actor, Critic, PPO
from src.algorithms.rewards import compute_density_penalty, compute_detect_density
from src.utils.save import save_best_config, save_to_file, save_models
from src.utils.geometry import get_cameras_pointers

def flatten_list(list_scores):
    flattened_list = []
    for item in list_scores:
        if isinstance(item, tuple):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    return flattened_list

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed = config['hmrd_ppo']['SEED']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    results_dir = config['evaluation']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    for level in [1, 2, 3, 4, 5, 6]:
        for critic_lr in config['hmrd_ppo']['critic_lr_list']:
            for cdfc_actor_lr in config['hmrd_ppo']['cdfc_actor_lr_list']:
                for cfc_actor_lr in config['hmrd_ppo']['cfc_actor_lr_list']:

                    excel_file = os.path.join(os.path.dirname(__file__), '..', config['data']['placement_path'])
                    interdetec_df = pd.read_excel(os.path.abspath(excel_file), sheet_name="Placements")
                    launch_pads = interdetec_df["Launch Pads"].apply(
                        lambda x: eval(x) if isinstance(x, str) else x).tolist()
                    cameras = interdetec_df["Cameras"].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()

                    if isinstance(launch_pads[0], list): launch_pads = launch_pads[0]
                    if isinstance(cameras[0], list): cameras = cameras[0]

                    num_launch_pads = len(launch_pads)
                    num_cameras = len(cameras)

                    extras_cameras = config['hmrd_ppo']['extras_cameras']
                    extras_launch_pads = config['hmrd_ppo']['extras_launch_pads']
                    capture_radius = config['simulation']['capture_radius']
                    camera_range = config['simulation']['camera_range']

                    cdfc_state = launch_pads
                    cfc_state = cameras

                    if extras_cameras:
                        num_cameras += extras_cameras
                        for _ in range(extras_cameras):
                            cfc_state.extend([tuple(config['hmrd_ppo']['extras_camera_pos'])])
                    if extras_launch_pads:
                        num_launch_pads += extras_launch_pads
                        for _ in range(extras_launch_pads):
                            cdfc_state.extend([tuple(config['hmrd_ppo']['extras_launch_pad_pos'])])

                    print(cdfc_state)
                    print(cfc_state)

                    simulation = DroneSimulation(config_path, cdfc_state, cfc_state)

                    critic_lr = float(critic_lr)
                    cdfc_actor_lr = float(cdfc_actor_lr)
                    cfc_actor_lr = float(cfc_actor_lr)

                    cdfc_actor = CDFC_Actor(num_launch_pads * 3 + num_cameras * 3, num_launch_pads * 2).to(device)
                    cdfc_critic = Critic(num_launch_pads * 3 + num_cameras * 3).to(device)
                    cfc_actor = CFC_Actor(num_launch_pads * 3 + num_cameras * 3, num_cameras * 3).to(device)
                    cfc_critic = Critic(num_launch_pads * 3 + num_cameras * 3).to(device)

                    simulation.level = level
                    simulation.init_level()

                    cdfc_agent = PPO(cdfc_actor, cdfc_critic, actor_lr=cdfc_actor_lr, critic_lr=critic_lr, gamma=config['hmrd_ppo']['gamma'], clip_eps=config['hmrd_ppo']['clip_eps'])
                    cfc_agent = PPO(cfc_actor, cfc_critic, actor_lr=cfc_actor_lr, critic_lr=critic_lr, gamma=config['hmrd_ppo']['gamma'], clip_eps=config['hmrd_ppo']['clip_eps'])

                    cdfc_rewards_log = []
                    cfc_rewards_log = []
                    cdfc_avg_reward_log = []
                    cfc_avg_reward_log = []
                    cdfc_episodes_log = []
                    cfc_episodes_log = []
                    level_log = []
                    capture_rate_log = []
                    best_configs = []
                    cdfc_monitor = []
                    cfc_monitor = []
                    dump = []
                    dump_float = 0
                    how_many_100 = 0
                    last_updated = 0

                    cdfc_rewards, cfc_rewards, cdfc_scores, cfc_scores, dump_float = simulation.simulate()

                    for episode in range(config['hmrd_ppo']['episodes']):

                        flattened_cdfc_scores = flatten_list(cdfc_scores)
                        flatten_cfc_state = flatten_list(cfc_state)
                        current_state = flattened_cdfc_scores + flatten_cfc_state
                        state_tensor = torch.FloatTensor(current_state).to(device)


                        cdfc_state, cdfc_value, cdfc_monitor = cdfc_agent.select_action(state_tensor.flatten(), episode,
                                                                                            cdfc_monitor, i=-1, j=2)
                        cfc_state, cfc_value, cfc_monitor = cfc_agent.select_action(
                            state_tensor.flatten(), episode, cfc_monitor, i=-1, j=3)

                        value = cdfc_value
                        cdfc_state = [tuple(pair.tolist()) for pair in cdfc_state]
                        cdfc_state = [(x, y) for (x, y) in cdfc_state]

                        cfc_state = [tuple(pair.tolist()) for pair in cfc_state]
                        cfc_state = [(x, y, theta) for (x, y, theta) in cfc_state]


                        simulation.update_launch_pads(cdfc_state)
                        simulation.update_cameras(cfc_state)

                        cdfc_rewards, cfc_rewards, cdfc_scores, cfc_scores, capture_rate = simulation.simulate()

                        capture_rate_log.append(capture_rate)
                        penalty = compute_density_penalty(cdfc_state, min_dist=capture_radius - 10,
                                                             penalty_factor=1.5)
                        cdfc_rewards = cdfc_rewards - penalty

                        # Compute penalty
                        cfc_penalty = compute_detect_density(cfc_state, camera_range, 30, get_cameras_pointers)
                        cfc_rewards = cfc_rewards - cfc_penalty

                        total_cfc_rewards = cdfc_rewards + cfc_rewards

                        flattened_cdfc_scores = flatten_list(cdfc_scores)
                        flatten_cfc_state = flatten_list(cfc_state)
                        current_state = flattened_cdfc_scores + flatten_cfc_state
                        next_state_tensor = torch.FloatTensor(current_state).to(device)
                        cdfc_next_state_pred, next_value, dump = cdfc_agent.select_action(next_state_tensor.flatten(),
                                                                                            episode, dump, i=-1, j=2)
                        cfc_next_state_pred, cfc_next_value, dump = cfc_agent.select_action(
                            next_state_tensor.flatten(), episode, dump, i=-1, j=3)

                        cdfc_episodes_log.append(episode)
                        cfc_episodes_log.append(episode)
                        cdfc_rewards_log.append(cdfc_rewards)
                        cfc_rewards_log.append(cfc_rewards)
                        level_log.append(simulation.level)
                        cdfc_agent.train(state_tensor, next_state_tensor, cdfc_rewards, value, next_value)
                        cfc_agent.train(state_tensor, next_state_tensor, total_cfc_rewards, cfc_value,
                                           cfc_next_value)

                        if len(cdfc_rewards_log) >= 50:
                            cdfc_avg_reward = sum(cdfc_rewards_log[-50:]) / 50
                            how_many_100 = capture_rate_log.count(100.0)
                            if capture_rate == 100:
                                print(f'!!!!!   Interception 100%   !!!!!')
                                print(f'Number of 100% = {how_many_100}')
                            cdfc_avg_reward_log.append(cdfc_avg_reward)
                            print(
                                f"Episode {episode}, Reward: {cdfc_rewards}, Interception Rate: {capture_rate}, Avg Last 50: {cdfc_avg_reward} Level {simulation.level}")
                        else:
                            cdfc_avg_reward = sum(cdfc_rewards_log) / len(cdfc_rewards_log)
                            print(
                                f"Episode {episode}, Reward: {cdfc_rewards}, Interception Rate: {capture_rate}, Level {simulation.level}")
                            cdfc_avg_reward_log.append(cdfc_avg_reward)

                        if len(cfc_rewards_log) >= 50:
                            cfc_avg_reward = sum(cfc_rewards_log[-50:]) / 50
                            cfc_avg_reward_log.append(cfc_avg_reward)
                            print(
                                f"CFC Episode {episode}, CFC Reward: {cfc_rewards}, Avg Last 50: {cfc_avg_reward}")
                        else:
                            cfc_avg_reward = sum(cfc_rewards_log) / len(cfc_rewards_log)
                            print(f"CFC Episode {episode}, CFC Reward: {cfc_rewards}")
                            cfc_avg_reward_log.append(cfc_avg_reward)

                        best_configs = save_best_config(best_configs, cdfc_state, cfc_state, cdfc_rewards,
                                                           cdfc_avg_reward, simulation.level, episode=episode)

                        # Level Up Logic
                        if (((cdfc_avg_reward >= 1500 or how_many_100 > 50) and last_updated + 200 <= episode) or
                                (simulation.level >= 4 and (
                                        cdfc_avg_reward >= 1100 or how_many_100 > 10) and last_updated + 200 <= episode)):
                            save_models(cdfc_actor, cdfc_critic, cfc_actor, cfc_critic, simulation.level, results_dir=results_dir)
                            save_to_file(best_configs, simulation.level, results_dir=results_dir)
                            break

                        if (cdfc_avg_reward <= 1000 and how_many_100 < 10) and last_updated + 1000 <= episode:
                            how_many_100 = 0
                            simulation.leveldown()
                            simulation.levelup()
                            if simulation.init_level():
                                last_updated = episode
                                print("Level Reset!!!!")
                                cdfc_actor = CDFC_Actor(num_launch_pads * 3 + num_cameras * 3,
                                                          num_launch_pads * 2).to(device)
                                # cdfc_critic = Critic(num_launch_pads * 3 + num_cameras * 3).to(device)
                                cfc_actor = CFC_Actor(num_launch_pads * 3 + num_cameras * 3, num_cameras * 3).to(
                                    device)
                                # cfc_critic = Critic(num_launch_pads * 3 + num_cameras * 3).to(device)
                                cdfc_agent = PPO(cdfc_actor, cdfc_critic, actor_lr=cdfc_actor_lr, critic_lr=critic_lr, gamma=config['hmrd_ppo']['gamma'], clip_eps=config['hmrd_ppo']['clip_eps'])
                                cfc_agent = PPO(cfc_actor, cfc_critic, actor_lr=cfc_actor_lr,
                                                   critic_lr=critic_lr, gamma=config['hmrd_ppo']['gamma'], clip_eps=config['hmrd_ppo']['clip_eps'])
                            else:
                                break

                    # After training, save to JSON
                    with open(os.path.join(results_dir, f'cdfc_mean_std_log{simulation.level}.json'), 'w') as f:
                        json.dump(cdfc_monitor, f, indent=2, sort_keys=True)
                    with open(os.path.join(results_dir, f'cfc_mean_std_log{simulation.level}.json'), 'w') as f:
                        json.dump(cfc_monitor, f, indent=2, sort_keys=True)

                    # Save final best configurations to file
                    save_to_file(best_configs, simulation.level, results_dir=results_dir)
                    save_models(cdfc_actor, cdfc_critic, cfc_actor, cfc_critic, simulation.level, results_dir=results_dir)

                    # plot interception rewards
                    plt.figure(figsize=(10, 5))
                    plt.plot(cdfc_episodes_log, cdfc_avg_reward_log, label='Capture Avg Rewards',
                             color='b', linewidth=2)
                    plt.xlabel("Episodes")
                    plt.ylabel("Rewards")
                    plt.title(f"CDFC Lr = {cdfc_actor_lr}, CFC Lr = {cfc_actor_lr}, Crit Lr = {critic_lr}")
                    plt.legend()
                    plt.grid(True)

                    # Generate filename with learning rates
                    filename = os.path.join(results_dir, f"Level{simulation.level}_Train_cdfc.png")
                    plt.savefig(filename, dpi=300)
                    plt.close()  # Close plot to free memory

                    # plot detection rewards
                    plt.figure(figsize=(10, 5))
                    plt.plot(cfc_episodes_log, cfc_avg_reward_log, label='Detection Avg Rewards',
                             color='b', linewidth=2)
                    plt.xlabel("Episodes")
                    plt.ylabel("Rewards")
                    plt.title(f"CDFC Lr = {cdfc_actor_lr}, CFC Lr = {cfc_actor_lr}, Crit Lr = {critic_lr}")
                    plt.legend()
                    plt.grid(True)

                    # Generate filename with learning rates
                    filename = os.path.join(results_dir, f"Level{simulation.level}_Train_cfc.png")
                    plt.savefig(filename, dpi=300)
                    plt.close()  # Close plot to free memory

                    # Create level figure
                    plt.figure(figsize=(10, 5))
                    plt.plot(cdfc_episodes_log, level_log, label='Level', color='b', linewidth=2)
                    plt.xlabel("Episodes")
                    plt.ylabel("Level")
                    plt.title(f"CDFC Lr = {cdfc_actor_lr}, CFC Lr = {cfc_actor_lr}, Crit Lr = {critic_lr}")
                    plt.legend()
                    plt.grid(True)

                    # Generate filename with learning rates
                    filename = os.path.join(results_dir, f"Level{simulation.level}_levelTrain.png")
                    plt.savefig(filename, dpi=300)
                    plt.close()  # Close plot to free memory

if __name__ == '__main__':
    main()