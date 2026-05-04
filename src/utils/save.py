import heapq
import json
import torch
import os


def save_best_config(best_configs, launch_pads, cameras, cdfc_reward, avg_reward, level, episode):
    combined_score = cdfc_reward + avg_reward

    # Create a config dictionary
    config = {
        "episode": episode,
        "level": level,
        "launch_pads": launch_pads,
        "cameras": cameras,
        "cdfc_reward": cdfc_reward,
        "avg_reward": avg_reward,
        "combined_score": combined_score
    }

    # Use a min-heap to maintain top 10 configurations
    if len(best_configs) < 10:
        heapq.heappush(best_configs, (combined_score, episode, config))
    else:
        heapq.heappushpop(best_configs, (combined_score, episode, config))

    return best_configs

def save_to_file(best_configs, level, results_dir="results/"):
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"best_configs{level}.json")
    with open(filename, "w") as f:
        json.dump([config for _, _, config in sorted(best_configs, reverse=True)], f, indent=4)

def save_models(cdfc_actor, cdfc_critic, cfc_actor, cfc_critic, level, results_dir="results/"):
    folder = os.path.join(results_dir, "saved_models")
    os.makedirs(folder, exist_ok=True)

    torch.save(cdfc_actor.state_dict(), f"{folder}/cdfc_actor_level_{level}.pth")
    torch.save(cdfc_critic.state_dict(), f"{folder}/cdfc_critic_level_{level}.pth")
    torch.save(cfc_actor.state_dict(), f"{folder}/cfc_actor_level_{level}.pth")
    torch.save(cfc_critic.state_dict(), f"{folder}/cfc_critic_level_{level}.pth")
    print(f"✅ Models saved for level {level}")