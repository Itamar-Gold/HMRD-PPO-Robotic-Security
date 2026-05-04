import os
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def load_results(project_root):
    """Loads all JSON results from the results directory."""
    results_dir = os.path.join(project_root, "results")
    
    data = {
        "greedy": {},
        "human": {}
    }
    model_levels = [3, 5, 7, 9, 11]
    
    for ml in model_levels:
        for team in ["greedy", "human"]:
            model_name = f"model_BlueTeam_{team}_{ml}_results"
            folder_path = os.path.join(results_dir, model_name)
            file_path = os.path.join(folder_path, "performance_config0.json")
            
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = json.load(f)
                    
                config_data = content.get("config0", {})
                data[team][ml] = {}
                for level in range(1, 7):
                    level_key = f"Level_{level}"
                    if level_key in config_data:
                        data[team][ml][level] = {
                            "CR": config_data[level_key]["mean"],
                            "ST": config_data[level_key]["simulation mean time"]
                        }
            else:
                print(f"Warning: Results file not found for {team} model level {ml} at {file_path}")
            
    return data

def number_of_drones(mod_level):
    """Determines the number of Red Team drones based on the model difficulty level."""
    num_drones = 30
    if mod_level == 5:
        num_drones = 50
    if mod_level == 7:
        num_drones = 70
    if mod_level == 9:
        num_drones = 90
    if mod_level == 11:
        num_drones = 110
    return num_drones

def plot_level(data, save_dir, level_to_plot=1):
    """Generates the Level performance summary plots (CR and ST)."""
    
    # Get model levels that exist in both greedy and human data
    greedy_levels = set(data["greedy"].keys())
    human_levels = set(data["human"].keys())
    model_levels = sorted(list(greedy_levels.intersection(human_levels)))
    
    if not model_levels:
         model_levels = sorted(list(greedy_levels)) # fallback if no intersection
    
    drone_counts = [number_of_drones(ml) for ml in model_levels]
    
    cr_greedy = [data["greedy"][ml][level_to_plot]["CR"] if ml in data["greedy"] and level_to_plot in data["greedy"][ml] else 0 for ml in model_levels]
    st_greedy = [data["greedy"][ml][level_to_plot]["ST"] if ml in data["greedy"] and level_to_plot in data["greedy"][ml] else 0 for ml in model_levels]

    cr_human = [data["human"][ml][level_to_plot]["CR"] if ml in data["human"] and level_to_plot in data["human"][ml] else 0 for ml in model_levels]
    st_human = [data["human"][ml][level_to_plot]["ST"] if ml in data["human"] and level_to_plot in data["human"][ml] else 0 for ml in model_levels]

    x_pos = np.arange(len(drone_counts))
    width = 0.35

    # --- Plot 1: Mean Performance ---
    plt.figure(figsize=(8, 6))
    plt.bar(x_pos - width/2, cr_greedy, color='darkorange', width=width, label='Greedy')
    plt.bar(x_pos + width/2, cr_human, color='purple', width=width, label='Human')
    
    plt.title(f'Level {level_to_plot} - Capture Rate By Number Of Drones', fontsize=17)
    plt.xlabel('Number Of Red Drones', fontsize=16)
    plt.ylabel('Capture Rate [%]', fontsize=16)

    plt.xticks(x_pos, drone_counts)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='-', alpha=0.85)
    plt.tick_params(labelsize='large', width=2)
    
    cr_save_path = os.path.join(save_dir, f'greedy_vs_human_level{level_to_plot}_CR.png')
    plt.savefig(cr_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {cr_save_path}")
    plt.close()

    # --- Plot 2: Simulation Time ---
    plt.figure(figsize=(8, 6))
    plt.bar(x_pos - width/2, st_greedy, color='crimson', width=width, label='Greedy')
    plt.bar(x_pos + width/2, st_human, color='darkorchid', width=width, label='Human')

    plt.title(f'Level {level_to_plot} - Simulation Time By Number Of Drones', fontsize=17)
    plt.xlabel('Number Of Red Drones', fontsize=16)
    plt.ylabel('Simulation Time [sec]', fontsize=16)

    plt.xticks(x_pos, drone_counts)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='-', alpha=0.85)
    plt.tick_params(labelsize='large', width=2)
    
    st_save_path = os.path.join(save_dir, f'greedy_vs_human_level{level_to_plot}_ST.png')
    plt.savefig(st_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {st_save_path}")
    plt.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    save_dir = os.path.join(project_root, "results", "analysis_plots")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading data...")
    data = load_results(project_root)
    
    if not data["greedy"] and not data["human"]:
        print("No data loaded. Please ensure the evaluation script has run and saved results in /results.")
    else:
        print("Generating plots...")
        for level in [1,2,3,4,5,6]:
            plot_level(data, save_dir, level_to_plot=level)
        print(f"Analysis complete. Plots saved in {save_dir}")
