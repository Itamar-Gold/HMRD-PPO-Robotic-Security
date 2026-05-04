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
    
    # Store data as: data[model_level][simulation_level] = {"mean": x, "time": y}
    data = {}
    
    # Expected models to look for based on our evaluation script
    model_levels = [3, 5, 7, 9, 11]
    
    for ml in model_levels:
        model_name = f"model_BlueTeam_greedy_{ml}_results"
        folder_path = os.path.join(results_dir, model_name)
        file_path = os.path.join(folder_path, "performance_config0.json")
        
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = json.load(f)
                
            config_data = content.get("config0", {})
            data[ml] = {}
            for level in range(1, 7):
                level_key = f"Level_{level}"
                if level_key in config_data:
                    data[ml][level] = {
                        "CR": config_data[level_key]["mean"],
                        "ST": config_data[level_key]["simulation mean time"]
                    }
        else:
            print(f"Warning: Results file not found for model level {ml} at {file_path}")
            
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

def get_optimal_bounds(matrix, buffer_factor=0.05):
    """
    Calculates optimized minimum and maximum bounds for the colormap based on the data.
    Adds a small buffer to the min and max to ensure colors are visible.
    """
    min_val = np.nanmin(matrix)
    max_val = np.nanmax(matrix)
    
    # Add buffer
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1 # Prevent division by zero if all values are the same
        
    vmin = max(0, min_val - (range_val * buffer_factor))
    vmax = max_val + (range_val * buffer_factor)
    
    return vmin, vmax

def plot_performance_matrix(data, save_dir):
    """Plots Heatmap Performance Matrices for Capture Rate and Simulation Time."""
    levels = [1, 2, 3, 4, 5, 6]
    model_levels = sorted(data.keys())
    drone_counts = [number_of_drones(ml) for ml in model_levels]
    
    # Initialize matrices
    # Rows: Levels (from 1 to 6) -> Index 0 is Level 1
    # Cols: Drones
    cr_matrix = np.zeros((len(levels), len(drone_counts)))
    st_matrix = np.zeros((len(levels), len(drone_counts)))
    
    for i, lvl in enumerate(levels):
        for j, ml in enumerate(model_levels):
            cr_matrix[i, j] = data.get(ml, {}).get(lvl, {}).get("CR", np.nan)
            st_matrix[i, j] = data.get(ml, {}).get(lvl, {}).get("ST", np.nan)
            
    # Calculate optimized bounds for colormaps
    cr_vmin, cr_vmax = get_optimal_bounds(cr_matrix)
    st_vmin, st_vmax = get_optimal_bounds(st_matrix)

    # --- 1. Plot Capture Rate (CR) Performance Matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use RdYlGn: Red for lowest, Green for highest
    cmap_cr = plt.cm.RdYlGn
    cax1 = ax.imshow(cr_matrix, origin='lower', cmap=cmap_cr, vmin=cr_vmin, vmax=cr_vmax, aspect='auto')
    
    # Set axis ticks and labels
    ax.set_xticks(np.arange(len(drone_counts)))
    ax.set_yticks(np.arange(len(levels)))
    ax.set_xticklabels(drone_counts, fontsize=12)
    ax.set_yticklabels(levels, fontsize=12)
    
    ax.set_xlabel('Number of Red Drones', fontsize=14)
    ax.set_ylabel('Simulation Level', fontsize=14)
    ax.set_title('Performance Matrix: Capture Rate [%]', fontsize=16)
    
    # Annotate cells with numerical values
    mid_cr = (cr_vmin + cr_vmax) / 2
    for i in range(len(levels)):
        for j in range(len(drone_counts)):
            val = cr_matrix[i, j]
            if not np.isnan(val):
                # Text color logic based on intensity of background
                text_color = "white" if abs(val - mid_cr) > (cr_vmax - cr_vmin) * 0.25 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=13, fontweight='bold')
                
    # Add Colorbar
    cbar1 = fig.colorbar(cax1, ax=ax, pad=0.02)
    cbar1.set_label('Capture Rate [%]', fontsize=13)
    
    plt.tight_layout()
    cr_save_path = os.path.join(save_dir, f'heatmap_matrix_CR_greedy.png')
    plt.savefig(cr_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {cr_save_path}")
    plt.close()

    # --- 2. Plot Simulation Time (ST) Performance Matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # For ST, lower time is better. 
    # Use RdYlGn_r (reversed): Red for highest time (worst), Green for lowest time (best).
    cmap_st = plt.cm.RdYlGn_r
    
    cax2 = ax.imshow(st_matrix, origin='lower', cmap=cmap_st, vmin=st_vmin, vmax=st_vmax, aspect='auto')
    
    # Set axis ticks and labels
    ax.set_xticks(np.arange(len(drone_counts)))
    ax.set_yticks(np.arange(len(levels)))
    ax.set_xticklabels(drone_counts, fontsize=12)
    ax.set_yticklabels(levels, fontsize=12)
    
    ax.set_xlabel('Number of Red Drones', fontsize=14)
    ax.set_ylabel('Simulation Level', fontsize=14)
    ax.set_title('Performance Matrix: Simulation Time [s]', fontsize=16)
    
    # Annotate cells with numerical values
    mid_st = (st_vmin + st_vmax) / 2
    for i in range(len(levels)):
        for j in range(len(drone_counts)):
            val = st_matrix[i, j]
            if not np.isnan(val):
                # Text color logic based on intensity of background
                text_color = "white" if abs(val - mid_st) > (st_vmax - st_vmin) * 0.25 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=13, fontweight='bold')
                
    # Add Colorbar
    cbar2 = fig.colorbar(cax2, ax=ax, pad=0.02)
    cbar2.set_label('Simulation Time [s]', fontsize=13)
    
    plt.tight_layout()
    st_save_path = os.path.join(save_dir, 'heatmap_matrix_ST_greedy.png')
    plt.savefig(st_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {st_save_path}")
    plt.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    save_dir = os.path.join(project_root, "results", "analysis_plots")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading data for Heatmap generation...")
    data = load_results(project_root)
    
    if not data:
        print("No data loaded. Please ensure the evaluation script has run and saved results in /results.")
    else:
        print("Generating Performance Matrices (Heatmaps)...")
        plot_performance_matrix(data, save_dir)
        print(f"Analysis complete. Heatmaps saved in {save_dir}")
