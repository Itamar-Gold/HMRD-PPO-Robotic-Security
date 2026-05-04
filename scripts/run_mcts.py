import os
import sys
import pandas as pd
import multiprocessing
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.algorithms.mcts import run_mcts
from src.simulation.engine import DroneSimulation

_worker_sim_instance = None

def init_worker(config_path):
    global _worker_sim_instance
    _worker_sim_instance = DroneSimulation(config_path, [], [])

def run_single_simulation(args):
    global _worker_sim_instance
    cdfc_state_list, fixed_cameras, sim_level, instance_penalty = args
    
    _worker_sim_instance.update_launch_pads(cdfc_state_list)
    _worker_sim_instance.update_cameras(fixed_cameras)
    _worker_sim_instance.level = sim_level
    _worker_sim_instance.init_level()
        
    cdfc_rewards, _, _, _, _ = _worker_sim_instance.simulate()
        
    # Calculate penalty based on the number of deployed launch pads
    total_penalty = len(cdfc_state_list) * instance_penalty
    
    # Maximize sum of rewards minus the deployment penalty
    return cdfc_rewards - total_penalty

class BatchEvaluator:
    def __init__(self, config_path, fixed_cameras, sim_levels, instance_penalty):
        self.config_path = config_path
        self.fixed_cameras = fixed_cameras
        self.sim_levels = sim_levels
        self.instance_penalty = instance_penalty
        
        self.total_cores = multiprocessing.cpu_count()
        self.num_processes = max(1, self.total_cores - 4)
        
        # Initialize pool once
        self.pool = multiprocessing.Pool(
            processes=self.num_processes,
            initializer=init_worker,
            initargs=(self.config_path,)
        )
        
    def __call__(self, states):
        worker_args = []
        num_levels = len(self.sim_levels)
        
        # Distribute states across different levels in a round-robin fashion
        for i, s in enumerate(states):
            level = self.sim_levels[i % num_levels]
            worker_args.append((
                list(s), self.fixed_cameras, level, self.instance_penalty
            ))
            
        if not worker_args:
            return []
            
        # Run exactly len(states) simulations total. 
        # For example, if there are 20 states and 5 levels, there will be exactly 20 calls to run_single_simulation
        # (4 calls for each level).
        rewards = self.pool.map(run_single_simulation, worker_args)
        
        return rewards
        
    def close(self):
        self.pool.close()
        self.pool.join()

def main():
    print("Initializing MCTS Spatial Optimization Script...")
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    excel_file = os.path.join(os.path.dirname(__file__), '..', config['data']['placement_path'])
    
    interdetec_df = pd.read_excel(os.path.abspath(excel_file), sheet_name="Placements")
    launch_pads = interdetec_df["Launch Pads"].apply(
        lambda x: eval(x) if isinstance(x, str) else x).tolist()
    cameras = interdetec_df["Cameras"].apply(
        lambda x: eval(x) if isinstance(x, str) else x).tolist()

    if isinstance(launch_pads[0], list): launch_pads = launch_pads[0]
    if isinstance(cameras[0], list): cameras = cameras[0]
            
    grid_step = config['mcts']['grid_step']
    U = set()
    for x in range(config['simulation']['x_map_lim'][0], config['simulation']['x_map_lim'][1] + 5, grid_step):
        for y in range(config['simulation']['y_map_lim'][0], config['simulation']['y_map_lim'][1] - 40, grid_step): # up to 75
            U.add((x, y))
            
    # Snap initial launch pads to grid to form canonical set
    snapped_launch_pads = set()
    for x, y in launch_pads:
        sx = round(x / grid_step) * grid_step
        sy = round(y / grid_step) * grid_step
        sx = max(config['simulation']['x_map_lim'][0], min(config['simulation']['x_map_lim'][1], sx))
        sy = max(config['simulation']['y_map_lim'][0], min(config['simulation']['border_y'], sy))
        
        # Ensure distinct grid positions
        if (sx, sy) in snapped_launch_pads:
            placed = False
            for dx in [-grid_step, 0, grid_step]:
                for dy in [-grid_step, 0, grid_step]:
                    cand = (sx+dx, sy+dy)
                    if cand in U and cand not in snapped_launch_pads:
                        snapped_launch_pads.add(cand)
                        placed = True
                        break
                if placed: break
        else:
            snapped_launch_pads.add((sx, sy))
            
    initial_state = frozenset(snapped_launch_pads)
    k = len(initial_state)
    
    # MCTS Configuration
    P = max(1, multiprocessing.cpu_count() - 4) # Batch size based on available cores
    C = config['mcts']['C'] # UCT exploration constant (normalized rewards [0,1], so C=1 is robust)
    H = config['mcts']['H']
    
    density_radius = grid_step * config['mcts']['density_radius_multiplier']  # Mask out immediately adjacent cells
    d_max = grid_step * config['mcts']['d_max_multiplier']           # Allow moving up to d_max cells away
    iterations = config['mcts']['iterations']
    
    # Simulation Configuration
    sim_levels = config['mcts']['sim_level']
    if not isinstance(sim_levels, list):
        sim_levels = [sim_levels]
    
    # Penalty assigned to each deployed launch pad to encourage fewer instances
    instance_penalty = config['mcts']['instance_penalty']
    
    evaluate_batch = BatchEvaluator(config_path, cameras, sim_levels, instance_penalty)
    
    print(f"Starting MCTS with {iterations} iterations, Batch size P={P}, Grid step={grid_step}")
    print(f"Evaluating over levels: {sim_levels}")
    print(f"Initial Canonical State: {list(initial_state)}")
    
    best_state, best_reward, T = run_mcts(U, k, d_max, H, P, C, iterations, evaluate_batch, initial_state, density_radius, config['mcts']['q_min'], config['mcts']['q_max'])
    
    evaluate_batch.close()
    
    print("\n--- MCTS OPTIMIZATION COMPLETE ---")
    print(f"Best Found Launch Pads Configuration (Size {len(best_state)}): {list(best_state)}")
    print(f"Highest Associated Average Reward: {best_reward}")

if __name__ == '__main__':
    main()