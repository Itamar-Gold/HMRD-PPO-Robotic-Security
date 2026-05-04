"""Evaluates Blue Team configurations across varying Red Team difficulty levels."""
import json
import time
import os
import yaml
import numpy as np

from src.simulation.engine import DroneSimulation

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


num_test = 50

# Dynamically find the project root from this file's location
current_dir = os.path.dirname(os.path.abspath(__file__))
# scripts -> project root
project_root = os.path.abspath(os.path.join(current_dir, ".."))
config_path = os.path.join(project_root, "config.yaml")


for model_level in [3, 5, 7, 9, 11]:
    model_num = f'BlueTeam_greedy_{model_level}'

    capture_rec_log = []
    camera_rec_log = []
    runs_record = []
    capture_rate_rec_log = []
    time_record = []

    for config in [0]:

        # Blue Team launch pad coordinates
        launch_pad_state = [(25, 38), (61, 40), (53, 62), (25, 14), (7, 68), (53, 6), (65, 66), (9, 8)]

        # Blue Team camera coordinates and orientations
        camera_state = [(25, 8, 90), (39, 16, 45), (37, 52, 20), (17, 36, 90), (21, 2, 135), (33, 8, 45), (65, 40, 150), (53, 50, 45)]

        # Initiate simulation
        simulation = DroneSimulation(config_path, launch_pad_state, camera_state)

        # Log performance metrics for each difficulty level
        performance_data = {f"config{config}": {}}

        for level in range(1, 7): # range(1, 7)
            simulation.level = level
            simulation.init_level()
            simulation.num_drones = number_of_drones(model_level)
            print(f"Model Number = {model_level} ----- Number of drones = {simulation.num_drones}")
            simulation.blue_drone_speed = 0.03
            capture_rate_rec_log = []
            time_record = []
            
            # Run simulation multiple times for statistical robustness
            for _ in range(num_test):
                start_time = time.perf_counter()
                capture_rate = simulation.test_simulate()
                simulation_time = time.perf_counter() - start_time
                time_record.append(simulation_time)
                capture_rate_rec_log.append(capture_rate)
                runs_record.append(_)
                print(f"Level: {level} -- Test Number {_} -- Capture rate: {capture_rate} -- Simulation Time: {simulation_time}")

            mean = np.mean(capture_rate_rec_log)
            std = np.std(capture_rate_rec_log)
            time_mean = np.mean(time_record)
            time_std = np.std(time_record)
            print(f"Level: {level} -- Capture Rate Mean: {mean} -- Capture Rate Std: {std} -- Configuration: {config}")
            # Update performance_data
            performance_data[f"config{config}"][f"Level_{level}"] = {
                "mean": float(mean),
                "std": float(std),
                "simulation mean time": float(time_mean),
                "simulation std time": float(time_std),
                "num_launch_pads": len(launch_pad_state),
                "num_cameras": len(camera_state),
                "launch_pad_state": launch_pad_state,
                "camera_state": camera_state
            }

        # Choose your save directory
        with open(config_path, "r") as f:
             conf = yaml.safe_load(f)
        
        # Resolve the results_dir absolutely using the project_root
        results_dir = os.path.join(project_root, conf["evaluation"]["results_dir"])
        save_dir = os.path.join(results_dir, f"model_{model_num}_results")
        os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

        # Create a filename
        filename = f"performance_config{config}.json"

        # Full path
        file_path = os.path.join(save_dir, filename)

        # Save performance_data
        with open(file_path, "w") as f:
            json.dump(performance_data, f, indent=4)

        print(f"Performance data saved to {file_path}")