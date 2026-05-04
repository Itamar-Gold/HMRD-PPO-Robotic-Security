"""Greedy heuristic algorithm for Set Covering Problem (SCP) deployment of Blue Team assets."""
import numpy as np
import pandas as pd
import random
import os
import yaml
from src.data.prep import load_roi
from src.utils.geometry import is_in_range


def assign_table(table, values, radius, shape=1, sweep_angle=60):
    """
    Populates a coverage matrix mapping generated points to secure points.
    Fills the table with 1s and 0s by checking whether each generated point is within range of each secure point.

    Parameters:
    - table (pd.DataFrame): DataFrame with secure points (launch pads or cameras) as index.
    - values (list of tuples): List of generated points.
    - radius (float): Detection range.
    - shape (int): 1 for launch pads, 2 for sector-based detection.

    Returns:
    - score (np.array): Score for each generated point (how many secure points it covers).
    - table (pd.DataFrame): Updated table with secured values.
    """
    temp = np.zeros((len(table.index),), dtype=int)
    score = np.zeros(len(values), dtype=int)  # Array of scores

    for pos, gen_point in enumerate(values):
        if shape == 2:
            x, y, camera_angle = gen_point  # Unpack (x, y, angle)
        else:
            x, y = gen_point
            camera_angle = 0  # No angle needed for circular detection

        for index, sec_point in enumerate(table.index):
            if shape == 2:
                temp[index] = is_in_range(sec_point, (x, y, camera_angle), radius, shape, camera_angle, sweep_angle)
            else:
                temp[index] = is_in_range(sec_point, (x, y), radius, shape, camera_angle, sweep_angle)

        if shape == 2:
            col_name = f"({int(gen_point[0])}, {int(gen_point[1])}, {int(camera_angle)})"
        else:
            col_name = f"({int(gen_point[0])}, {int(gen_point[1])})"
        table[col_name] = temp
        score[pos] = np.sum(temp)

    return score, table


def recursive_solver(
    table,
    shape=1,
    depth=0,
    max_lead_solutions=50,
    current_solution=None,
    min_score_threshold=5,
    max_low_score_solutions=2,
    random_seed=None
):
    """
    Recursively attempts to find the optimal minimal set of assets to cover the environment.
    Solves the set cover problem recursively to find complete solutions, prioritizing higher scores and adding randomness.

    Parameters:
        table (pd.DataFrame): The DataFrame representing the problem.
        depth (int): Depth of recursion, for debugging purposes.
        max_lead_solutions (int): Maximum number of solutions to find at each depth.
        current_solution (list): Current path of selected columns (coordinates).
        min_score_threshold (int): Minimum score to prioritize without limiting solutions.
        max_low_score_solutions (int): Maximum number of low-score solutions to explore.
        random_seed (int): Random seed for reproducibility (optional).

    Returns:
        solutions (list): List of complete solutions (sequences of coordinates).
    """
    if current_solution is None:
        current_solution = []

    # Set random seed for reproducibility if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Base case: If the table is empty, return the current solution as a valid solution
    if table.empty:
        print(f"Solution found at depth {depth}: {current_solution}")
        return [current_solution]

    # Step 1: Calculate scores for each column
    column_scores = table.sum(axis=0)

    # Step 2: Find columns with the highest scores
    max_score = column_scores.max()
    if pd.isna(max_score):  # Handle NaN case if the table is invalid
        print(f"Stopping recursion at depth {depth}: No valid scores in table.")
        return []

    # Identify top columns and their coordinates
    top_columns = column_scores[column_scores == max_score].index.tolist()
    lead_coordinates = [tuple(map(int, col.strip("()").split(", "))) for col in top_columns]

    # Limit solutions for lower scores and add randomness
    if max_score < min_score_threshold and len(lead_coordinates) > max_low_score_solutions:
        lead_coordinates = random.sample(lead_coordinates, max_low_score_solutions)

    print(f"Depth {depth}: Best score {max_score} with {len(lead_coordinates)} lead solutions.")
    print(f"Lead coordinates: {lead_coordinates}")

    # List to store all complete solutions
    complete_solutions = []

    # Step 3: Recurse for each lead solution
    for lead_coord in lead_coordinates:
        # Duplicate the table
        table_copy = table.copy()

        # Get the column name corresponding to the lead coordinate
        col_name = str(lead_coord)

        if col_name in table_copy.columns:
            # Drop the column and associated rows
            rows_to_drop = table_copy.index[table_copy[col_name] == 1].tolist()
            table_copy.drop(columns=[col_name], inplace=True)
            table_copy.drop(index=rows_to_drop, inplace=True)

        print(f"Depth {depth}: Generated table after removing {lead_coord}. Remaining size: {table_copy.shape}")

        # Recur on the modified table
        solutions_from_here = recursive_solver(
            table_copy,
            shape,
            depth + 1,
            max_lead_solutions,
            current_solution + [lead_coord],
            min_score_threshold,
            max_low_score_solutions,
            random_seed
        )

        # Add these solutions to the complete solutions list
        complete_solutions.extend(solutions_from_here)

        # Stop early if we have enough solutions
        if len(complete_solutions) >= max_lead_solutions:
            break

    # Return all complete solutions
    return complete_solutions

def find_placements(config, project_root):
    # Construct absolute paths for data files
    raw_roi_path = os.path.join(project_root, config['data']['raw_roi_path'])
    placement_path = os.path.join(project_root, config['data']['placement_path'])
    
    secure_points_sorted = load_roi(raw_roi_path)
    
    # Define the range for x and y
    # x_min, x_max = config['simulation']['x_map_lim'][0] + 5, config['simulation']['x_map_lim'][0] - 5  # Incorrect
    x_min, x_max = config['simulation']['x_map_lim'][0] + 5, config['simulation']['x_map_lim'][1] - 5  # Corrected Range for x
    y_min, y_max = config['simulation']['y_map_lim'][0], config['simulation']['border_y']              # Range for y
    
    # Define the grid resolution (step size)
    x_step, y_step = 2, 2  # Step size for x and y
    
    # Generate the grid points
    x = np.arange(x_min, x_max + x_step, x_step)
    y = np.arange(y_min, y_max + y_step, y_step)
    
    # Combine grid points into a single array of shape (N, 2)
    grid_points = [(round(xi, 2), round(yi, 2)) for xi in x for yi in y]
    
    # Ensure unique points only
    sorted_points_formatted = list(set(grid_points))
    
    # Sort the points for consistent formatting
    sorted_points_formatted = sorted(sorted_points_formatted, key=lambda p: (p[0], p[1]))
    
    # Create DataFrame for the detection
    camera_range = config['simulation']['camera_range']
    angles = [20, 45, 60, 90, 120, 135, 150]  # List of possible angles
    
    # Create a new list with (x, y, angle) for each point
    grid_points_sorted_with_angles = [(x, y, angle) for (x, y) in sorted_points_formatted for angle in angles]
    
    # Write the launch pads and camera placements to excel
    excel_writer = pd.ExcelWriter(placement_path, engine="xlsxwriter")
    
    # Update `original_df_detection` to use this new list
    original_df_detection = pd.DataFrame(index=secure_points_sorted)
    
    # Create an empty DataFrame
    original_df_launch_pads = pd.DataFrame(index=secure_points_sorted)
    
    score_launch_pads, original_df_launch_pads  = assign_table(original_df_launch_pads, sorted_points_formatted,
                                                               radius=20, shape=1, sweep_angle=config['simulation']['camera_sweep_angle'])
    score_detection, original_df_detection  = assign_table(original_df_detection, grid_points_sorted_with_angles,
                                                           radius=camera_range, shape=2, sweep_angle=config['simulation']['camera_sweep_angle'])
    
    launch_pads_solutions = recursive_solver(
        original_df_launch_pads,
        depth=0,
        max_lead_solutions=150,
        min_score_threshold=5,
        max_low_score_solutions=2
    )
    
    detection_solutions = recursive_solver(
        original_df_detection,
        shape=2,
        depth=0,
        max_lead_solutions=300,
        min_score_threshold=10,
        max_low_score_solutions=1
    )
    placements_df = pd.DataFrame(columns=["Launch Pads", "Cameras", "Number of Launch Pads", "Number of Cameras"])
    
    if not detection_solutions:
        print("No valid detection solutions found.")
        return
        
    detection_min_sol = detection_solutions[0]
    for sol in detection_solutions:
        if len(sol) < len(detection_min_sol):
            detection_min_sol = sol
    print(f'Best solution is: {detection_min_sol}, \nwith {len(detection_min_sol)} Cameras')
    
    if not launch_pads_solutions:
        print("No valid launch pads solutions found.")
        return
        
    lp_min_sol = launch_pads_solutions[0]
    for sol in launch_pads_solutions:
        if len(sol) < len(lp_min_sol):
            lp_min_sol = sol
    print(f'Best solution is: {lp_min_sol}, \nwith {len(lp_min_sol)} Launch Pads')
    
    # History assign
    placements_df.loc[len(placements_df)] = [
        lp_min_sol,
        detection_min_sol,
        len(lp_min_sol),
        len(detection_min_sol)
    ]
    # Save each run as a separate sheet
    placements_df.to_excel(excel_writer, sheet_name="Placements", index=False)
    excel_writer.close()

if __name__ == "__main__":
    # Dynamically find the project root from this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src/algorithms -> src -> project root
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    find_placements(config, project_root)