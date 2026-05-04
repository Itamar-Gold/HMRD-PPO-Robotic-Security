"""Core simulation engine for the pursuit-evasion scenarios between Blue Team and Red Team."""
import matplotlib.pyplot as plt
import math
import os
import random
import yaml

from src.simulation.physics import Drone, set_score, pure_pursuit_target, proportional_navigation, assign_blue_drone
from src.algorithms.aaa_star import generate_drop_zones, aaa_star_drone
from src.utils.geometry import is_in_range, fast_euclid_distance

# Set seeds for stability
SEED = 42
random.seed(SEED)


class DroneSimulation:
    """Simulation class for handling dynamic adversarial scenarios."""

    def __init__(self, config_path, launch_pads, cameras):
        """Initializes simulation constants, Blue Team assets, and Red Team parameters from config."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        sim_config = self.config["simulation"]
        
        # Constant Definitions
        self.time_step = sim_config["time_step"]
        self.red_drone_speed = sim_config["red_drone_speed"]
        self.blue_drone_speed = sim_config["blue_drone_speed"]
        self.launch_delay = sim_config["launch_delay"]
        self.N = sim_config["proportional_nav_N"]
        self.border_y = sim_config["border_y"]
        self.camera_range = sim_config["camera_range"]
        self.x_map_lim = tuple(sim_config["x_map_lim"])
        self.y_map_lim = tuple(sim_config["y_map_lim"])
        self.capture_predict_steps = self.launch_delay * self.time_step
        
        # aaa* parameters
        self.aaa_step_size = sim_config["aaa_star"]["grid_step_size"]
        self.aaa_penalty_radius = sim_config["aaa_star"]["penalty_radius"]
        self.aaa_penalty_weight = sim_config["aaa_star"]["penalty_weight"]
        self.aaa_stretch_factor = sim_config["aaa_star"]["stretch_factor"]
        
        # pure pursuit parameters
        self.pp_lookahead = sim_config["pure_pursuit"]["lookahead_dist"]

        self.launch_pads = launch_pads
        self.cameras = cameras
        self.all_targets = sim_config["selected_targets"]

        # Dynamic Definitions to change in different levels
        self.level = 1
        self._apply_level_config()

    def _apply_level_config(self):
        """Applies level-specific dynamic settings from the config based on self.level."""
        sim_config = self.config["simulation"]
        self.num_drones = sim_config["num_drones"]
        
        # Match level details from YAML
        level_data = next((l for l in sim_config["levels"] if l["id"] == self.level), None)
        if level_data:
            self.adaptability = level_data["adaptability"]
            self.start_pos = [tuple(pos) for pos in level_data["start_pos"]]
            
            if level_data["target_slice"] == "all":
                self.target_coords = self.all_targets
            else:
                self.target_coords = self.all_targets[:level_data["target_slice"]]
            # print(f"Initialized Level {self.level}: {self.adaptability} adaptability, {self.num_drones} Drones and {len(self.target_coords)} targets.")
        else:
            if self.level == 0:
                print("Learning Failed")
            elif self.level == 7:
                print("Learning succeed !!")
            else:
                print("Level Config Missing.")


    def levelup(self):
        """Increments the difficulty level of the simulation."""
        self.level += 1
        return self.init_level()

    def leveldown(self):
        """Decrements the difficulty level of the simulation."""
        self.level -= 1
        return self.init_level()
        
    def init_level(self):
        self._apply_level_config()
        if self.level == 0 or self.level == 7:
             return 0
        return 1

    def update_launch_pads(self, launch_pads):
        """Updates the locations of the Blue Team launch pads."""
        self.launch_pads = launch_pads

    def update_cameras(self, cameras):
        """Updates the locations and angles of the Blue Team cameras."""
        self.cameras = cameras

    def simulate(self):
        """Executes a headless simulation loop for performance evaluation."""
        # The level of the simulation is set based on the following:
        #       Number of drones = [20, 100]
        #       Number of targets for the red drone
        #       Initial positions of the attackers [(0 -> 80, 115)]
        #       AAA* with or without adaptability
        captured_points = []
        target_hits = 0
        capture_hits = 0
        inter_reward = 0
        detect_reward = 0
        # Initialize score list for the launch_pads with default values (e.g., 0 for each coordinate)
        capture_list = []
        detect_list = []
        for coord in self.launch_pads:
            capture_list.append(coord)
            capture_list.append(0)
        for coord in self.cameras:
            detect_list.append(coord)
            detect_list.append(0)

        for drone_id in range(self.num_drones):
            start = (random.choice(self.start_pos))  # Start above the defense border
            if "no" in self.adaptability:
                captured_points = []
            if "partial" in self.adaptability:  # Partial adaptability to guide the learning
                if drone_id % 10 == 0:
                    captured_points = []

            # Initialize the red drone
            red_drone = Drone(position=start, velocity=self.red_drone_speed, angle=math.radians(0))
            # Create a copy of the red drone to monitor last position in case of no visibility
            last_seen_drone = Drone(position=red_drone.position, velocity=red_drone.velocity, angle=red_drone.angle)
            target = (random.choice(self.target_coords))
            if captured_points:
                drop_zones = generate_drop_zones(captured_points, radius=self.aaa_penalty_radius, weight=self.aaa_penalty_weight, stretch_factor=self.aaa_stretch_factor,
                                                      step_size=self.aaa_step_size)
                path = aaa_star_drone(start, target, self.x_map_lim, self.y_map_lim, self.aaa_step_size, drop_zones)
            else:
                path = aaa_star_drone(start, target, self.x_map_lim, self.y_map_lim, self.aaa_step_size, captured_points)

            if not path:
                print("Path Was not found")
                continue  # Skip if no valid path found

            # Reset Variables
            vision = False
            blue_drone_assign = False
            last_index = 0  # Start tracking from first path point
            delay = 0

            # Start the flight
            for step in range(1000000):  # Allow enough iterations for the drone to reach the target
                # start follow the path
                last_index = pure_pursuit_target(red_drone, path, lookahead_dist=self.pp_lookahead, time_step=self.time_step,
                                                 last_index=last_index)
                # detection scan
                for detection in self.cameras:
                    if is_in_range(red_drone.position, detection, self.camera_range, shape=2, camera_angle=detection[2]):
                        vision = True
                        set_score(detect_list, detection)
                        # save "last seen" data for when the red drone left the detection area
                        last_seen_drone.position = red_drone.position
                        last_seen_drone.angle = red_drone.angle
                        detect_reward += 1
                        break
                    else:
                        vision = False
                        if red_drone.position[1] < self.border_y and self.cameras[-1] == detection:
                            detect_reward -= 1

                if vision and not blue_drone_assign:  # If the red drone is in sight of a camera and
                    # an intercepted has not assigned yet
                    inter_reward += 10
                    blue_drone_assign, blue_drone, assigned_launch_pad = assign_blue_drone(self.launch_pads, red_drone,
                                                                                              self.blue_drone_speed,
                                                                                              self.capture_predict_steps)

                if vision and blue_drone_assign:
                    # if the red drone is in sight and a blue drone is assigned
                    # Apply launch delay
                    delay += 1
                    if delay > self.launch_delay:
                        # After delay start moving towards the red drone
                        proportional_navigation(blue_drone, red_drone, self.time_step, self.N)

                if blue_drone_assign and not vision:  # If a blue drone is assigned and there is no vision
                    delay += 1
                    if delay > self.launch_delay:
                        # Onboard vision
                        # If the range between the blue drone and red drone is less than 4m -> apply vision
                        if fast_euclid_distance(red_drone.position, blue_drone.position) <= (0.04 ** 2):
                            proportional_navigation(blue_drone, red_drone, self.time_step, self.N)
                        else:
                            proportional_navigation(blue_drone, last_seen_drone, self.time_step, self.N)

                # Target Hit!
                if fast_euclid_distance(red_drone.position, target) <= (0.01 ** 2): # 1 m from the target
                    target_hits += 1
                    inter_reward -= 40
                    del red_drone
                    del last_seen_drone
                    if blue_drone_assign:
                        del blue_drone
                    else:
                        inter_reward -= 100
                    break
                # Capture!
                if blue_drone_assign:
                    if fast_euclid_distance(red_drone.position, blue_drone.position) <= (0.01 ** 2): # 1m from the target
                        capture_hits += 1
                        inter_reward += 50
                        set_score(capture_list, assigned_launch_pad)
                        captured_points.append(red_drone.position)
                        del red_drone
                        del last_seen_drone
                        del blue_drone
                        break

        capture_rate = capture_hits / self.num_drones
        capture_rate = capture_rate * 100
        inter_reward = inter_reward + capture_rate
        return round(inter_reward, 1), round(detect_reward, 1), capture_list, detect_list, round(capture_rate, 1)

    def test_simulate(self):
        """Executes a validation simulation loop focusing only on capture rates."""
        captured_points = []
        target_hits = 0
        capture_hits = 0
        # Initialize score list for the launch_pads with default values (e.g., 0 for each coordinate)
        capture_list = []
        detect_list = []
        for coord in self.launch_pads:
            capture_list.append(coord)
            capture_list.append(0)
        for coord in self.cameras:
            detect_list.append(coord)
            detect_list.append(0)

        for drone_id in range(self.num_drones):
            start = (random.choice(self.start_pos))  # Start above the defense border
            if "no" in self.adaptability:
                captured_points = []
            if "partial" in self.adaptability:
                if drone_id % 10 == 0:
                    captured_points = []

            # Initialize the red drone
            red_drone = Drone(position=start, velocity=self.red_drone_speed, angle=math.radians(0))  # Moving at 270°
            # Create a copy of the red drone to monitor last position in case of no visibility
            last_seen_drone = Drone(position=red_drone.position, velocity=red_drone.velocity, angle=red_drone.angle)
            target = (random.choice(self.target_coords))
            if captured_points:
                drop_zones = generate_drop_zones(captured_points, radius=self.aaa_penalty_radius, weight=self.aaa_penalty_weight, stretch_factor=self.aaa_stretch_factor,
                                                      step_size=self.aaa_step_size)
                path = aaa_star_drone(start, target, self.x_map_lim, self.y_map_lim, self.aaa_step_size, drop_zones)
            else:
                path = aaa_star_drone(start, target, self.x_map_lim, self.y_map_lim, self.aaa_step_size, captured_points)

            if not path:
                print("Path Was not found")
                continue  # Skip if no valid path found

            # Reset Variables
            vision = False
            blue_drone_assign = False
            last_index = 0  # Start tracking from first path point
            delay = 0

            # Start the attack
            for step in range(1000000):  # Allow enough iterations for the drone to reach the end
                # start follow the path of attack
                last_index = pure_pursuit_target(red_drone, path, lookahead_dist=self.pp_lookahead, time_step=self.time_step,
                                                 last_index=last_index)

                for detection in self.cameras:
                    if is_in_range(red_drone.position, detection, self.camera_range, shape=2, camera_angle=detection[2]):
                        vision = True
                        set_score(detect_list, detection)
                        last_seen_drone.position = red_drone.position
                        last_seen_drone.angle = red_drone.angle
                        break
                    else:
                        vision = False

                if vision and not blue_drone_assign:  # If the red drone is in sight of a camera and
                    # an intercepted has not assigned yet
                    blue_drone_assign, blue_drone, assigned_launch_pad = assign_blue_drone(self.launch_pads, red_drone,
                                                                                              self.blue_drone_speed,
                                                                                              self.capture_predict_steps)

                if vision and blue_drone_assign:
                    # if the red drone is in sight and a blue drone is assigned
                    # Apply launch delay
                    delay += 1
                    if delay > self.launch_delay:
                        # After delay start moving towards the red drone
                        proportional_navigation(blue_drone, red_drone, self.time_step, self.N)

                if blue_drone_assign and not vision:  # If a blue drone is assigned and there is no vision
                    delay += 1
                    if delay > self.launch_delay:

                        if fast_euclid_distance(red_drone.position, blue_drone.position) <= (0.04 ** 2):
                            proportional_navigation(blue_drone, red_drone, self.time_step, self.N)
                        else:
                            proportional_navigation(blue_drone, last_seen_drone, self.time_step, self.N)

                # Target Hit!
                if fast_euclid_distance(red_drone.position, target) <= (0.01 ** 2):
                    target_hits += 1
                    del red_drone
                    del last_seen_drone
                    if blue_drone_assign:
                        del blue_drone
                    break
                # Capture!
                if blue_drone_assign:
                    if fast_euclid_distance(red_drone.position, blue_drone.position) <= (0.01 ** 2):
                        capture_hits += 1
                        set_score(capture_list, assigned_launch_pad)
                        captured_points.append(red_drone.position)
                        del red_drone
                        del last_seen_drone
                        del blue_drone
                        break

        capture_rate = capture_hits / self.num_drones
        return capture_rate * 100


    def _render_frame(self, frame_counter, animation_dir, episode, drop_zones, target, start, red_drone, paths, drone_id, capture_hits, target_hits, blue_drone_assign=False, blue_drone=None, assigned_launch_pad=None):
        """Helper method to render and save a single simulation frame."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        
        plt.figure(figsize=(8, 6))
        if drop_zones:
            for point in drop_zones:
                plt.plot(point[0], point[1], 'r.') 
        plt.plot(target[0], target[1], 'rX')
        plt.plot(start[0], start[1], 'k1')
        plt.plot(red_drone.position[0], red_drone.position[1], 'rx', label='Red Drone', markersize=10)
        
        if paths:
            path_x, path_y = zip(*paths[-1])
            plt.plot(path_x, path_y, '--k')
            
        plt.plot([0, 80], [70, 70], '-r')
        for launch_pad_id in self.launch_pads:
            plt.plot(launch_pad_id[0], launch_pad_id[1], 'ko')
            
        if blue_drone_assign and blue_drone:
            plt.plot(blue_drone.position[0], blue_drone.position[1], 'bp', label='Blue Drone', markersize=10)
            plt.plot(assigned_launch_pad[0], assigned_launch_pad[1], 'bo')

        plt.xlim([-5, 85])
        plt.ylim([-5, 120])
        title_line2 = f"Drone: {drone_id:<10} Captured: {capture_hits:<10} Missed: {target_hits:<10}"
        plt.suptitle(f"Level {self.level}", fontsize=16, fontweight='bold')
        plt.title(title_line2, fontsize=12, loc='center')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{animation_dir}/episode{episode}/frame{frame_counter}.png', dpi=200)
        plt.close()

    def animate_simulate(self, episode):
        """Executes the simulation and generates frame-by-frame visualizations."""
        import matplotlib
        from matplotlib.patches import Wedge

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        
        captured_points = []
        drop_zones = []
        paths = []
        target_hits = 0
        capture_hits = 0
        # Initialize score list for the launch_pads with default values (e.g., 0 for each coordinate)
        capture_list = []
        detect_list = []
        total_steps = 0
        frame_counter = 0
        
        animation_dir = self.config["evaluation"]["animation_dir"]
        os.makedirs(f'{animation_dir}/episode{episode}', exist_ok=True)  # Create the folder if it doesn't exist

        for coord in self.launch_pads:
            capture_list.append(coord)
            capture_list.append(0)
        for coord in self.cameras:
            detect_list.append(coord)
            detect_list.append(0)

        for drone_id in range(self.num_drones):
            start = (random.choice(self.start_pos))  # Start above the defense border
            if "no" in self.adaptability:
                captured_points = []
            if "partial" in self.adaptability:
                if drone_id % 10 == 0:
                    captured_points = []
            red_drone = Drone(position=start, velocity=self.red_drone_speed,
                                   angle=math.radians(0))  # Moving at 270°
            # Create a copy of the red drone to monitor last position in case of no visibility
            last_seen_drone = Drone(position=red_drone.position, velocity=red_drone.velocity, angle=red_drone.angle)
            target = (random.choice(self.target_coords))
            if captured_points:
                drop_zones = generate_drop_zones(captured_points, radius=self.aaa_penalty_radius, weight=self.aaa_penalty_weight, stretch_factor=self.aaa_stretch_factor,
                                                      step_size=self.aaa_step_size)
                path = aaa_star_drone(start, target, self.x_map_lim, self.y_map_lim, self.aaa_step_size, drop_zones)
                paths.append(path)
                path_x, path_y = zip(*path)
            else:
                path = aaa_star_drone(start, target, self.x_map_lim, self.y_map_lim, self.aaa_step_size, captured_points)
                paths.append(path)
                path_x, path_y = zip(*path)

            if not path:
                print("Path Was not found")
                continue  # Skip if no valid path found

            # Reset Variables
            vision = False
            blue_drone_assign = False
            last_index = 0  # Start tracking from first path point
            delay = 0

            # Start the attack
            for step in range(1000000):  # Allow enough iterations for the drone to reach the end
                # start follow the path of attack
                last_index = pure_pursuit_target(red_drone, path, lookahead_dist=self.pp_lookahead, time_step=self.time_step,
                                                      last_index=last_index)
                # plot every 200 steps
                if step % 300 == 0:
                    frame_counter += 1
                    self._render_frame(frame_counter, animation_dir, episode, drop_zones, target, start, red_drone, paths, drone_id, capture_hits, target_hits, blue_drone_assign, blue_drone if blue_drone_assign else None, assigned_launch_pad if blue_drone_assign else None)

                for detection in self.cameras:
                    if is_in_range(red_drone.position, detection, self.camera_range, shape=2,
                                        camera_angle=detection[2]):
                        vision = True
                        set_score(detect_list, detection)
                        last_seen_drone.position = red_drone.position
                        last_seen_drone.angle = red_drone.angle
                        break
                    else:
                        vision = False

                if vision and not blue_drone_assign:  # If the red drone is in sight of a camera and
                    # an intercepted has not assigned yet
                    blue_drone_assign, blue_drone, assigned_launch_pad = assign_blue_drone(self.launch_pads, red_drone,
                                                                                              self.blue_drone_speed,
                                                                                              self.capture_predict_steps)

                if vision and blue_drone_assign:
                    # if the red drone is in sight and a blue drone is assigned
                    # Apply launch delay
                    delay += 1
                    if delay > self.launch_delay:
                        # After delay start moving towards the red drone
                        proportional_navigation(blue_drone, red_drone, self.time_step, self.N)

                if blue_drone_assign and not vision:  # If a blue drone is assigned and there is no vision
                    delay += 1
                    if delay > self.launch_delay:

                        if fast_euclid_distance(red_drone.position, blue_drone.position) <= (0.04 ** 2):
                            proportional_navigation(blue_drone, red_drone, self.time_step, self.N)
                        else:
                            proportional_navigation(blue_drone, last_seen_drone, self.time_step, self.N)

                # Target Hit!
                if fast_euclid_distance(red_drone.position, target) <= (0.01 ** 2):
                    target_hits += 1
                    print(f'drone num {drone_id}, CR = {capture_hits * 100 / (drone_id + 1)} steps = {step: ,}')
                    total_steps += step

                    # plot frame if target hit
                    frame_counter += 1
                    self._render_frame(frame_counter, animation_dir, episode, drop_zones, target, start, red_drone, paths, drone_id, capture_hits, target_hits, blue_drone_assign, blue_drone if blue_drone_assign else None, assigned_launch_pad if blue_drone_assign else None)

                    del red_drone
                    del last_seen_drone
                    if blue_drone_assign:
                        del blue_drone
                    break
                # Capture!
                if blue_drone_assign:
                    if fast_euclid_distance(red_drone.position, blue_drone.position) <= (0.01 ** 2):
                        capture_hits += 1
                        # plot frame if intercepted
                        frame_counter += 1
                        self._render_frame(frame_counter, animation_dir, episode, drop_zones, target, start, red_drone, paths, drone_id, capture_hits, target_hits, blue_drone_assign, blue_drone, assigned_launch_pad)

                        set_score(capture_list, assigned_launch_pad)
                        captured_points.append(red_drone.position)
                        print(f'drone num {drone_id}, CR = {capture_hits * 100 / (drone_id + 1)} steps = {step: ,}')
                        total_steps += step
                        del red_drone
                        del last_seen_drone
                        del blue_drone
                        break

        capture_rate = capture_hits / self.num_drones
        return capture_rate * 100, total_steps
