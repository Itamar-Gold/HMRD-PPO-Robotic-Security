import math
from src.utils.geometry import fast_euclid_distance

class Drone:
    """Base entity representing a physical drone with kinematics."""

    def __init__(self, position, velocity, angle):
        self.position = position  # (x, y) coordinates
        self.velocity = velocity  # Speed (constant)
        self.angle = angle  # Heading angle in radians
        self.prev_theta_LOS = None

    def move(self, time_step=0.5):
        """Moves the drone in the direction of its heading."""
        x, y = self.position
        x += self.velocity * math.cos(self.angle) * time_step # Update x-coordinate
        y += self.velocity * math.sin(self.angle) * time_step # Update y-coordinate
        self.position = (x, y)  # Update position

def set_score(coordinate_list, coord):
    """Increments the capture/detection score for a specific Blue Team asset."""
    try:
        index = coordinate_list.index(coord) + 1  # Find coordinate and get value index
        coordinate_list[index] += 1
    except ValueError:
        print(f"Coordinate {coord} not found!")

def find_lookahead_point(position, path, lookahead_dist, last_index):
    """Identifies the next valid lookahead waypoint for the Pure Pursuit controller."""
    for i in range(last_index, len(path)):  # Start from last_index to avoid backtracking
        if fast_euclid_distance(position, path[i]) >= lookahead_dist ** 2:
            return path[i], i  # Return new target point and update last index
    return path[-1], len(path) - 1  # If no point found, return last waypoint

def pure_pursuit_target(drone, path, lookahead_dist=2.0, time_step=0.5, last_index=0):
    """Steers a drone towards a trajectory path using a Pure Pursuit control law."""
    lookahead_point, new_index = find_lookahead_point(drone.position, path, lookahead_dist, last_index)

    # Compute new heading angle toward lookahead point
    new_angle = math.atan2(lookahead_point[1] - drone.position[1], lookahead_point[0] - drone.position[0])

    # Update drone's heading and move forward
    drone.angle = new_angle
    drone.move(time_step)

    return new_index  # Return updated index to track progress


def proportional_navigation(blue_drone, target, time_step, N=2.5):
    """Navigates the Blue Drone towards the Red Drone using a Proportional Navigation (PN) law."""
    # Extract positions of the blue_drone and target
    x_I, y_I = blue_drone.position
    x_T, y_T = target.position

    # Compute the current Line-of-Sight (LOS) angle between the blue_drone and target
    theta_LOS = math.atan2(y_T - y_I, x_T - x_I)  # Angle from blue_drone to target

    # Compute the rate of change of the LOS angle (approximate derivative)
    if blue_drone.prev_theta_LOS is None:
        blue_drone.prev_theta_LOS = theta_LOS  # Initialize the first time

    theta_LOS_rate = theta_LOS - blue_drone.prev_theta_LOS  # Change in LOS angle
    blue_drone.prev_theta_LOS = theta_LOS  # Update for next step

    # Adjust the blue_drone’s heading angle based on Proportional Navigation formula
    blue_drone.angle += N * theta_LOS_rate / time_step  # Proportional correction

    # Move the blue_drone in its new direction
    blue_drone.move(time_step)

def assign_blue_drone(launch_pads, red_drone, blue_drone_speed, prediction_steps=5):
    """Selects the optimal Blue Drone launch pad based on the predicted Red Drone trajectory."""
    predicted_x = red_drone.position[0] + red_drone.velocity * math.cos(red_drone.angle) * prediction_steps
    predicted_y = red_drone.position[1] + red_drone.velocity * math.sin(red_drone.angle) * prediction_steps
    predicted_pos = (predicted_x, predicted_y)

    best_score = 0
    best_launch_pad = None
    for inter in launch_pads:  # Iterate through available launch_pads
        delta_x = predicted_pos[0] - inter[0]
        delta_y = predicted_pos[1] - inter[1]
        angle_to_target = math.degrees(math.atan2(delta_y, delta_x))

        delta_angle = abs(angle_to_target - (math.degrees(red_drone.angle) + 180))

        score = 10 * (1 / (delta_angle + 1e-6))
        if score > best_score:
            best_score = score
            best_launch_pad = inter

    if best_launch_pad:
        delta_x = predicted_pos[0] - best_launch_pad[0]
        delta_y = predicted_pos[1] - best_launch_pad[1]
        angle_to_target = math.atan2(delta_y, delta_x)

        blue_drone = Drone(position=best_launch_pad, velocity=blue_drone_speed, angle=angle_to_target)
        return True, blue_drone, best_launch_pad
    return False, None, None