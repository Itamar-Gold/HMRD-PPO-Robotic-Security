import math
from queue import PriorityQueue
import numpy as np
from src.utils.geometry import euclid_distance

def generate_drop_zones(centers, radius=3, weight=5, stretch_factor=2, step_size=0.5):
    """Creates a repulsive penalty field map from historically captured Red Drone positions."""
    drop_zone = {}

    for center in centers:
        cx, cy = center
        # Rounding the number to match the step size
        cx = round(cx / step_size) * step_size
        cy = round(cy / step_size) * step_size

        # Iterate using step_size instead of integers
        dx_range = np.arange(-radius, radius + step_size, step_size)
        dy_range = np.arange(0, radius * stretch_factor + step_size, step_size)

        for dx in dx_range:
            for dy in dy_range:
                distance = np.sqrt(dx ** 2 + (dy / stretch_factor) ** 2)  # Stretch vertically
                if distance <= radius:
                    penalty = weight * (1 - (distance / radius))  # Decay effect
                    key = (round(cx + dx, 2), round(cy + dy, 2))  # Ensure precision

                    # Merge penalties if multiple drop zones overlap
                    drop_zone[key] = max(drop_zone.get(key, 0), penalty)

    return drop_zone

def aaa_star_drone(start, end, x_lim, y_lim, step_size, captured_points):
    """Adaptive Adversarial A* (AAA*) algorithm. Computes evasion paths considering dynamic penalty fields."""
    start = (float(start[0]), float(start[1]))
    end = (float(end[0]), float(end[1]))
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: euclid_distance(start, end)}
    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == end:
            return reconstruct_path(came_from, end)

        # Possible moves: up, down, left, right
        directions = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                      (-step_size, -step_size), (-step_size, step_size), (step_size, -step_size), (step_size, step_size)]

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # Ensure neighbor is within bounds (not outside the map)
            if x_lim[0] <= neighbor[0] <= x_lim[1] and y_lim[0] <= neighbor[1] <= y_lim[1]:
                move_cost = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                temp_g_score = g_score[current] + move_cost

                # If the neighbor was an captured point or in an avoidance zone, increase its cost
                if neighbor in captured_points:
                    temp_g_score += captured_points[neighbor]  # Increase cost dynamically

                if temp_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + euclid_distance(neighbor, end)

                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

    return None  # No path found

def reconstruct_path(came_from, current):
    """Reconstructs the A* path recursively from the destination back to the start."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]