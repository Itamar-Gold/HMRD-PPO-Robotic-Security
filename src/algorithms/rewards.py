import math

def compute_density_penalty(cdfc_state, min_dist, penalty_factor=10):
    """
    Computes a penalty based on the density of launch pads.

    Args:
        cdfc_state (list of tuples): List of (x, y) positions of launch pads.
        min_dist (float): Minimum allowed distance between launch pads.
        penalty_factor (float): Scaling factor for the penalty.

    Returns:
        float: Penalty value.
    """
    num_launch_pads = len(cdfc_state)
    penalty = 0.0

    # Compute pairwise distances
    for i in range(num_launch_pads):
        for j in range(i + 1, num_launch_pads):
            dist = (cdfc_state[i][0] - cdfc_state[j][0]) ** 2 + (cdfc_state[i][1] - cdfc_state[j][1]) ** 2
            if dist < min_dist ** 2:  # instead of using sqrt()
                penalty += penalty_factor * (min_dist - math.sqrt(dist))

    return round(penalty, 1)

def compute_detect_density(cfc_state, radius, density, get_cameras_pointers):
    """
    Computes a penalty based on the density of cameras.

    Args:
        cfc_state (list of tuples): List of (x, y, angle) positions of cameras.
        radius (float): Detection radius of the camera.
        density (float): Minimum allowed distance between camera detection centers.
        get_cameras_pointers (function): Function to calculate camera detection centers.

    Returns:
        float: Penalty value.
    """
    pointers = get_cameras_pointers(cfc_state, radius)
    penalty = compute_density_penalty(pointers, density)
    return round(penalty, 1)