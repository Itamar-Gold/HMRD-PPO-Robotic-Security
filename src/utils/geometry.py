import math
from typing import Tuple, Union

def euclid_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Computes standard Euclidean distance between two points."""
    distance = math.sqrt(pow((point2[1] - point1[1]), 2) + pow((point2[0] - point1[0]), 2))
    return distance

def fast_euclid_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Computes squared Euclidean distance for optimized proximity checks (no sqrt)."""
    distance = pow((point2[1] - point1[1]), 2) + pow((point2[0] - point1[0]), 2)
    return distance

def is_in_range(point1: Tuple[float, float], point2: Tuple[float, float, float], det_range: float, shape: int = 1, camera_angle: float = 0.0, sweep_angle: float = 60.0) -> Union[int, bool]:
    """
    Checks if point1 is within the detection range of point2, considering the shape of the camera.

    Parameters:
        point1 (tuple): Coordinates of the point to check (x, y).
        point2 (tuple): Coordinates and angle of the camera (x, y, angle).
        range (float): Detection range.
        shape (int): 1 for circular, 2 for sector-based.
        camera_angle (float): Orientation angle of the camera (in degrees).
        sweep_angle (float): Sweep angle of the sector (in degrees).

    Returns:
        int/bool: 1/True if within range, 0/False otherwise.
    """
    if shape == 1:
        # Circular detection range
        distance = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        return 1 if distance < det_range ** 2 else 0
    elif shape == 2:
        # Sector-based detection range
        distance = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        if distance > det_range ** 2:
            return False  # Out of range

        # Compute angle between camera and point
        delta_x = point1[0] - point2[0]
        delta_y = point1[1] - point2[1]
        point_angle = math.degrees(math.atan2(delta_y, delta_x))  # Angle in degrees

        # Normalize angles to [0, 360)
        camera_angle = camera_angle % 360
        point_angle = point_angle % 360

        # Calculate angle difference
        angle_diff = (point_angle - camera_angle + 180) % 360 - 180

        # Check if within sweep angle
        if -sweep_angle / 2 <= angle_diff <= sweep_angle / 2:
            return True  # Within detection sector

        return False  # Outside detection sector
    else:
        return -1  # Indicate an error

def get_arc_edge_point(camera, radius):
    """
    Returns the (x, y) coordinate at the center of the arc along the center direction.

    Args:
        camera: [(x, y, angle)]
        radius (float): Distance from the center to the edge of the arc

    Returns:
        (float, float): Coordinates of the center point of detection along the camera's center line
    """
    x, y, angle_deg = camera
    radius = radius / 2  # We want to use the center of the vector to avoid covering the same areas
    angle_rad = math.radians(angle_deg)
    edge_x = round(x + radius * math.cos(angle_rad), 2)
    edge_y = round(y + radius * math.sin(angle_rad), 2)
    return edge_x, edge_y

def get_cameras_pointers(cameras, radius):
    """

    Args:
        cameras: [[(x, y, angle)],...] list of cameras
        radius: The cameras radius

    Returns:
        pointers to the center of detection
    """
    pointers = []
    for camera in cameras:
        edge_point = get_arc_edge_point(camera, radius)
        pointers.append(edge_point)
    return pointers