import numpy as np
from scipy.ndimage import map_coordinates
import math

def get_line_profile(data, start, angle_deg, num_points=None):
    """
    Extract a line profile from image data at a specified angle and starting point.

    Args:
        data (ndarray): 2D image data
        start (tuple): Starting coordinates (y, x)
        angle_deg (float): Angle in degrees
        num_points (int, optional): Number of sampling points along the line

    Returns:
        dict: Profile information
            - 'distances': Array of distances from the start point
            - 'values': Values along the line
            - 'coords': Coordinates of the line (xs, ys)
    """
    h, w = data.shape
    y0, x0 = start
    angle_rad = math.radians(angle_deg)
    
    dy = math.sin(angle_rad)
    dx = math.cos(angle_rad)

    # Calculate the maximum distance to the image boundary
    ts = []
    if dx != 0:
        ts.extend([(0 - x0) / dx, ((w - 1) - x0) / dx])
    if dy != 0:
        ts.extend([(0 - y0) / dy, ((h - 1) - y0) / dy])
    
    # Find the minimum positive intersection in the positive direction
    valid_ts = [t for t in ts if t > 0]
    t_max = min(valid_ts) if valid_ts else 0

    # Number of sampling points (default to diagonal length if not specified)
    if num_points is None:
        num_points = int(math.hypot(h, w)) + 1

    distances = np.linspace(0, t_max, num_points)
    xs = x0 + distances * dx
    ys = y0 + distances * dy

    # Get values using bilinear interpolation
    line_values = map_coordinates(data, [ys, xs], order=1)

    return {
        "distances": distances,
        "values": line_values,
        "coords": (xs, ys)
    }
        