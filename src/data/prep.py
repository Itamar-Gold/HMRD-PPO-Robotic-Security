import numpy as np
import pandas as pd


def generate_fence(x0, x1, y0, y1, density):
    X = np.linspace(x0, x1, density)
    Y = np.linspace(y0, y1, density)
    border_points = [(float(X[i]), float(Y[i])) for i in range(len(Y))]
    return border_points

def load_roi(excel_file='data/roi.xlsx'):
    # Load Region of Interest data from Excel file
    roi_df = pd.read_excel(excel_file)
    rois = roi_df.to_dict(orient='records')
    
    
    facility_fence = generate_fence(0, 80, 70, 70, 10)
    
    # Plot RoI
    centers = []
    edges = []
    for roi in rois:
        # Determine the number of edges by counting available edge_x columns
        num_edges = sum(1 for key in roi if 'edge_' in key and '_x' in key)
        centers.append((roi['center_x'], roi['center_y']))
    
        # Collect all edge points
        for i in range(num_edges):
            edges.append((roi[f'edge_{i}_x'], roi[f'edge_{i}_y']))
    
    # Clean NaN from edges
    edges = [(x, y) for x, y in edges if not (np.isnan(x) or np.isnan(y))]
    
    
    # --- Output Data ---
    secure_points = []  # Store RoI points (edges and centers)
    secure_points.extend(centers)  # Save RoI center as a secure points
    secure_points.extend(edges)  # Save RoI edges as a secure points
    
    secure_points.extend(facility_fence) # Save fence as a secure points
    
    # Sort all points by x-coordinate, then by y-coordinate
    secure_points_sorted = sorted(secure_points, key=lambda p: (p[0], p[1]))
    return secure_points_sorted