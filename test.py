import numpy as np
from scipy.spatial import cKDTree

# Function to fill slt with nearby non-zero values
def fill_nearby_slt(gribfield_mod, lsm_id, slt_id):
    # Get coordinates of all points
    num_points = len(gribfield_mod[slt_id])
    points = np.arange(num_points)

    # Identify valid slt values (slt > 0 and lsm > 0.5)
    valid_mask = (gribfield_mod[slt_id] > 0) & (gribfield_mod[lsm_id] > 0.5)
    valid_points = points[valid_mask]
    valid_slt_values = gribfield_mod[slt_id][valid_mask]

    # Identify the points where we need to fill (slt == 0 and lsm > 0.5)
    fill_mask = (gribfield_mod[slt_id] == 0) & (gribfield_mod[lsm_id] > 0.5)
    fill_points = points[fill_mask]

    # Perform nearest neighbor interpolation
    if len(valid_points) > 0:
        tree = cKDTree(valid_points.reshape(-1, 1))
        nearest_indices = tree.query(fill_points.reshape(-1, 1), k=1)[1]
        nearest_values = valid_slt_values[nearest_indices]
        
        # Fill the missing slt values
        gribfield_mod[slt_id][fill_mask] = nearest_values

    return gribfield_mod

# Example usage
gribfield_mod = {
    'lsm': np.array([0, 0.6, 0.8, 0.4, 0.7]),  # Example lsm
    'slt': np.array([0, 3, 0, 0, 6])           # Example slt, needs to fill nearby
}

filled_slt = fill_nearby_slt(gribfield_mod, 'lsm', 'slt')
print(filled_slt)
