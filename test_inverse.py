import numpy as np
from scipy.spatial.transform import Rotation as R

pos = np.array([0.0, 0.1, -0.1558])
rot = np.array([np.pi, 0.0, 0.0])
R_mat = R.from_euler('xyz', rot).as_matrix()

T = np.eye(4)
T[:3, :3] = R_mat
T[:3, 3] = pos

print("T:")
print(np.round(T, 4))
print("inv(T):")
print(np.round(np.linalg.inv(T), 4))
print("Are they equal?", np.allclose(T, np.linalg.inv(T)))
