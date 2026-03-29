import sys
import os
import mujoco as mj
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_started.inline_scene_adapter import generate_inline_scene
from metasim.utils.math import quat_from_matrix

xml_str = generate_inline_scene("asset_baituan/example_scene_y.xml")
model = mj.MjModel.from_xml_string(xml_str)
data = mj.MjData(model)

# 确保全部关节在默认0位
mj.mj_forward(model, data)

wrist_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
tcp_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "tcp_link")

wrist_pos = data.xpos[wrist_id]
wrist_mat = data.xmat[wrist_id].reshape(3, 3)

tcp_pos = data.xpos[tcp_id]
tcp_mat = data.xmat[tcp_id].reshape(3, 3)

# 1. 计算位置相对偏移 (在wrist局部坐标系下)
offset_world = tcp_pos - wrist_pos
offset_local = wrist_mat.T @ offset_world

# 2. 计算旋转差异 (R_wrist^tcp = R_wrist^T * R_tcp)
rot_diff = wrist_mat.T @ tcp_mat

print("---")
print("Local Translation Offset (wrist -> tcp):", offset_local)
print("Rotation Diff Matrix (wrist -> tcp):")
print(rot_diff)

# 看看 quat_from_matrix (如果有的话) 我们直接手算
# 为了查看旋转差异的大致情况：
import scipy.spatial.transform
r = scipy.spatial.transform.Rotation.from_matrix(rot_diff)
print("Local Euler (xyz):", r.as_euler('xyz', degrees=True))
print("---")
