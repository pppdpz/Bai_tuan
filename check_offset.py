import mujoco as mj
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_started.inline_scene_adapter import generate_inline_scene

xml_str = generate_inline_scene("asset_baituan/example_scene_y.xml")
model = mj.MjModel.from_xml_string(xml_str)
data = mj.MjData(model)
mj.mj_forward(model, data)

wrist_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
tcp_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "tcp_link")

print("Wrist Pos:", data.xpos[wrist_id])
print("TCP Pos:", data.xpos[tcp_id])

offset = data.xpos[tcp_id] - data.xpos[wrist_id]
wrist_rot = data.xmat[wrist_id].reshape(3, 3)
local_offset = wrist_rot.T @ offset

print("Local Translation Offset:", local_offset)
print("Local TCP Rot:", wrist_rot.T @ data.xmat[tcp_id].reshape(3, 3))
