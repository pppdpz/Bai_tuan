import torch
from metasim.scenario.robot import RobotCfg
import mujoco

model = mujoco.MjModel.from_xml_path("../asset_baituan/example_scene_y.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

jnames = []
for i in range(model.njnt):
    jnames.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

print("MuJoCo joints:", jnames)

