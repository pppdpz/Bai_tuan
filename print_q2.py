import sys
sys.path.append("/home/e0/RoboVerse")
import mujoco
from inline_scene_adapter import install_inline_scene_patches
install_inline_scene_patches()

model = mujoco.MjModel.from_xml_path("asset_baituan/example_scene_y.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ur5e_base")
tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
tcp_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tcp_link")
print("ur5e_base:", data.xpos[base_bid])
print("tcp site:", data.site_xpos[tcp_sid])
print("tcp_link body:", data.xpos[tcp_link_id])

tip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panel4_tip_L")
print("panel tip L:", data.site_xpos[tip_sid])
