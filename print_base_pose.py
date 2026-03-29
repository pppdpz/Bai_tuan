import mujoco
from inline_scene_adapter import install_inline_scene_patches
install_inline_scene_patches()

model = mujoco.MjModel.from_xml_path("asset_baituan/example_scene_y.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ur5e_base")
print("ur5e_base pos:", data.xpos[base_bid])
print("ur5e_base quat:", data.xquat[base_bid])
print("ur5e_base rot mat:\n", data.xmat[base_bid].reshape(3,3))
