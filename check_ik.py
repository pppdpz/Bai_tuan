import sys, os
import torch
import importlib
from metasim.utils.math import matrix_from_quat
from get_started.inline_scene_adapter import install_inline_scene_patches
from metasim.utils.setup_util import get_handler
from metasim.scenario.scenario import ScenarioCfg

try:
    module = importlib.import_module("get_started.4_motion_planning_baituan")
    BaituanSceneCfg = getattr(module, "BaituanSceneCfg")

    install_inline_scene_patches()
    scenario = ScenarioCfg(
        scene=BaituanSceneCfg(
            name="baituan_satellite",
            mjcf_path="asset_baituan/example_scene_y.xml",
        ),
        robots=["ur5e_2f85"],
        objects=[],
        simulator="mujoco",
        headless=True,
        num_envs=1,
        gravity=(0.0, 0.0, 0.0),
    )
    
    handler = get_handler(scenario, "mujoco")
    
    _model = handler.mj_model
    _data = handler.mj_data
    import metasim.callbacks
    _mj = metasim.callbacks.get_mujoco()

    wrist_id = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
    tcp_id = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, "tcp_link")

    wrist_pos = _data.xpos[wrist_id]
    tcp_pos = _data.xpos[tcp_id]
    wrist_mat = _data.xmat[wrist_id].reshape(3, 3)

    tcp_offset_world = tcp_pos - wrist_pos
    tcp_offset_wrist = wrist_mat.T @ tcp_offset_world
    print("tcp_offset_wrist (local xyz):", tcp_offset_wrist)

    # Test the transformation block logic
    EE_QUAT = torch.tensor([[0.707, 0.707, 0.0, 0.0]])
    tcp_target = torch.tensor([[0.5, 0.035, 0.485]])
    
    tcp_offset_tor = torch.tensor(tcp_offset_wrist).float()
    rotmat = matrix_from_quat(EE_QUAT)
    wrist_target = tcp_target - torch.bmm(rotmat, tcp_offset_tor.view(1, 3, 1)).squeeze(-1)
    
    print("TCP Target Base:", tcp_target.numpy())
    print("EE_QUAT:", EE_QUAT.numpy())
    print("Rotmat:\n", rotmat.numpy())
    print("Wrist Target Base derived:", wrist_target.numpy())
    
except Exception as e:
    import traceback
    traceback.print_exc()
