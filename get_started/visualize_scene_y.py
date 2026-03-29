"""可视化 example_scene_y.xml 场景（UR5e + 卫星 + 太阳能板）。

用法:
    # 交互式窗口（默认）
    python get_started/visualize_scene_y.py

    # 无头模式，仅保存截图
    python get_started/visualize_scene_y.py --headless True

    # 自定义相机位置
    python get_started/visualize_scene_y.py --cam-pos 3.0 -3.0 3.0 --cam-look-at 0.0 0.0 0.5
"""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from typing import Literal, Tuple

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import imageio

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.utils.configclass import configclass
from metasim.utils.setup_util import get_handler

from inline_scene_adapter import install_inline_scene_patches


@configclass
class BaituanSceneCfg(SceneCfg):
    def __post_init__(self):
        if self.mjx_mjcf_path is None:
            self.mjx_mjcf_path = self.mjcf_path


@configclass
class Args:
    sim: Literal["mujoco", "mjx"] = "mujoco"
    num_envs: int = 1
    headless: bool = False
    cam_pos: Tuple[float, float, float] = (4.0, -4.0, 4.0)
    cam_look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    save_path: str = "get_started/output/scene_y_preview.png"


args = tyro.cli(Args)

install_inline_scene_patches()

scenario = ScenarioCfg(
    scene=BaituanSceneCfg(
        name="baituan_satellite",
        mjcf_path="asset_baituan/example_scene_y.xml",
    ),
    robots=["ur5e_2f85"],
    objects=[],
    simulator=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
    # gravity=(0.0, 0.0, -9.81),
    gravity=(0.0, 0.0, 0.0),
)

scenario.cameras = [
    PinholeCameraCfg(
        name="scene_cam",
        width=1024,
        height=1024,
        pos=args.cam_pos,
        look_at=args.cam_look_at,
    )
]

handler = get_handler(scenario)

# UR5e 初始关节角度
robot_dict = {
    "ur5e_2f85": {
        "pos": torch.tensor([0.0, 0.0, 0.0]),
        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "dof_pos": {
            "shoulder_pan_joint": 0.0,
            # "shoulder_lift_joint": 0.0,
            "shoulder_lift_joint": -0.754,
            # "elbow_joint": 1.5708,
            "elbow_joint": -1.290,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 4.7124,
            "wrist_3_joint": -1.5708,
            "right_driver_joint": 0.0, "right_coupler_joint": 0.0,
            "right_spring_link_joint": 0.0, "right_follower_joint": 0.0,
            "left_driver_joint": 0.0, "left_coupler_joint": 0.0,
            "left_spring_link_joint": 0.0, "left_follower_joint": 0.0,
        },
    }
}

init_states = [{"objects": {}, "robots": robot_dict} for _ in range(args.num_envs)]

handler.set_states(init_states)

import mujoco as mj

_model = handler.physics.model.ptr   # 原生 MjModel
_data  = handler.physics.data.ptr    # 原生 MjData

# 先执行一次 forward 确保 site_xpos 已更新
mj.mj_forward(_model, _data)

# 读取所有红色小球（site）的位置
site_names = [
    "panel4_tip",        # 小卫星碎片 - 右翼末端
    "panel4_tip_L",      # 小卫星碎片 - 左翼末端
    "carrier_panel4_tip",    # 大卫星 - 右翼末端
    "carrier_panel4_tip_L",  # 大卫星 - 左翼末端
]

for name in site_names:
    sid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        pos = _data.site_xpos[sid]
        log.info(f"{name}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    else:
        log.warning(f"site '{name}' 未找到")

# 在这里添加（约第 138 行，site_names 循环结束后）

# ===== 验证坐标偏移 =====
log.info("=== 坐标偏移验证 ===")

wrist_bid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
tcp_bid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_BODY, "tcp_link")
pinch_sid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "pinch")

if wrist_bid >= 0 and tcp_bid >= 0 and pinch_sid >= 0:
    wrist_pos = _data.xpos[wrist_bid]
    tcp_pos = _data.xpos[tcp_bid]
    pinch_pos = _data.site_xpos[pinch_sid]
    
    log.info(f"wrist_3_link: [{wrist_pos[0]:.4f}, {wrist_pos[1]:.4f}, {wrist_pos[2]:.4f}]")
    log.info(f"tcp_link:     [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}]")
    log.info(f"pinch:        [{pinch_pos[0]:.4f}, {pinch_pos[1]:.4f}, {pinch_pos[2]:.4f}]")
    
    offset_wrist_tcp = tcp_pos - wrist_pos
    offset_tcp_pinch = pinch_pos - tcp_pos
    offset_wrist_pinch = pinch_pos - wrist_pos
    
    log.info(f"wrist→tcp:    [{offset_wrist_tcp[0]:.4f}, {offset_wrist_tcp[1]:.4f}, {offset_wrist_tcp[2]:.4f}]")
    log.info(f"tcp→pinch:    [{offset_tcp_pinch[0]:.4f}, {offset_tcp_pinch[1]:.4f}, {offset_tcp_pinch[2]:.4f}]")
    log.info(f"wrist→pinch:  [{offset_wrist_pinch[0]:.4f}, {offset_wrist_pinch[1]:.4f}, {offset_wrist_pinch[2]:.4f}]")
else:
    log.warning("无法找到 wrist_3_link, tcp_link 或 pinch")



# 读取机械臂关键位置
key_sites = {
    "tcp": "机械臂工具中心点",
    "pinch": "夹爪捏合点",
}

log.info("=== 机械臂末端执行器位置 ===")
for site_name, description in key_sites.items():
    sid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, site_name)
    if sid >= 0:
        pos = _data.site_xpos[sid]
        log.info(f"{description} ({site_name}): x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
        
        # 计算与 panel4_tip_L 的相对位置
        panel_sid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "panel4_tip_L")
        if panel_sid >= 0:
            panel_pos = _data.site_xpos[panel_sid]
            relative_pos = panel_pos - pos
            distance = ((relative_pos[0]**2 + relative_pos[1]**2 + relative_pos[2]**2)**0.5)
            log.info(f"  → panel4_tip_L 相对偏移: dx={relative_pos[0]:.4f}, dy={relative_pos[1]:.4f}, dz={relative_pos[2]:.4f}")
            log.info(f"  → 距离: {distance:.4f} m")
    else:
        log.warning(f"{description} site '{site_name}' 未找到")



# === 折叠小卫星太阳翼 ===
_fold_angles = {
    "hinge_1": -1.5708, "hinge_2": 3.1416,
    "hinge_3": -3.1416, "hinge_4": 1.5708,
    "hinge_1_L": -1.5708, "hinge_2_L": 3.1416,
    "hinge_3_L": -3.1416, "hinge_4_L": 1.5708,
}

for _jname, _angle in _fold_angles.items():
    _jid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_JOINT, _jname)
    if _jid >= 0:
        _data.qpos[_model.jnt_qposadr[_jid]] = _angle

# 在这里添加（约第 203 行，折叠后的验证循环结束后）

# ===== 折叠后再次验证坐标偏移 =====
log.info("=== 折叠后坐标偏移验证 ===")

if wrist_bid >= 0 and tcp_bid >= 0 and pinch_sid >= 0:
    wrist_pos = _data.xpos[wrist_bid]
    tcp_pos = _data.xpos[tcp_bid]
    pinch_pos = _data.site_xpos[pinch_sid]
    
    log.info(f"wrist_3_link: [{wrist_pos[0]:.4f}, {wrist_pos[1]:.4f}, {wrist_pos[2]:.4f}]")
    log.info(f"tcp_link:     [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}]")
    log.info(f"pinch:        [{pinch_pos[0]:.4f}, {pinch_pos[1]:.4f}, {pinch_pos[2]:.4f}]")
    
    offset_wrist_tcp = tcp_pos - wrist_pos
    offset_tcp_pinch = pinch_pos - tcp_pos
    offset_wrist_pinch = pinch_pos - wrist_pos
    
    log.info(f"wrist→tcp:    [{offset_wrist_tcp[0]:.4f}, {offset_wrist_tcp[1]:.4f}, {offset_wrist_tcp[2]:.4f}]")
    log.info(f"tcp→pinch:    [{offset_tcp_pinch[0]:.4f}, {offset_tcp_pinch[1]:.4f}, {offset_tcp_pinch[2]:.4f}]")
    log.info(f"wrist→pinch:  [{offset_wrist_pinch[0]:.4f}, {offset_wrist_pinch[1]:.4f}, {offset_wrist_pinch[2]:.4f}]")


mj.mj_forward(_model, _data)

# 折叠后再次读取 site 位置
log.info("=== 折叠后 site 位置 ===")
for name in site_names:
    sid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        pos = _data.site_xpos[sid]
        log.info(f"{name}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    else:
        log.warning(f"site '{name}' 未找到")



log.info("=== 折叠后机械臂末端执行器位置 ===")
for site_name, description in key_sites.items():
    sid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, site_name)
    if sid >= 0:
        pos = _data.site_xpos[sid]
        log.info(f"{description} ({site_name}): x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
        
        panel_sid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "panel4_tip_L")
        if panel_sid >= 0:
            panel_pos = _data.site_xpos[panel_sid]
            relative_pos = panel_pos - pos
            distance = ((relative_pos[0]**2 + relative_pos[1]**2 + relative_pos[2]**2)**0.5)
            log.info(f"  → panel4_tip_L 相对偏移: dx={relative_pos[0]:.4f}, dy={relative_pos[1]:.4f}, dz={relative_pos[2]:.4f}")
            log.info(f"  → 距离: {distance:.4f} m")




handler.set_states(init_states)

mj.mj_forward(_model, _data)

obs = handler.get_states(mode="tensor")

# 保存截图
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
rgb = next(iter(obs.cameras.values())).rgb[0].cpu().numpy()
imageio.imwrite(args.save_path, rgb)
log.info(f"截图已保存: {args.save_path}")

# 非 headless 模式下保持窗口打开，方便交互查看
if not args.headless:
    log.info("交互窗口已打开，关闭窗口退出。")
    try:
        while True:
            handler.simulate()
    except KeyboardInterrupt:
        pass

handler.close()
