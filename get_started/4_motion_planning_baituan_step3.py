# 4_motion_planning_baituan.py (精简后)
"""UR5e + Robotiq 2F-85 motion planning demo with inline MuJoCo scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from typing import Literal

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.utils.configclass import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler

# Import and install the adapter BEFORE get_handler()
from inline_scene_adapter import install_inline_scene_patches


@configclass
class BaituanSceneCfg(SceneCfg):
    def __post_init__(self):
        if self.mjx_mjcf_path is None:
            self.mjx_mjcf_path = self.mjcf_path


@configclass
class Args:
    robot: str = "ur5e_2f85"
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet",
                 "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    num_envs: int = 1
    headless: bool = True
    solver: Literal["curobo", "pyroki"] = "pyroki"

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)
log.info(f"Using IK solver: {args.solver}")

# Install patches once
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
    gravity=(0.0, 0.0, 0.0),
)

scenario.cameras = [PinholeCameraCfg(
    width=1024, height=1024,
    pos=(4.0, -4.0, 4.0),
    look_at=(0.0, 0.0, 0.0),
)]

handler = get_handler(scenario)

# UR5e initial joint config
robot_dict = {
    "ur5e_2f85": {
        "pos": torch.tensor([1.0, 0.0, 0.895]),
        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "dof_pos": {
            "shoulder_pan_joint": 0.0,
            # "shoulder_pan_joint": 1.5708,
            "shoulder_lift_joint": 0.0,
            # "shoulder_lift_joint": -1.5708,   # -90度，让上臂竖起
            # "elbow_joint": 1.5708,
            "elbow_joint": 0.0,               # 0度，肘部伸直
            "wrist_1_joint": 0.0,
            # "wrist_2_joint": 4.7124,
            # "wrist_3_joint": -1.5708,
            # "wrist_1_joint": -1.5708,         # -90度，前臂自然下垂
            "wrist_2_joint": 0.0,             # 0度，手腕不扭转
            "wrist_3_joint": 0.0,             # 0度，末端执行器朝下
            "right_driver_joint": 0.0, "right_coupler_joint": 0.0,
            "right_spring_link_joint": 0.0, "right_follower_joint": 0.0,
            "left_driver_joint": 0.0, "left_coupler_joint": 0.0,
            "left_spring_link_joint": 0.0, "left_follower_joint": 0.0,
        },
    }
}

init_states = [{"objects": {}, "robots": robot_dict} for _ in range(args.num_envs)]

robot = scenario.robots[0]
from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver
from metasim.utils.kinematics import ee_pose_from_tcp_pose

ik_solver = setup_ik_solver(robot, args.solver)

handler.set_states(init_states)

# ===== 折叠小卫星太阳翼到初始状态 =====
import mujoco as mj

_model = handler.physics.model.ptr   # 原生 MjModel
_data  = handler.physics.data.ptr    # 原生 MjData

# 设置折叠角度（太阳能板完全折叠）
_fold_angles = {
    "hinge_1": -1.5708,      # -90° (range 下限)
    "hinge_2": 3.1416,       # 180° (range 上限)
    "hinge_3": -3.1416,      # -180° (range 下限)
    "hinge_4": 1.5708,       # 90° (range 上限)
    "hinge_1_L": -1.5708,    # 左侧镜像
    "hinge_2_L": 3.1416,
    "hinge_3_L": -3.1416,
    "hinge_4_L": 1.5708,
}

for _jname, _angle in _fold_angles.items():
    _jid = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_JOINT, _jname)
    if _jid >= 0:
        _data.qpos[_model.jnt_qposadr[_jid]] = _angle
        log.info(f"设置关节 {_jname} 到折叠角度: {_angle:.4f} rad")
    else:
        log.warning(f"未找到关节: {_jname}")

# 更新物理状态
mj.mj_forward(_model, _data)
log.info("小卫星太阳翼已折叠到初始状态")

# ===== 新增：校准 wrist_3_link → pinch 的局部偏移 =====
import numpy as np
_wrist_body_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
_pinch_site_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "pinch")
_wrist_pos_world = _data.xpos[_wrist_body_id].copy()
_wrist_rotmat = _data.xmat[_wrist_body_id].reshape(3, 3)
_pinch_pos_world = _data.site_xpos[_pinch_site_id].copy()
_world_offset = _pinch_pos_world - _wrist_pos_world
_local_offset = _wrist_rotmat.T @ _world_offset  # 转换到 wrist_3_link 局部坐标系
log.info(f"[校准] wrist_3_link 世界坐标: {_wrist_pos_world}")
log.info(f"[校准] pinch site 世界坐标: {_pinch_pos_world}")
log.info(f"[校准] 世界坐标偏移 (world): {_world_offset}")
log.info(f"[校准] 局部坐标偏移 (local tcp_rel_pos): {_local_offset}")
# 覆盖 robot config 中的 curobo_tcp_rel_pos 为 MJCF 场景的实际偏移
# 原值 [0, 0, -0.0635] 只适用于 curobo 的运动学模型，不适用于 MJCF
robot.curobo_tcp_rel_pos = _local_offset.tolist()
log.info(f"已覆盖 curobo_tcp_rel_pos 为: {robot.curobo_tcp_rel_pos}")


obs = handler.get_states(mode="tensor")
os.makedirs("get_started/output", exist_ok=True)

obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_baituan_{args.sim}.mp4")
obs_saver.add(obs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smooth_interpolation(t):
    """
    S曲线插值函数，提供平滑的加速和减速
    
    Args:
        t: 插值参数 [0, 1]
    
    Returns:
        平滑插值后的值 [0, 1]
    """
    return 3 * t**2 - 2 * t**3


# def get_trajectory_targets(step, device):
def get_trajectory_targets(step, device, robot_cfg):
    """
    根据当前步数生成末端执行器的目标位置、姿态和夹爪状态
    
    操作分为6个阶段：
    - 阶段1 (0-50):    接近 - 移动到太阳能板上方
    - 阶段2 (50-70):   精确定位 - 下降到抓取位置
    - 阶段3 (70-90):   抓取 - 闭合夹爪
    - 阶段4 (90-170):  展开 - 拉动太阳能板
    - 阶段5 (170-190): 释放 - 打开夹爪
    - 阶段6 (190-220): 撤离 - 远离太阳能板
    
    Args:
        step: 当前步数 [0, 220)
        device: torch device
    
    Returns:
        ee_pos_target: 末端位置目标 [1, 3]
        ee_quat_target: 末端姿态目标 [1, 4]
        gripper_width: 夹爪开度 [0, 1]
    """

    # 当前 pinch 位置（初始状态）
    # 实测：[1.0000, 0.3898, 1.9750]
    initial_pos_pinch = [1.0, 0.3898, 1.975]
    
    # 接近位置（太阳能板上方，保持安全距离）
    # 目标：在折叠后的 panel4_tip_L 上方 15cm
    # panel4_tip_L 折叠后：[0.15, 0.24, 1.28]
    approach_pos_pinch = [0.15, 0.24, 1.28 + 0.15]  # = [0.15, 0.24, 1.43]
    
    # 抓取位置（太阳能板边缘，折叠状态）
    # 目标：pinch 点对准 panel4_tip_L
    # 实测折叠后 panel4_tip_L: [0.15, 0.24, 1.28]
    grasp_pos_pinch = [0.15, 0.24, 1.28]
    
    # 展开位置（太阳能板完全展开后的位置）
    # 目标：拉动到展开状态的 panel4_tip_L
    # 实测展开时 panel4_tip_L: [0.15, -0.30, 1.10]
    deploy_pos_pinch = [0.15, -0.30, 1.10]
    
    # 撤离位置（展开后向上移动，避免碰撞）
    retract_pos_pinch = [0.15, -0.30, 1.10 + 0.20]  # = [0.15, -0.30, 1.30]
    

    # 方案B: 直接使用 pinch/TCP 坐标定义轨迹
    # 在函数末尾通过 ee_pose_from_tcp_pose() 转换为 wrist_3_link 坐标
    initial_pos = initial_pos_pinch
    approach_pos = approach_pos_pinch
    grasp_pos = grasp_pos_pinch
    deploy_pos = deploy_pos_pinch
    retract_pos = retract_pos_pinch


    # 固定姿态：垂直向下
    # 四元数 [qx, qy, qz, qw] = [0, 1, 0, 0] 表示末端执行器朝下
    quat_down = [0.0, 1.0, 0.0, 0.0]
    quat_wrist3 = [-0.5, 0.5, -0.5, 0.5]  # (w, x, y, z)

    # ===== 阶段1: 接近 (步数 0-50) =====
    if step < 50:
        t = step / 50.0
        # 线性插值从初始位置到接近位置
        ee_pos = [
            initial_pos[0] + t * (approach_pos[0] - initial_pos[0]),
            initial_pos[1] + t * (approach_pos[1] - initial_pos[1]),
            initial_pos[2] + t * (approach_pos[2] - initial_pos[2])
        ]
        # ee_quat = quat_down
        ee_quat = quat_wrist3

        gripper_width = 1.0  # 夹爪完全打开
    
    # ===== 阶段2: 精确定位 (步数 50-70) =====
    elif step < 70:
        t = (step - 50) / 20.0
        # 缓慢下降到抓取位置
        ee_pos = [
            approach_pos[0] + t * (grasp_pos[0] - approach_pos[0]),
            approach_pos[1] + t * (grasp_pos[1] - approach_pos[1]),
            approach_pos[2] + t * (grasp_pos[2] - approach_pos[2])
        ]
        ee_quat = quat_down
        gripper_width = 1.0  # 保持打开
    
    # ===== 阶段3: 抓取 (步数 70-90) =====
    elif step < 90:
        t = (step - 70) / 20.0
        # 位置保持不变
        ee_pos = grasp_pos
        ee_quat = quat_down
        # 夹爪逐渐闭合，从1.0到0.2（留20%开口，避免夹坏太阳能板）
        gripper_width = 1.0 - t * 0.8
    
    # ===== 阶段4: 展开 (步数 90-170) ⭐ 关键阶段 =====
    elif step < 170:
        t = (step - 90) / 80.0
        # 使用S曲线插值，提供平滑的运动
        t_smooth = smooth_interpolation(t)
        # 从抓取位置平滑移动到展开位置
        ee_pos = [
            grasp_pos[0] + t_smooth * (deploy_pos[0] - grasp_pos[0]),
            grasp_pos[1] + t_smooth * (deploy_pos[1] - grasp_pos[1]),
            grasp_pos[2] + t_smooth * (deploy_pos[2] - grasp_pos[2])
        ]
        ee_quat = quat_down
        gripper_width = 0.2  # 保持闭合状态
    
    # ===== 阶段5: 释放 (步数 170-190) =====
    elif step < 190:
        t = (step - 170) / 20.0
        # 位置保持在展开位置
        ee_pos = deploy_pos
        ee_quat = quat_down
        # 夹爪逐渐打开，从0.2到1.0
        gripper_width = 0.2 + t * 0.8
    
    # ===== 阶段6: 撤离 (步数 190-220) =====
    else:
        t = (step - 190) / 30.0
        # 从展开位置移动到撤离位置
        ee_pos = [
            deploy_pos[0] + t * (retract_pos[0] - deploy_pos[0]),
            deploy_pos[1] + t * (retract_pos[1] - deploy_pos[1]),
            deploy_pos[2] + t * (retract_pos[2] - deploy_pos[2])
        ]
        ee_quat = quat_down
        gripper_width = 1.0  # 保持打开
    
    # # 转换为torch tensor
    # ee_pos_target = torch.tensor([ee_pos], device=device, dtype=torch.float32)
    # ee_quat_target = torch.tensor([ee_quat], device=device, dtype=torch.float32)
    
    # return ee_pos_target, ee_quat_target, gripper_width

    # 方案B: 先创建 TCP/pinch 坐标的 tensor，然后通过旋转矩阵转换为 wrist_3_link 坐标
    # ee_pos_tcp = torch.tensor([ee_pos], device=device, dtype=torch.float32)
    # ee_quat_tcp = torch.tensor([ee_quat], device=device, dtype=torch.float32)
    
    # # 使用框架的 ee_pose_from_tcp_pose 将 TCP 位姿转换为 EE (wrist_3_link) 位姿
    # # 内部公式: ee_pos = tcp_pos + R_tcp @ (-tcp_rel_pos)
    # # 其中 tcp_rel_pos 已在修改2中被校准为 MJCF 的实际偏移
    # ee_pos_target, ee_quat_target = ee_pose_from_tcp_pose(robot_cfg, ee_pos_tcp, ee_quat_tcp)
    
    ee_pos_tcp = torch.tensor([ee_pos], device=device, dtype=torch.float32)
    ee_quat_wrist = torch.tensor([ee_quat], device=device, dtype=torch.float32)

    from metasim.utils.math import matrix_from_quat
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(device)
    ee_pos_target = ee_pos_tcp + torch.matmul(
        matrix_from_quat(ee_quat_wrist), 
        -tcp_rel_pos.unsqueeze(-1)
    ).squeeze()
    ee_quat_target = ee_quat_wrist

    return ee_pos_target, ee_quat_target, gripper_width



# 修改循环范围，从200改为220（增加撤离阶段）
for step in range(220):
    log.debug(f"Step {step}")
    states = handler.get_states()

    # ===== 修改点1: 使用轨迹生成函数 =====
    # 替换原来的固定轨迹
    # ee_pos_target, ee_quat_target, gripper_width = get_trajectory_targets(step, device)
    ee_pos_target, ee_quat_target, gripper_width = get_trajectory_targets(step, device, robot)
    
    # 扩展到所有环境（如果有多个并行环境）
    ee_pos_target = ee_pos_target.repeat(args.num_envs, 1)
    ee_quat_target = ee_quat_target.repeat(args.num_envs, 1)

    # 获取当前关节位置（保持不变）
    reorder_idx = handler.get_joint_reindex(robot.name)
    inv_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
    curr_q = obs.robots[robot.name].joint_pos[:, inv_idx]

    # 求解逆运动学（保持不变）
    q_sol, _ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_q)

    # ===== 修改点2: 使用动态夹爪控制 =====
    # 替换原来的 torch.ones
    gripper_command = torch.full((args.num_envs,), gripper_width, device=device)
    gripper_widths = process_gripper_command(gripper_command, robot, device)

    # 组合动作并执行（保持不变）
    actions = ik_solver.compose_joint_action(q_sol, gripper_widths, curr_q, return_dict=True)

    handler.set_dof_targets(actions)
    handler.simulate()
    obs = handler.get_states(mode="tensor")
    obs_saver.add(obs)

obs_saver.save()
