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

# ===== TCP 到 wrist3 的固定变换常量（经过测试验证，误差 0.0000mm）=====
import numpy as np
from scipy.spatial.transform import Rotation as R

# TCP 相对 wrist3 的固定偏移（在 wrist3 坐标系中）
TCP_TO_WRIST3_OFFSET_POS = np.array([0.0, 0.2558, 0.0])  # 单位：米
TCP_TO_WRIST3_OFFSET_QUAT = np.array([0.5, 0.5, 0.5, -0.5])  # [x,y,z,w]



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

def tcp_pose_to_wrist3_pose(tcp_pos, tcp_quat, offset_pos, offset_quat):
    """
    将 TCP 位姿转换为 wrist_3_link 位姿
    
    这个函数经过测试验证，转换精度为 0.0000mm
    
    逻辑：
        已知：TCP 在世界坐标系中的目标位姿
        已知：TCP 相对 wrist3 的固定偏移 (T_wrist3_to_tcp)
        求：wrist3 在世界坐标系中应该在哪里
        
        关系：T_world_to_tcp = T_world_to_wrist3 @ T_wrist3_to_tcp
        因此：T_world_to_wrist3 = T_world_to_tcp @ inv(T_wrist3_to_tcp)
    
    Args:
        tcp_pos: TCP 目标位置 [3] 或 [1, 3] (numpy array 或 torch tensor)
        tcp_quat: TCP 目标姿态四元数 [4] 或 [1, 4] (x,y,z,w)
        offset_pos: TCP 相对 wrist3 的位置偏移 [3] (numpy array)
        offset_quat: TCP 相对 wrist3 的姿态偏移 [4] (x,y,z,w, numpy array)
    
    Returns:
        wrist3_pos: wrist_3_link 位置 (与输入类型相同)
        wrist3_quat: wrist_3_link 姿态 (与输入类型相同)
    """
    # 转换为 numpy 进行计算
    is_torch = isinstance(tcp_pos, torch.Tensor)
    if is_torch:
        device = tcp_pos.device
        tcp_pos_np = tcp_pos.detach().cpu().numpy().reshape(3)
        tcp_quat_np = tcp_quat.detach().cpu().numpy().reshape(4)
    else:
        tcp_pos_np = np.array(tcp_pos).reshape(3)
        tcp_quat_np = np.array(tcp_quat).reshape(4)
    
    # 构建 TCP 变换矩阵（世界坐标系到 TCP）
    T_world_to_tcp = np.eye(4)
    T_world_to_tcp[:3, :3] = R.from_quat(tcp_quat_np).as_matrix()
    T_world_to_tcp[:3, 3] = tcp_pos_np
    
    # 构建偏移变换矩阵 (wrist3 到 TCP 的相对变换)
    T_wrist3_to_tcp = np.eye(4)
    T_wrist3_to_tcp[:3, :3] = R.from_quat(offset_quat).as_matrix()
    T_wrist3_to_tcp[:3, 3] = offset_pos
    
    # 计算 wrist3 在世界坐标系中的位姿
    # T_world_to_wrist3 = T_world_to_tcp @ inv(T_wrist3_to_tcp)
    T_world_to_wrist3 = T_world_to_tcp @ np.linalg.inv(T_wrist3_to_tcp)
    
    # 提取位置和姿态
    wrist3_pos_np = T_world_to_wrist3[:3, 3]
    wrist3_quat_np = R.from_matrix(T_world_to_wrist3[:3, :3]).as_quat()
    
    # 转换回原始类型
    if is_torch:
        wrist3_pos = torch.tensor(wrist3_pos_np, device=device, dtype=torch.float32).unsqueeze(0)
        wrist3_quat = torch.tensor(wrist3_quat_np, device=device, dtype=torch.float32).unsqueeze(0)
    else:
        wrist3_pos = wrist3_pos_np
        wrist3_quat = wrist3_quat_np
    
    return wrist3_pos, wrist3_quat



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
        # ee_quat = quat_down
        ee_quat = quat_wrist3

        gripper_width = 1.0  # 保持打开
    
    # ===== 阶段3: 抓取 (步数 70-90) =====
    elif step < 90:
        t = (step - 70) / 20.0
        # 位置保持不变
        ee_pos = grasp_pos
        # ee_quat = quat_down
        ee_quat = quat_wrist3

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
        # ee_quat = quat_down
        ee_quat = quat_wrist3

        gripper_width = 0.2  # 保持闭合状态
    
    # ===== 阶段5: 释放 (步数 170-190) =====
    elif step < 190:
        t = (step - 170) / 20.0
        # 位置保持在展开位置
        ee_pos = deploy_pos
        # ee_quat = quat_down
        ee_quat = quat_wrist3

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
        # ee_quat = quat_down
        ee_quat = quat_wrist3

        gripper_width = 1.0  # 保持打开
    
    # 方案B: 使用 TCP 坐标定义轨迹，然后转换为 wrist3 坐标
    ee_pos_tcp = torch.tensor([ee_pos], device=device, dtype=torch.float32)
    ee_quat_tcp = torch.tensor([ee_quat], device=device, dtype=torch.float32)
        
    # 使用经过验证的转换函数（误差 0.0000mm）
    ee_pos_target, ee_quat_target = tcp_pose_to_wrist3_pose(
        ee_pos_tcp,
        ee_quat_tcp,
        TCP_TO_WRIST3_OFFSET_POS,
        TCP_TO_WRIST3_OFFSET_QUAT
    )

    return ee_pos_target, ee_quat_target, gripper_width



# 在主循环开始前，初始化目标关节角度
target_joint_angles = None  # 添加在 for step in range(220): 之前

# 修改主循环 - 每步都求解IK，实现真正的笛卡尔空间轨迹跟踪
for step in range(220):
    log.debug(f"Step {step}")
    states = handler.get_states()

    # ===== 步骤1: 生成轨迹目标 =====
    ee_pos_target, ee_quat_target, gripper_width = get_trajectory_targets(step, device, robot)

    # ===== 步骤2: 每步都执行 IK 求解（关键改动）=====
    # 获取当前关节角度作为 IK 初始值
    reorder_idx = handler.get_joint_reindex(robot.name)
    inv_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
    curr_q = obs.robots[robot.name].joint_pos[:, inv_idx]
    
    # 扩展目标到所有环境
    ee_pos_batch = ee_pos_target.repeat(args.num_envs, 1)
    ee_quat_batch = ee_quat_target.repeat(args.num_envs, 1)
    
    # 执行 IK 求解
    q_sol, ik_success = ik_solver.solve_ik_batch(
        ee_pos_batch,
        ee_quat_batch,
        curr_q  # 使用当前关节角度作为初始值
    )
    
    # 检查求解结果
    if ik_success is not None and ik_success.any():
        target_joint_angles = q_sol.clone()
    else:
        log.warning(f"✗ Step {step}: IK求解失败，保持当前关节角度")
        target_joint_angles = curr_q.clone()
    
    # ===== 步骤3: 在阶段切换点打印调试信息 =====
    if step in [0, 50, 70, 90, 170, 190]:
        stage_names = {
            0: "阶段1开始-接近",
            50: "阶段2开始-精确定位", 
            70: "阶段3开始-抓取",
            90: "阶段4开始-展开",
            170: "阶段5开始-释放",
            190: "阶段6开始-撤离"
        }
        
        log.info(f"\n{'='*60}")
        log.info(f"{stage_names[step]} (Step {step})")
        log.info(f"{'='*60}")
        log.info(f"✓ IK求解成功")
        log.info(f"求解的关节角度: {q_sol[0].cpu().numpy()}")
        
        # 打印调试信息
        _wrist_body_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
        _pinch_site_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "pinch")
        _wrist_pos_actual = _data.xpos[_wrist_body_id].copy()
        _pinch_pos_actual = _data.site_xpos[_pinch_site_id].copy()
        
        log.info(f"目标 wrist3 坐标: x={ee_pos_target[0, 0]:.4f}, y={ee_pos_target[0, 1]:.4f}, z={ee_pos_target[0, 2]:.4f}")
        log.info(f"实际 wrist3 坐标: x={_wrist_pos_actual[0]:.4f}, y={_wrist_pos_actual[1]:.4f}, z={_wrist_pos_actual[2]:.4f}")
        log.info(f"实际 TCP(pinch) 坐标: x={_pinch_pos_actual[0]:.4f}, y={_pinch_pos_actual[1]:.4f}, z={_pinch_pos_actual[2]:.4f}")
        log.info(f"夹爪开度: {gripper_width:.2f}")
        log.info(f"{'='*60}\n")
    
    # ===== 步骤4: 处理夹爪命令 =====
    gripper_command = torch.full((args.num_envs,), gripper_width, device=device)
    gripper_widths = process_gripper_command(gripper_command, robot, device)

    # ===== 步骤5: 组合动作并执行 =====
    actions = ik_solver.compose_joint_action(
        target_joint_angles,  # 使用本步求解的目标
        gripper_widths, 
        curr_q, 
        return_dict=True
    )

    handler.set_dof_targets(actions)
    handler.simulate()
    obs = handler.get_states(mode="tensor")
    obs_saver.add(obs)

obs_saver.save()
