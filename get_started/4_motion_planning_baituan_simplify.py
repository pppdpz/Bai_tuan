"""UR5e + Robotiq 2F-85 motion planning demo with inline MuJoCo scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import gc
import os
import warnings
from typing import Literal

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*os.fork().*")

import mujoco as mj
import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler
from scipy.spatial.transform import Rotation as R

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.utils.configclass import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler
from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver
from inline_scene_adapter import install_inline_scene_patches


@configclass
class BaituanSceneCfg(SceneCfg):
    def __post_init__(self) -> None:
        if self.mjx_mjcf_path is None:
            self.mjx_mjcf_path = self.mjcf_path


@configclass
class Args:
    robot: str = "ur5e_2f85"
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    num_envs: int = 1
    headless: bool = True
    solver: Literal["curobo", "pyroki"] = "pyroki"
    grasp_offset_x: float = 0.0
    grasp_offset_y: float = 0.0
    grasp_offset_z: float = 0.0
    use_normal_offset: bool = False
    normal_offset_dist: float = 0.01


# ============ 常量定义 ============
CFG_REL_IS_TCP_TO_EE = True
W3_TO_EE_BRIDGE_POS = np.array([0.1, 0.1, 0.0], dtype=np.float64)
W3_TO_EE_BRIDGE_RPY = np.array([np.pi / 2.0, -np.pi / 2.0, 0.0], dtype=np.float64)

FOLD_ANGLES = {
    "hinge_1": -1.5708, "hinge_2": 3.1416, "hinge_3": -3.1416, "hinge_4": 1.5708,
    "hinge_1_L": -1.5708, "hinge_2_L": 3.1416, "hinge_3_L": -3.1416, "hinge_4_L": 1.5708,
}

ROBOT_INIT_CONFIG = {
    "pos": torch.tensor([1.0, 0.0, 0.895]),
    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    "dof_pos": {
        "shoulder_pan_joint": 0.3, "elbow_joint": -1.6,
        "wrist_1_joint": 0.0, "wrist_2_joint": 0.0, "wrist_3_joint": 0.0,
        **{f"{side}_{joint}_joint": 0.0 
           for side in ["right", "left"] 
           for joint in ["driver", "coupler", "spring_link", "follower"]}
    }
}


# ============ 工具函数 ============
def smooth_interpolation(t):
    """S曲线插值"""
    return 3 * t**2 - 2 * t**3


def quat_xyzw_to_wxyz(quat_xyzw):
    """四元数格式转换 [x,y,z,w] -> [w,x,y,z]"""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def matrix_to_quat_wxyz(mat):
    """旋转矩阵转四元数 [w,x,y,z]"""
    return quat_xyzw_to_wxyz(R.from_matrix(mat).as_quat())


def build_transform_matrix(pos, quat_wxyz=None, rpy_xyz=None):
    """构建4x4变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    if quat_wxyz is not None:
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    elif rpy_xyz is not None:
        T[:3, :3] = R.from_euler("xyz", rpy_xyz).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def get_site_world_pos(data, site_id):
    """读取 MuJoCo site 世界坐标"""
    return data.site_xpos[site_id].copy() if site_id >= 0 else None


def get_panel_normal_vector(data, model, panel_body_name="panel_4_L"):
    """计算太阳能板法向量"""
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, panel_body_name)
    if body_id < 0:
        return np.array([0.0, 0.0, 1.0])
    body_xmat = data.xmat[body_id].reshape(3, 3).copy()
    normal = body_xmat[:, 2]  # 局部Z轴
    return normal / np.linalg.norm(normal)


def compute_grasp_point_with_offset(panel_tip_pos, offset_xyz, normal_vector=None, normal_offset=0.0):
    """计算带偏移的抓取点"""
    grasp_pos = np.asarray(panel_tip_pos).reshape(3) + np.asarray(offset_xyz).reshape(3)
    if normal_vector is not None and abs(normal_offset) > 1e-6:
        normal = np.asarray(normal_vector).reshape(3)
        grasp_pos += (normal / np.linalg.norm(normal)) * normal_offset
    return grasp_pos


def world_pose_to_base_pose(world_pos, world_quat, base_pos, base_quat):
    """世界坐标系位姿转base坐标系"""
    device = world_pos.device
    to_np = lambda t: t.detach().cpu().numpy().reshape(-1)
    
    T_world_target = build_transform_matrix(to_np(world_pos), quat_wxyz=to_np(world_quat))
    T_world_base = build_transform_matrix(to_np(base_pos), quat_wxyz=to_np(base_quat))
    T_base_target = np.linalg.inv(T_world_base) @ T_world_target
    
    base_pos_np = T_base_target[:3, 3]
    base_quat_wxyz = matrix_to_quat_wxyz(T_base_target[:3, :3])
    
    return (torch.tensor(base_pos_np, device=device, dtype=torch.float32).unsqueeze(0),
            torch.tensor(base_quat_wxyz, device=device, dtype=torch.float32).unsqueeze(0))


def measure_T_w3_tcp(data, wrist_body_id, pinch_site_id):
    """测量wrist3到TCP的变换矩阵"""
    T_world_w3 = build_transform_matrix(
        data.xpos[wrist_body_id].copy(),
        quat_wxyz=matrix_to_quat_wxyz(data.xmat[wrist_body_id].reshape(3, 3).copy())
    )
    T_world_tcp = build_transform_matrix(
        data.site_xpos[pinch_site_id].copy(),
        quat_wxyz=matrix_to_quat_wxyz(data.site_xmat[pinch_site_id].reshape(3, 3).copy())
    )
    return np.linalg.inv(T_world_w3) @ T_world_tcp


def tcp_pose_to_wrist3_pose(tcp_pos, tcp_quat, T_wrist3_tcp):
    """TCP位姿转wrist3位姿"""
    is_torch = isinstance(tcp_pos, torch.Tensor)
    device = tcp_pos.device if is_torch else None
    
    tcp_pos_np = tcp_pos.detach().cpu().numpy().reshape(3) if is_torch else np.array(tcp_pos).reshape(3)
    tcp_quat_np = tcp_quat.detach().cpu().numpy().reshape(4) if is_torch else np.array(tcp_quat).reshape(4)
    
    T_world_tcp = build_transform_matrix(tcp_pos_np, quat_wxyz=tcp_quat_np)
    T_world_w3 = T_world_tcp @ np.linalg.inv(T_wrist3_tcp)
    
    w3_pos = T_world_w3[:3, 3]
    w3_quat = matrix_to_quat_wxyz(T_world_w3[:3, :3])
    
    if is_torch:
        return (torch.tensor(w3_pos, device=device, dtype=torch.float32).unsqueeze(0),
                torch.tensor(w3_quat, device=device, dtype=torch.float32).unsqueeze(0))
    return w3_pos, w3_quat


def get_trajectory_targets(step, device, tcp_quat_wxyz, panel_tip_world_pos, args, model, data, T_wrist3_tcp):
    """生成轨迹目标点"""
    # 初始位置和路径点
    initial_pos = [0.2844, 0.1867, 1.1826]
    waypoint_pos = [0.3, 0.07, 1.20]
    approach_pos = [0.3, 0.24, 1.43]
    grasp_pos = [0.3, 0.24, 1.28]
    deploy_pos = [0.3, -0.30, 1.30]
    retract_pos = [0.3, -0.30, 1.40]
    
    # 动态计算抓取点
    if panel_tip_world_pos is not None:
        panel_normal = None
        if args.use_normal_offset:
            panel_normal = get_panel_normal_vector(data, model, "panel_4_L")
        
        offset_xyz = [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z]
        normal_dist = args.normal_offset_dist if args.use_normal_offset else 0.0
        
        grasp_pos_np = compute_grasp_point_with_offset(panel_tip_world_pos, offset_xyz, panel_normal, normal_dist)
        grasp_pos = grasp_pos_np.tolist()
        approach_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.15]
        deploy_pos = [grasp_pos[0], grasp_pos[1] - 0.54, grasp_pos[2] + 0.02]
        retract_pos = [deploy_pos[0], deploy_pos[1], deploy_pos[2] + 0.10]
    
    grasp_pos_inset = np.array(grasp_pos) + np.array([0.0, 0.04, 0.0])
    
    # 轨迹阶段定义
    stages = [
        (30, initial_pos, waypoint_pos, 1.0),
        (50, waypoint_pos, approach_pos, 1.0),
        (70, approach_pos, grasp_pos, 1.0),
        (90, grasp_pos_inset, grasp_pos_inset, lambda t: 1.0 - t),
        (120, grasp_pos_inset, grasp_pos_inset, 0.0),
        (320, grasp_pos, deploy_pos, 0.0, True),  # 使用S曲线
        (340, deploy_pos, deploy_pos, lambda t: t),
        (370, deploy_pos, deploy_pos, 1.0),
        (400, deploy_pos, retract_pos, 1.0),
    ]
    
    # 计算当前阶段
    ee_pos, gripper_width = retract_pos, 1.0
    prev_end = 0
    
    for stage_end, *stage_params in stages:
        if step < stage_end:
            start_pos, end_pos = stage_params[0], stage_params[1]
            gripper = stage_params[2]
            use_smooth = len(stage_params) > 3 and stage_params[3]
            
            t = (step - prev_end) / (stage_end - prev_end)
            if use_smooth:
                t = smooth_interpolation(t)
            
            ee_pos = np.array(start_pos) + t * (np.array(end_pos) - np.array(start_pos))
            gripper_width = gripper(t) if callable(gripper) else gripper
            break
        prev_end = stage_end
    
    # 转换为wrist3坐标
    ee_pos_tcp = torch.tensor([ee_pos], device=device, dtype=torch.float32)
    ee_quat_tcp = torch.tensor([tcp_quat_wxyz], device=device, dtype=torch.float32)
    
    ee_pos_target, ee_quat_target = tcp_pose_to_wrist3_pose(ee_pos_tcp, ee_quat_tcp, T_wrist3_tcp)
    
    return ee_pos_target, ee_quat_target, gripper_width

# ============ 主程序 ============
def main():
    args = tyro.cli(Args)
    install_inline_scene_patches()
    
    # 场景配置
    scenario = ScenarioCfg(
        scene=BaituanSceneCfg(name="baituan_satellite", mjcf_path="asset_baituan/example_scene_y.xml"),
        robots=["ur5e_2f85"],
        objects=[],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
        gravity=(0.0, 0.0, 0.0),
    )
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(4.0, -4.0, 4.0), look_at=(0.0, 0.0, 0.0))]
    
    handler = get_handler(scenario)
    robot = scenario.robots[0]
    ik_solver = setup_ik_solver(robot, args.solver)
    
    # 初始化状态
    robot_dict = {"ur5e_2f85": ROBOT_INIT_CONFIG}
    init_states = [{"objects": {}, "robots": robot_dict} for _ in range(args.num_envs)]
    handler.set_states(init_states)
    
    # MuJoCo物理引擎
    model, data = handler.physics.model.ptr, handler.physics.data.ptr
    
    # 折叠太阳翼
    for jname, angle in FOLD_ANGLES.items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle
    mj.mj_forward(model, data)
    
    # 获取关键ID
    pinch_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pinch")
    wrist_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
    panel_tip_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "panel4_tip_L")
    
    # 读取真实TCP姿态
    if pinch_site_id >= 0 and wrist_body_id >= 0:
        pinch_xmat = data.site_xmat[pinch_site_id].reshape(3, 3)
        REAL_TCP_DOWN_QUAT_WXYZ = matrix_to_quat_wxyz(pinch_xmat)
    else:
        REAL_TCP_DOWN_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
    
    # 测量TCP变换
    if pinch_site_id >= 0 and wrist_body_id >= 0:
        T_W3_TCP_MEASURED = measure_T_w3_tcp(data, wrist_body_id, pinch_site_id)
        if np.linalg.norm(T_W3_TCP_MEASURED[:3, 3]) < 0.05:
            raise RuntimeError(f"TCP测量异常：偏移量过小")
    else:
        raise RuntimeError("无法执行TCP链路自检")
    
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_base_pos_world = ROBOT_INIT_CONFIG["pos"].to(device).unsqueeze(0)
    robot_base_quat_world = ROBOT_INIT_CONFIG["rot"].to(device).unsqueeze(0)
    
    os.makedirs("get_started/output", exist_ok=True)
    obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_baituan_{args.sim}.mp4")
    obs = handler.get_states(mode="tensor")
    obs_saver.add(obs)
    
    # 主循环
    for step in range(450):
        panel_tip_world_pos = get_site_world_pos(data, panel_tip_site_id)
        ee_pos_target, ee_quat_target, gripper_width = get_trajectory_targets(
            step, device, REAL_TCP_DOWN_QUAT_WXYZ, panel_tip_world_pos, args, model, data, T_W3_TCP_MEASURED
        )
        
        # IK求解
        reorder_idx = handler.get_joint_reindex(robot.name)
        inv_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        curr_q = obs.robots[robot.name].joint_pos[:, inv_idx]
        
        ee_pos_batch = ee_pos_target.repeat(args.num_envs, 1)
        ee_quat_batch = ee_quat_target.repeat(args.num_envs, 1)
        
        ee_pos_base, ee_quat_base = world_pose_to_base_pose(
            ee_pos_batch[0], ee_quat_batch[0], robot_base_pos_world[0], robot_base_quat_world[0]
        )
        
        ee_pos_base_batch = ee_pos_base.repeat(args.num_envs, 1)
        ee_quat_base_batch = ee_quat_base.repeat(args.num_envs, 1)
        
        # URDF坐标系转换
        curr_q_urdf = curr_q.clone()
        curr_q_urdf[:, 1] -= np.pi / 2
        curr_q_urdf[:, 3] -= np.pi / 2
        
        q_sol, ik_success = ik_solver.solve_ik_batch(ee_pos_base_batch, ee_quat_base_batch, curr_q_urdf)
        
        # 转回MJCF坐标系
        if ik_success is not None and ik_success.any():
            q_sol_mjcf = q_sol.clone()
            q_sol_mjcf[:, 1] += np.pi / 2
            q_sol_mjcf[:, 3] += np.pi / 2
            target_joint_angles = q_sol_mjcf
        else:
            target_joint_angles = curr_q.clone()
        
        # 执行动作
        gripper_command = torch.full((args.num_envs,), gripper_width, device=device)
        gripper_widths = process_gripper_command(gripper_command, robot, device)
        actions = ik_solver.compose_joint_action(target_joint_angles, gripper_widths, curr_q, return_dict=True)
        
        handler.set_dof_targets(actions)
        handler.simulate()
        obs = handler.get_states(mode="tensor")
        obs_saver.add(obs)
    
    obs_saver.save()
    
    # 清理
    if hasattr(handler, "close"):
        handler.close()
    del obs, handler, obs_saver, model, data
    gc.collect()


if __name__ == "__main__":
    main()
