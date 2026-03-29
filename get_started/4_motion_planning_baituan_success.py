# 4_motion_planning_baituan.py (精简后)
"""UR5e + Robotiq 2F-85 motion planning demo with inline MuJoCo scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*os.fork().*")

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

import numpy as np
from scipy.spatial.transform import Rotation as R


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
    
    # ===== 新增：抓取点位偏移参数 =====
    grasp_offset_x: float = 0.0      # X方向偏移（米）
    grasp_offset_y: float = 0.0      # Y方向偏移（米）
    grasp_offset_z: float = -0.0    # Z方向偏移（米），负值表示向下
    use_normal_offset: bool = False  # 是否使用法向偏移（默认禁用）
    normal_offset_dist: float = 0.01 # 沿法向偏移距离（米）

    def __post_init__(self):
        pass


args = tyro.cli(Args)

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
            "shoulder_pan_joint": 0.3,
            # "shoulder_pan_joint": 3.14,
            # "shoulder_lift_joint": -0.5,
            # "shoulder_lift_joint": -1.57,
            "elbow_joint": -1.6,               
            # "elbow_joint": 1.57,               
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,             
            "wrist_3_joint": 0.0,             
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

# 更新物理状态
mj.mj_forward(_model, _data)

# ===== 诊断：读取初始状态下的真实姿态 =====
pinch_site_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "pinch")
wrist_body_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
panel_tip_site_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "panel4_tip_L")

if pinch_site_id >= 0 and wrist_body_id >= 0:
    # 读取姿态矩阵并转换为四元数
    pinch_xmat = _data.site_xmat[pinch_site_id].reshape(3, 3)
    wrist_xmat = _data.xmat[wrist_body_id].reshape(3, 3)
    
    # scipy 输出 [x,y,z,w]，转换为 [w,x,y,z]
    pinch_quat_scipy = R.from_matrix(pinch_xmat).as_quat()  # [x,y,z,w]
    pinch_quat_wxyz = np.array([pinch_quat_scipy[3], pinch_quat_scipy[0], 
                                pinch_quat_scipy[1], pinch_quat_scipy[2]])  # [w,x,y,z]
    
    wrist_quat_scipy = R.from_matrix(wrist_xmat).as_quat()  # [x,y,z,w]
    wrist_quat_wxyz = np.array([wrist_quat_scipy[3], wrist_quat_scipy[0], 
                                wrist_quat_scipy[1], wrist_quat_scipy[2]])  # [w,x,y,z]
    
    pinch_pos = _data.site_xpos[pinch_site_id]
    wrist_pos = _data.xpos[wrist_body_id]
    
    # 保存真实的 TCP 朝下姿态供后续使用 [w,x,y,z] 格式
    REAL_TCP_DOWN_QUAT_WXYZ = pinch_quat_wxyz.copy()
else:
    REAL_TCP_DOWN_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])  # 回退值 [w,x,y,z]

if panel_tip_site_id >= 0:
    _panel_tip_pos = _data.site_xpos[panel_tip_site_id]



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

def get_site_world_pos(_data, site_id):
    """读取 MuJoCo site 的世界坐标，失败时返回 None。"""
    if site_id < 0:
        return None
    return _data.site_xpos[site_id].copy()


def get_panel_normal_vector(_data, _model, panel_body_name="panel_4_L"):
    """
    计算太阳能板的法向量（指向外侧）
    
    Args:
        _data: MuJoCo data
        _model: MuJoCo model
        panel_body_name: 太阳能板 body 名称
    
    Returns:
        normal_vector: 单位法向量 [3] (numpy array)，失败返回 [0, 0, 1]
    """
    body_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_BODY, panel_body_name)
    if body_id < 0:
        return np.array([0.0, 0.0, 1.0])
    
    # 读取 body 的旋转矩阵（3x3）
    body_xmat = _data.xmat[body_id].reshape(3, 3).copy()
    
    # 假设太阳能板的局部 Z 轴是法向（根据实际模型调整）
    # 如果是 X 轴，改为 body_xmat[:, 0]
    # 如果是 Y 轴，改为 body_xmat[:, 1]
    normal_vector = body_xmat[:, 2]  # 局部 Z 轴
    
    # 归一化
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    return normal_vector


def compute_grasp_point_with_offset(
    panel_tip_pos, 
    offset_xyz, 
    normal_vector=None, 
    normal_offset=0.0
):
    """
    计算带偏移的抓取点
    
    Args:
        panel_tip_pos: panel_tip 的世界坐标 [3]
        offset_xyz: 直角坐标系偏移 [x, y, z]
        normal_vector: 太阳能板法向量 [3]（可选）
        normal_offset: 沿法向的偏移距离（正值表示沿法向外侧）
    
    Returns:
        grasp_pos: 最终抓取点 [3]
    """
    panel_tip = np.asarray(panel_tip_pos).reshape(3)
    offset = np.asarray(offset_xyz).reshape(3)
    
    # 基础偏移：直角坐标系
    grasp_pos = panel_tip + offset
    
    # 法向偏移（如果提供）
    if normal_vector is not None and abs(normal_offset) > 1e-6:
        normal = np.asarray(normal_vector).reshape(3)
        normal = normal / np.linalg.norm(normal)  # 确保归一化
        grasp_pos += normal * normal_offset
    
    return grasp_pos


def world_pose_to_base_pose(world_pos, world_quat, base_pos, base_quat):
    """
    将世界坐标系位姿转换为 base 坐标系位姿
    
    Args:
        world_pos: 世界坐标系位置 [3] (torch tensor)
        world_quat: 世界坐标系姿态 [4] (w,x,y,z) (torch tensor)
        base_pos: base 在世界坐标系中的位置 [3] (torch tensor)
        base_quat: base 在世界坐标系中的姿态 [4] (w,x,y,z) (torch tensor)
    
    Returns:
        base_pos_target: base 坐标系位置 [1, 3] (torch tensor)
        base_quat_target: base 坐标系姿态 [1, 4] (w,x,y,z) (torch tensor)
    """
    device = world_pos.device
    world_pos_np = world_pos.detach().cpu().numpy().reshape(3)
    world_quat_np = world_quat.detach().cpu().numpy().reshape(4)  # [w,x,y,z]
    base_pos_np = base_pos.detach().cpu().numpy().reshape(3)
    base_quat_np = base_quat.detach().cpu().numpy().reshape(4)    # [w,x,y,z]

    # 构建世界到目标的变换矩阵
    T_world_to_target = _T_from_pos_quat_wxyz(world_pos_np, world_quat_np)

    # 构建世界到 base 的变换矩阵
    T_world_to_base = _T_from_pos_quat_wxyz(base_pos_np, base_quat_np)

    # 计算 base 到目标的变换
    T_base_to_target = np.linalg.inv(T_world_to_base) @ T_world_to_target

    # 提取位置和姿态
    base_pos_target_np = T_base_to_target[:3, 3]
    # 从旋转矩阵转换为四元数 [x,y,z,w] (scipy格式)
    base_quat_target_xyzw = R.from_matrix(T_base_to_target[:3, :3]).as_quat()
    # 转换为 [w,x,y,z] 格式
    base_quat_target_wxyz = np.array([base_quat_target_xyzw[3], base_quat_target_xyzw[0], 
                                       base_quat_target_xyzw[1], base_quat_target_xyzw[2]])

    base_pos_target = torch.tensor(base_pos_target_np, device=device, dtype=torch.float32).unsqueeze(0)
    base_quat_target = torch.tensor(base_quat_target_wxyz, device=device, dtype=torch.float32).unsqueeze(0)

    return base_pos_target, base_quat_target


# -----------------------------------------------------------------------------
# TCP/EE unified transform source
# 统一采用 T_A_B 记号: B 坐标系 -> A 坐标系
# 统一主语义: T_ee_tcp
# -----------------------------------------------------------------------------
CFG_REL_IS_TCP_TO_EE = True  # docs 语义: curobo_tcp_rel_* 是 TCP -> EE

# 桥接常量（wrist_3_link -> cuRobo ee_link），建议通过离线脚本标定后固化
W3_TO_EE_BRIDGE_POS = np.array([0.1, 0.1, 0.0], dtype=np.float64)
W3_TO_EE_BRIDGE_RPY = np.array([np.pi / 2.0, -np.pi / 2.0, 0.0], dtype=np.float64)


def _T_from_pos_quat_wxyz(pos3, quat_wxyz):
    """构建变换矩阵，四元数格式为 [w,x,y,z]"""
    T = np.eye(4, dtype=np.float64)
    # 转换为 scipy 格式 [x,y,z,w]
    quat_wxyz_arr = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat_wxyz_arr[1], quat_wxyz_arr[2], quat_wxyz_arr[3], quat_wxyz_arr[0]])
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos3, dtype=np.float64).reshape(3)
    return T


def _T_from_pos_rpy_xyz(pos3, rpy_xyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(rpy_xyz, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(pos3, dtype=np.float64).reshape(3)
    return T


def _cfg_T_ee_tcp(robot_cfg):
    T_cfg = _T_from_pos_rpy_xyz(robot_cfg.curobo_tcp_rel_pos, robot_cfg.curobo_tcp_rel_rot)
    return np.linalg.inv(T_cfg) if CFG_REL_IS_TCP_TO_EE else T_cfg


def _bridge_T_w3_ee():
    return _T_from_pos_rpy_xyz(W3_TO_EE_BRIDGE_POS, W3_TO_EE_BRIDGE_RPY)


def _T_w3_tcp_from_cfg(robot_cfg):
    return _bridge_T_w3_ee() @ _cfg_T_ee_tcp(robot_cfg)


def _measure_T_w3_tcp(data, wrist_body_id, pinch_site_id):
    """测量 wrist3 到 TCP 的变换矩阵"""
    wrist_pos = data.xpos[wrist_body_id].copy()
    wrist_xmat = data.xmat[wrist_body_id].reshape(3, 3).copy()
    tcp_pos = data.site_xpos[pinch_site_id].copy()
    tcp_xmat = data.site_xmat[pinch_site_id].reshape(3, 3).copy()
    T_world_w3 = np.eye(4, dtype=np.float64)
    T_world_w3[:3, :3] = wrist_xmat
    T_world_w3[:3, 3] = wrist_pos
    T_world_tcp = np.eye(4, dtype=np.float64)
    T_world_tcp[:3, :3] = tcp_xmat
    T_world_tcp[:3, 3] = tcp_pos
    return np.linalg.inv(T_world_w3) @ T_world_tcp


def tcp_pose_to_wrist3_pose(tcp_pos, tcp_quat, robot_cfg):
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
        tcp_quat: TCP 目标姿态四元数 [4] 或 [1, 4] (w,x,y,z)
        robot_cfg: 机器人配置（读取 curobo_tcp_rel_pos/rot 并叠加桥接）
    
    Returns:
        wrist3_pos: wrist_3_link 位置 (与输入类型相同)
        wrist3_quat: wrist_3_link 姿态 [w,x,y,z] (与输入类型相同)
    """
    # 转换为 numpy 进行计算
    is_torch = isinstance(tcp_pos, torch.Tensor)
    if is_torch:
        device = tcp_pos.device
        tcp_pos_np = tcp_pos.detach().cpu().numpy().reshape(3)
        tcp_quat_np = tcp_quat.detach().cpu().numpy().reshape(4)  # [w,x,y,z]
    else:
        tcp_pos_np = np.array(tcp_pos).reshape(3)
        tcp_quat_np = np.array(tcp_quat).reshape(4)  # [w,x,y,z]
    
    # 构建 TCP 变换矩阵（世界坐标系到 TCP）
    T_world_to_tcp = _T_from_pos_quat_wxyz(tcp_pos_np, tcp_quat_np)

    # 从统一配置链生成 (wrist3 -> tcp)
    # T_wrist3_to_tcp = _T_w3_tcp_from_cfg(robot_cfg)
    
    # 改为：直接用 MuJoCo 实测值
    T_wrist3_to_tcp = T_W3_TCP_MEASURED

    # 计算 wrist3 在世界坐标系中的位姿
    # T_world_to_wrist3 = T_world_to_tcp @ inv(T_wrist3_to_tcp)
    T_world_to_wrist3 = T_world_to_tcp @ np.linalg.inv(T_wrist3_to_tcp)
    
    # 提取位置和姿态
    wrist3_pos_np = T_world_to_wrist3[:3, 3]
    # 从旋转矩阵转换为四元数 [x,y,z,w] (scipy格式)
    wrist3_quat_xyzw = R.from_matrix(T_world_to_wrist3[:3, :3]).as_quat()
    # 转换为 [w,x,y,z] 格式
    wrist3_quat_wxyz = np.array([wrist3_quat_xyzw[3], wrist3_quat_xyzw[0], 
                                  wrist3_quat_xyzw[1], wrist3_quat_xyzw[2]])

    # 转换回原始类型，使用 [w,x,y,z] 格式
    if is_torch:
        wrist3_pos = torch.tensor(wrist3_pos_np, device=device, dtype=torch.float32).unsqueeze(0)
        wrist3_quat = torch.tensor(wrist3_quat_wxyz, device=device, dtype=torch.float32).unsqueeze(0)
    else:
        wrist3_pos = wrist3_pos_np
        wrist3_quat = wrist3_quat_wxyz

    return wrist3_pos, wrist3_quat




def get_trajectory_targets(step, device, robot_cfg, tcp_quat_wxyz, panel_tip_world_pos=None):
    """
    生成轨迹目标点
    
    Args:
        step: 当前步数 [0, 220)
        device: torch device
        robot_cfg: 机器人配置（读取 TCP/EE 统一定义）
        tcp_quat_wxyz: TCP 朝下的真实姿态 [w,x,y,z] (从 MuJoCo 读取并转换)
        panel_tip_world_pos: panel4_tip_L 的世界坐标位置
    
    Returns:
        ee_pos_target: wrist3 目标位置 [1, 3]
        ee_quat_target: wrist3 目标姿态 [1, 4] (w,x,y,z)
        gripper_width: 夹爪开度 [0, 1]
    """

    # ===== 关键修正：从 MuJoCo 读取的初始 TCP 位置作为起点 =====
    # 这个位置是在世界坐标系中测量的真实位置
    # 初始 TCP(pinch) 位置: [0.2844, 0.1867, 1.1826] (从启动诊断中读取)
    initial_pos_pinch = [0.2844, 0.1867, 1.1826]

    # 中间安全点（世界坐标系）
    # waypoint_pos_pinch = [0.3, 0.17, 1.10]
    waypoint_pos_pinch = [0.3, 0.07, 1.20]

    # 默认回退目标（世界坐标系，site 读取失败时使用）
    # 注意：panel4_tip_L 初始位置是 [0.3000, 0.2400, 1.2800]
    approach_pos_pinch = [0.3, 0.24, 1.43]   # 在抓取点上方 15cm
    grasp_pos_pinch = [0.3, 0.24, 1.28]      # 对准 panel4_tip_L
    # grasp_pos_pinch = [0.3, 0.4, 1.28]      # 对准 panel4_tip_L
    deploy_pos_pinch = [0.3, -0.30, 1.30]    # 沿 -Y 方向拉动 54cm
    retract_pos_pinch = [0.3, -0.30, 1.40]   # 上抬 10cm


    # ===== 修改：使用 panel4_tip_L site 动态生成目标（带偏移） =====
    if panel_tip_world_pos is not None:
        panel_tip = np.asarray(panel_tip_world_pos).reshape(3)

        # ===== 新增：读取太阳能板法向量（仅在启用时） =====
        panel_normal = None
        if hasattr(get_trajectory_targets, '_model') and hasattr(get_trajectory_targets, '_data'):
            if hasattr(get_trajectory_targets, '_args') and get_trajectory_targets._args.use_normal_offset:
                panel_normal = get_panel_normal_vector(
                    get_trajectory_targets._data, 
                    get_trajectory_targets._model, 
                    panel_body_name="panel_4_L"  # 根据实际模型调整
                )
        
        # ===== 新增：从 args 读取偏移参数 =====
        if hasattr(get_trajectory_targets, '_args'):
            _args = get_trajectory_targets._args
            offset_xyz = [_args.grasp_offset_x, _args.grasp_offset_y, _args.grasp_offset_z]
            use_normal = _args.use_normal_offset
            normal_dist = _args.normal_offset_dist if use_normal else 0.0
        else:
            # 回退默认值
            offset_xyz = [0.0, 0.0, -0.02]
            normal_dist = 0.0
            panel_normal = None
        
        # ===== 核心修改：计算带偏移的抓取点 =====
        grasp_pos_pinch_np = compute_grasp_point_with_offset(
            panel_tip, 
            offset_xyz, 
            normal_vector=panel_normal,
            normal_offset=normal_dist
        )
        
        grasp_pos_pinch = [
            float(grasp_pos_pinch_np[0]),
            float(grasp_pos_pinch_np[1]),
            float(grasp_pos_pinch_np[2]),
        ]

        # approach: 在抓取点上方保留安全高度
        approach_pos_pinch = [
            grasp_pos_pinch[0],
            grasp_pos_pinch[1],
            grasp_pos_pinch[2] + 0.15,
        ]

        # deploy: 保持你原来的相对拉动趋势（主要沿 -Y）
        deploy_pos_pinch = [
            grasp_pos_pinch[0] + 0.0,
            grasp_pos_pinch[1] - 0.54,
            grasp_pos_pinch[2] + 0.02,
        ]

        # retract: 展开后上抬，减少干涉风险
        retract_pos_pinch = [
            deploy_pos_pinch[0],
            deploy_pos_pinch[1],
            deploy_pos_pinch[2] + 0.10,
        ]

    # 方案B: 直接使用 pinch/TCP 坐标定义轨迹
    # 在函数末尾通过 ee_pose_from_tcp_pose() 转换为 wrist_3_link 坐标
    initial_pos = initial_pos_pinch
    approach_pos = approach_pos_pinch
    grasp_pos = grasp_pos_pinch
    deploy_pos = deploy_pos_pinch
    retract_pos = retract_pos_pinch

    inset = 0.04  # 2cm
    # 假设“向内”是世界系 -y；如果方向反了就改成 +inset
    grasp_pos_inset = grasp_pos + np.array([0.0, inset, 0.0])

    # ===== 阶段1a: 移动到中间点 (步数 0-30) =====
    if step < 30:
        t = step / 30.0
        # 从初始位置线性插值到中间点
        ee_pos = [
            initial_pos[0] + t * (waypoint_pos_pinch[0] - initial_pos[0]),
            initial_pos[1] + t * (waypoint_pos_pinch[1] - initial_pos[1]),
            initial_pos[2] + t * (waypoint_pos_pinch[2] - initial_pos[2])
        ]
        ee_quat = tcp_quat_wxyz
        gripper_width = 1.0

    # ===== 阶段1b: 从中间点到接近位置 (步数 30-50) =====
    elif step < 50:
        t = (step - 30) / 20.0
        # 从中间点线性插值到接近位置
        ee_pos = [
            waypoint_pos_pinch[0] + t * (approach_pos[0] - waypoint_pos_pinch[0]),
            waypoint_pos_pinch[1] + t * (approach_pos[1] - waypoint_pos_pinch[1]),
            waypoint_pos_pinch[2] + t * (approach_pos[2] - waypoint_pos_pinch[2])
        ]
        ee_quat = tcp_quat_wxyz
        gripper_width = 1.0

    # ===== 阶段2: 精确定位 (步数 50-70) =====
    elif step < 70:
        t = (step - 50) / 20.0
        # 缓慢下降到抓取位置
        ee_pos = [
            approach_pos[0] + t * (grasp_pos[0] - approach_pos[0]),
            approach_pos[1] + t * (grasp_pos[1] - approach_pos[1]),
            approach_pos[2] + t * (grasp_pos[2] - approach_pos[2])
        ]
        ee_quat = tcp_quat_wxyz  # 使用真实的 TCP 姿态 [w,x,y,z]

        gripper_width = 1.0  # 保持打开    

    # ===== 阶段3: 抓取 (70-90) =====
    elif step < 90:
        t = (step - 70) / 20.0
        ee_pos = grasp_pos_inset
        ee_quat = tcp_quat_wxyz
        gripper_width = 1.0 - t * 1.0

    # ===== 阶段3.5: 抓取后稳固保持 (90-120) =====
    elif step < 120:
        ee_pos = grasp_pos_inset
        ee_quat = tcp_quat_wxyz
        gripper_width = 0.0


    # ===== 阶段4: 展开 (步数 120-320) ⭐ 关键阶段 =====
    # 将原来的80步延长到200步，放慢展开速度
    elif step < 320:
        t = (step - 120) / 200.0
        # 使用S曲线插值，提供平滑的运动
        t_smooth = smooth_interpolation(t)
        # 从抓取位置平滑移动到展开位置
        ee_pos = [
            grasp_pos[0] + t_smooth * (deploy_pos[0] - grasp_pos[0]),
            grasp_pos[1] + t_smooth * (deploy_pos[1] - grasp_pos[1]),
            grasp_pos[2] + t_smooth * (deploy_pos[2] - grasp_pos[2])
        ]
        ee_quat = tcp_quat_wxyz  # 使用真实的 TCP 姿态 [w,x,y,z]

        # gripper_width = 0.2  # 保持闭合状态
        gripper_width = 0.0  # 保持闭合状态
    
    # ===== 阶段5: 释放 (步数 320-340) =====
    elif step < 340:
        t = (step - 320) / 20.0
        # 位置保持在展开位置
        ee_pos = deploy_pos
        ee_quat = tcp_quat_wxyz  # 使用真实的 TCP 姿态 [w,x,y,z]

        # 夹爪逐渐打开，从0.0到1.0
        # gripper_width = 0.2 + t * 0.8
        gripper_width = 0.0 + t * 1.0
        
    # ===== 新增：阶段5.5: 释放后稳固保持 (步数 340-370) =====
    elif step < 370:
        # 等待手指完全离物
        ee_pos = deploy_pos
        ee_quat = tcp_quat_wxyz
        gripper_width = 1.0
    
    # ===== 阶段6: 撤离 (步数 370-400) =====
    elif step < 400:
        t = (step - 370) / 30.0
        # 从展开位置移动到撤离位置
        ee_pos = [
            deploy_pos[0] + t * (retract_pos[0] - deploy_pos[0]),
            deploy_pos[1] + t * (retract_pos[1] - deploy_pos[1]),
            deploy_pos[2] + t * (retract_pos[2] - deploy_pos[2])
        ]
        ee_quat = tcp_quat_wxyz  # 使用真实的 TCP 姿态 [w,x,y,z]

        gripper_width = 1.0  # 保持打开
    else:
        # 保持撤离位置
        ee_pos = retract_pos
        ee_quat = tcp_quat_wxyz
        gripper_width = 1.0
    
    # 方案B: 使用 TCP 坐标定义轨迹，然后转换为 wrist3 坐标
    ee_pos_tcp = torch.tensor([ee_pos], device=device, dtype=torch.float32)
    ee_quat_tcp = torch.tensor([ee_quat], device=device, dtype=torch.float32)  # [w,x,y,z] 格式

    # 在 tcp_pose_to_wrist3_pose 调用前后
    ee_pos_tcp_debug = ee_pos_tcp.clone()   # 转换前，真正的 TCP 目标
    ee_quat_tcp_debug = ee_quat_tcp.clone()

    # 使用经过验证的转换函数（误差 0.0000mm）
    ee_pos_target, ee_quat_target = tcp_pose_to_wrist3_pose(
        ee_pos_tcp,
        ee_quat_tcp,
        robot_cfg,
    )

    # 加在这里
    if step in (20, 40, 70):
        print(f"  [内部] TCP目标(真正) pos : {ee_pos_tcp_debug.detach().cpu().numpy().flatten()}")
        print(f"  [内部] wrist3目标(转换后): {ee_pos_target.detach().cpu().numpy().flatten()}")
        print(f"  [内部] 两者差值          : {(ee_pos_target - ee_pos_tcp_debug).detach().cpu().numpy().flatten()}")

    return ee_pos_target, ee_quat_target, gripper_width


# ===== 获取机器人 base_link 在世界坐标系中的位姿 =====
# 这个位姿在整个运动过程中保持不变（机器人底座固定）
# 从 robot_dict 中读取，与初始化时设置的值一致
robot_base_pos_world = robot_dict["ur5e_2f85"]["pos"]  # torch.tensor([1.0, 0.0, 0.895])
robot_base_quat_world = robot_dict["ur5e_2f85"]["rot"]  # torch.tensor([1.0, 0.0, 0.0, 0.0])

# 【修改这里】：专门给 IK 传入转了180度的虚拟 Base 四元数
# 抵消 URDF 和 MJCF 的内部原点反差
# robot_base_quat_world = torch.tensor([0.0, 0.0, 0.0, 1.0]) 

# 确保在正确的设备上
robot_base_pos_world = robot_base_pos_world.to(device).unsqueeze(0)  # [1, 3]
robot_base_quat_world = robot_base_quat_world.to(device).unsqueeze(0)  # [1, 4] (w,x,y,z)



# ===== 启动时一次性自检：测量链路 vs 配置链路 =====
if pinch_site_id >= 0 and wrist_body_id >= 0:
    T_W3_TCP_MEASURED = _measure_T_w3_tcp(_data, wrist_body_id, pinch_site_id)
    pos_offset = np.linalg.norm(T_W3_TCP_MEASURED[:3, 3])
    if pos_offset < 0.05:  # pinch 到 wrist3 至少应该有 5cm 偏移
        raise RuntimeError(f"TCP测量异常：偏移量 {pos_offset*1000:.1f}mm 过小，请检查 site/body ID")

else:
    raise RuntimeError("无法执行 TCP链路自检：缺少 pinch site 或 wrist_3_link body")



# 在主循环开始前，初始化目标关节角度
target_joint_angles = None  # 添加在 for step in range(220): 之前

# ===== 新增：将参数传递给轨迹生成函数 =====
# 通过函数属性传递，避免修改函数签名
get_trajectory_targets._args = args
get_trajectory_targets._model = _model
get_trajectory_targets._data = _data

# ============================================================
# 诊断打印：坐标系与变换链路验证
# ============================================================
import mujoco as mj
from scipy.spatial.transform import Rotation as R

print("\n" + "="*70)
print("坐标系诊断报告")
print("="*70)

# --- (A) MuJoCo 世界坐标系中的关键位置 ---
print("\n--- (A) MuJoCo 世界坐标：各关键 body/site 的绝对位姿 ---")

# A1: 机器人底座 (ur5e_base)
base_body_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_BODY, "ur5e_base")
if base_body_id >= 0:
    base_pos_mj = _data.xpos[base_body_id].copy()
    base_mat_mj = _data.xmat[base_body_id].reshape(3, 3).copy()
    base_quat_xyzw = R.from_matrix(base_mat_mj).as_quat()
    base_quat_wxyz = [base_quat_xyzw[3], base_quat_xyzw[0], base_quat_xyzw[1], base_quat_xyzw[2]]
    print(f"  ur5e_base  pos(world): {base_pos_mj}")
    print(f"  ur5e_base quat(wxyz): {base_quat_wxyz}")
else:
    print("  [WARN] ur5e_base body not found")

# A2: wrist_3_link
if wrist_body_id >= 0:
    w3_pos = _data.xpos[wrist_body_id].copy()
    w3_mat = _data.xmat[wrist_body_id].reshape(3, 3).copy()
    w3_quat_xyzw = R.from_matrix(w3_mat).as_quat()
    w3_quat_wxyz = [w3_quat_xyzw[3], w3_quat_xyzw[0], w3_quat_xyzw[1], w3_quat_xyzw[2]]
    print(f"  wrist_3_link  pos(world): {w3_pos}")
    print(f"  wrist_3_link quat(wxyz): {w3_quat_wxyz}")

# A3: flange
flange_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_BODY, "flange")
if flange_id >= 0:
    fl_pos = _data.xpos[flange_id].copy()
    fl_mat = _data.xmat[flange_id].reshape(3, 3).copy()
    fl_quat_xyzw = R.from_matrix(fl_mat).as_quat()
    fl_quat_wxyz = [fl_quat_xyzw[3], fl_quat_xyzw[0], fl_quat_xyzw[1], fl_quat_xyzw[2]]
    print(f"  flange        pos(world): {fl_pos}")
    print(f"  flange       quat(wxyz): {fl_quat_wxyz}")

# A4: pinch site (TCP)
if pinch_site_id >= 0:
    tcp_pos_mj = _data.site_xpos[pinch_site_id].copy()
    tcp_mat_mj = _data.site_xmat[pinch_site_id].reshape(3, 3).copy()
    tcp_quat_xyzw = R.from_matrix(tcp_mat_mj).as_quat()
    tcp_quat_wxyz = [tcp_quat_xyzw[3], tcp_quat_xyzw[0], tcp_quat_xyzw[1], tcp_quat_xyzw[2]]
    print(f"  pinch(TCP)    pos(world): {tcp_pos_mj}")
    print(f"  pinch(TCP)   quat(wxyz): {tcp_quat_wxyz}")

# A5: panel4_tip_L site
if panel_tip_site_id >= 0:
    pt_pos = _data.site_xpos[panel_tip_site_id].copy()
    print(f"  panel4_tip_L  pos(world): {pt_pos}")

# A6: tcp site (flange 上的 tcp site，不是 pinch)
tcp_site_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "tcp")
if tcp_site_id >= 0:
    tcp_site_pos = _data.site_xpos[tcp_site_id].copy()
    print(f"  tcp site      pos(world): {tcp_site_pos}")


# --- (B) 相对变换：MuJoCo 实测 ---
print("\n--- (B) MuJoCo 实测的相对变换 ---")

if wrist_body_id >= 0 and pinch_site_id >= 0:
    T_w3_tcp_meas = _measure_T_w3_tcp(_data, wrist_body_id, pinch_site_id)
    meas_pos = T_w3_tcp_meas[:3, 3]
    meas_rpy = R.from_matrix(T_w3_tcp_meas[:3, :3]).as_euler('xyz', degrees=True)
    print(f"  T(wrist3→TCP) 实测 pos: {meas_pos}")
    print(f"  T(wrist3→TCP) 实测 rpy(deg): {meas_rpy}")

if wrist_body_id >= 0 and flange_id >= 0:
    T_w_w3 = np.eye(4)
    T_w_w3[:3, :3] = _data.xmat[wrist_body_id].reshape(3, 3)
    T_w_w3[:3, 3] = _data.xpos[wrist_body_id]
    T_w_fl = np.eye(4)
    T_w_fl[:3, :3] = _data.xmat[flange_id].reshape(3, 3)
    T_w_fl[:3, 3] = _data.xpos[flange_id]
    T_w3_fl = np.linalg.inv(T_w_w3) @ T_w_fl
    w3fl_pos = T_w3_fl[:3, 3]
    w3fl_rpy = R.from_matrix(T_w3_fl[:3, :3]).as_euler('xyz', degrees=True)
    print(f"  T(wrist3→flange) 实测 pos: {w3fl_pos}")
    print(f"  T(wrist3→flange) 实测 rpy(deg): {w3fl_rpy}")


# --- (C) 配置链路：代码中的变换矩阵 ---
print("\n--- (C) 代码配置链路的变换矩阵 ---")

print(f"  robot.curobo_tcp_rel_pos: {robot.curobo_tcp_rel_pos}")
print(f"  robot.curobo_tcp_rel_rot: {robot.curobo_tcp_rel_rot}")
print(f"  CFG_REL_IS_TCP_TO_EE: {CFG_REL_IS_TCP_TO_EE}")
print(f"  W3_TO_EE_BRIDGE_POS: {W3_TO_EE_BRIDGE_POS}")
print(f"  W3_TO_EE_BRIDGE_RPY(deg): {np.degrees(W3_TO_EE_BRIDGE_RPY)}")

T_bridge = _bridge_T_w3_ee()
bridge_pos = T_bridge[:3, 3]
bridge_rpy = R.from_matrix(T_bridge[:3, :3]).as_euler('xyz', degrees=True)
print(f"  T_bridge(w3→ee) pos: {bridge_pos}")
print(f"  T_bridge(w3→ee) rpy(deg): {bridge_rpy}")

T_ee_tcp = _cfg_T_ee_tcp(robot)
eetcp_pos = T_ee_tcp[:3, 3]
eetcp_rpy = R.from_matrix(T_ee_tcp[:3, :3]).as_euler('xyz', degrees=True)
print(f"  T_cfg(ee→tcp) pos: {eetcp_pos}")
print(f"  T_cfg(ee→tcp) rpy(deg): {eetcp_rpy}")

T_w3_tcp_pred = _T_w3_tcp_from_cfg(robot)
pred_pos = T_w3_tcp_pred[:3, 3]
pred_rpy = R.from_matrix(T_w3_tcp_pred[:3, :3]).as_euler('xyz', degrees=True)
print(f"  T_pred(w3→tcp) = bridge @ cfg  pos: {pred_pos}")
print(f"  T_pred(w3→tcp) = bridge @ cfg  rpy(deg): {pred_rpy}")


# --- (D) 配置 vs 实测 对比 ---
print("\n--- (D) 配置 vs 实测 误差 ---")
if wrist_body_id >= 0 and pinch_site_id >= 0:
    T_err = np.linalg.inv(T_w3_tcp_pred) @ T_w3_tcp_meas
    pos_err = np.linalg.norm(T_err[:3, 3]) * 1000  # mm
    rot_err = np.degrees(np.linalg.norm(R.from_matrix(T_err[:3, :3]).as_rotvec()))
    print(f"  位置误差: {pos_err:.4f} mm")
    print(f"  姿态误差: {rot_err:.4f} deg")
    if pos_err > 1.0 or rot_err > 0.5:
        print("  ⚠️  误差过大，bridge 常量需要重新标定")
    else:
        print("  ✅ 链路一致")


# --- (E) IK 坐标系验证 ---
print("\n--- (E) IK 求解器坐标系验证 ---")
print(f"  handler 设置的 robot base pos: {robot_dict['ur5e_2f85']['pos'].tolist()}")
print(f"  handler 设置的 robot base rot: {robot_dict['ur5e_2f85']['rot'].tolist()}")
if base_body_id >= 0:
    print(f"  MuJoCo 实际 ur5e_base pos: {base_pos_mj.tolist()}")
    base_offset = np.linalg.norm(
        np.array(robot_dict['ur5e_2f85']['pos'].tolist()) - base_pos_mj
    )
    print(f"  设置值 vs MuJoCo实际 偏差: {base_offset*1000:.2f} mm")

print(f"  IK solver type: {args.solver}")
print(f"  ee_body_name: {robot.ee_body_name}")
print(f"  当前是否做 world→base 变换: 否 (方案A，直接传世界坐标)")
print(f"  ⚠️  如果 base 不在世界原点，IK 结果会有系统偏差")


# --- (F) 第一步轨迹目标验证 ---
print("\n--- (F) 第一步 (step=0) 轨迹目标 ---")
_panel_tip_test = get_site_world_pos(_data, panel_tip_site_id)
_ee_pos_t, _ee_quat_t, _gw = get_trajectory_targets(
    0, device, robot, REAL_TCP_DOWN_QUAT_WXYZ, _panel_tip_test
)
print(f"  TCP 目标 (世界): pos={_ee_pos_t.detach().cpu().numpy().flatten()}")
print(f"                   quat(wxyz)={_ee_quat_t.detach().cpu().numpy().flatten()}")
print(f"  注意: 上面是经过 tcp_pose_to_wrist3_pose 转换后的 wrist3 目标")
print(f"  gripper_width: {_gw}")

print("\n" + "="*70)



# 修改主循环 - 每步都求解IK，实现真正的笛卡尔空间轨迹跟踪
for step in range(450):
    states = handler.get_states()

    # ===== 步骤1: 读取 panel4_tip_L 位置并生成轨迹目标 =====
    panel_tip_world_pos = get_site_world_pos(_data, panel_tip_site_id)

    # ee_pos_target, ee_quat_target, gripper_width = get_trajectory_targets(step, device, robot)
    ee_pos_target, ee_quat_target, gripper_width = get_trajectory_targets(
        step, device, robot, REAL_TCP_DOWN_QUAT_WXYZ, panel_tip_world_pos
    )

    # ===== 步骤2: 每步都执行 IK 求解（关键改动）=====
    # 获取当前关节角度作为 IK 初始值
    reorder_idx = handler.get_joint_reindex(robot.name)
    inv_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
    curr_q = obs.robots[robot.name].joint_pos[:, inv_idx]

    # 扩展目标到所有环境
    ee_pos_batch_world = ee_pos_target.repeat(args.num_envs, 1)
    ee_quat_batch_world = ee_quat_target.repeat(args.num_envs, 1)
    
    ee_pos_base, ee_quat_base = world_pose_to_base_pose(
        ee_pos_batch_world[0], 
        ee_quat_batch_world[0], 
        robot_base_pos_world[0], 
        robot_base_quat_world[0]
    )    
    
    # 扩展到所有环境
    ee_pos_base_batch = ee_pos_base.repeat(args.num_envs, 1)
    ee_quat_base_batch = ee_quat_base.repeat(args.num_envs, 1)

    # ======= 【关键修改 1 / 2】：将其翻译进 IK 宇宙 (URDF) =======
    curr_q_urdf = curr_q.clone()
    curr_q_urdf[:, 1] -= np.pi / 2  # shoulder_lift 扣除 90°
    curr_q_urdf[:, 3] -= np.pi / 2  # wrist_1 扣除 90°
    # 执行 IK 求解（传入翻译后的初始值）
    q_sol, ik_success = ik_solver.solve_ik_batch(
        ee_pos_base_batch,      
        ee_quat_base_batch,     
        curr_q_urdf             # <---- 留意这里变更为伪装后的 curr_q_urdf
    )


    # # 执行 IK 求解（现在使用 base 坐标系的目标）
    # q_sol, ik_success = ik_solver.solve_ik_batch(
    #     ee_pos_base_batch,      # ✓ base 坐标系位置
    #     ee_quat_base_batch,     # ✓ base 坐标系姿态（已经是 [w,x,y,z] 格式）
    #     curr_q
    # )

    # # 检查求解结果
    # if ik_success is not None and ik_success.any():
    #     target_joint_angles = q_sol.clone()
    # else:
    #     target_joint_angles = curr_q.clone()

    # ======= 【关键修改 2 / 2】：将解算结果翻译回 MJCF 宇宙 =======
    if ik_success is not None and ik_success.any():
        q_sol_mjcf = q_sol.clone()
        q_sol_mjcf[:, 1] += np.pi / 2 # 补回真实的物理偏角
        q_sol_mjcf[:, 3] += np.pi / 2 # 补回真实的物理偏角
        target_joint_angles = q_sol_mjcf
    else:
        target_joint_angles = curr_q.clone()

    # ===== 调试打印：第20步和第40步 =====
    if step in (20, 40, 70):
        print(f"\n{'='*60}")
        print(f"[DEBUG] step={step}")
        print(f"  panel_tip_world_pos : {panel_tip_world_pos}")
        print(f"  wrist3目标(世界) pos: {ee_pos_target.detach().cpu().numpy().flatten()}")
        print(f"  wrist3目标(世界) quat:{ee_quat_target.detach().cpu().numpy().flatten()}  (w,x,y,z)")
        print(f"  wrist3目标(base) pos: {ee_pos_base.detach().cpu().numpy().flatten()}")
        print(f"  wrist3目标(base) quat:{ee_quat_base.detach().cpu().numpy().flatten()}  (w,x,y,z)")
        print(f"  curr_q              : {curr_q.detach().cpu().numpy().flatten()}")
        print(f"  ik_success          : {ik_success}")
        print(f"  q_sol               : {q_sol.detach().cpu().numpy().flatten() if q_sol is not None else None}")
        print(f"  target_joint_angles : {target_joint_angles.detach().cpu().numpy().flatten()}")
        print(f"  gripper_width       : {gripper_width}")
        print(f"{'='*60}\n")
        


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

# ===== 脚本退出清理：避免 Renderer 报错 =====
import gc
# 如果有旧的 context 没有释放，显式释放
if hasattr(handler, 'close'):
    handler.close()

# 删除各种可能持有 C++ / MuJoCo 对象的引用
del obs
del handler
del obs_saver
if '_model' in locals(): del _model
if '_data' in locals(): del _data

# 强制进行垃圾回收，在 Python 解释器销毁 glfw 等全局模块之前回收 C++ 对象
gc.collect()
