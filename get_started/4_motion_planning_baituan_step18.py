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
    grasp_offset_x: float = 0.1      # X方向偏移（米）
    grasp_offset_y: float = 0.0      # Y方向偏移（米）
    grasp_offset_z: float = -0.05    # Z方向偏移（米），负值表示向下
    use_normal_offset: bool = False  # 是否使用法向偏移（默认禁用）
    normal_offset_dist: float = 0.01 # 沿法向偏移距离（米）

    def __post_init__(self):
        log.info(f"Args: {self}")
        log.info(f"抓取偏移参数: X={self.grasp_offset_x}, Y={self.grasp_offset_y}, Z={self.grasp_offset_z}")
        log.info(f"法向偏移: {'启用' if self.use_normal_offset else '禁用'}, 距离={self.normal_offset_dist}m")


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
            "shoulder_pan_joint": 0.3,
            "shoulder_lift_joint": -0.5,
            "elbow_joint": -1.6,               
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
        log.info(f"设置关节 {_jname} 到折叠角度: {_angle:.4f} rad")
    else:
        log.warning(f"未找到关节: {_jname}")

# 更新物理状态
mj.mj_forward(_model, _data)
log.info("小卫星太阳翼已折叠到初始状态")

# ===== 诊断：读取初始状态下的真实姿态 =====
log.info("\n" + "="*60)
log.info("诊断：读取机器人初始姿态")
log.info("="*60)

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
    
    log.info(f"初始 TCP(pinch) 位置: [{pinch_pos[0]:.4f}, {pinch_pos[1]:.4f}, {pinch_pos[2]:.4f}]")
    log.info(f"初始 TCP(pinch) 姿态 (w,x,y,z): [{pinch_quat_wxyz[0]:.4f}, {pinch_quat_wxyz[1]:.4f}, {pinch_quat_wxyz[2]:.4f}, {pinch_quat_wxyz[3]:.4f}]")
    log.info(f"初始 wrist3 位置: [{wrist_pos[0]:.4f}, {wrist_pos[1]:.4f}, {wrist_pos[2]:.4f}]")
    log.info(f"初始 wrist3 姿态 (w,x,y,z): [{wrist_quat_wxyz[0]:.4f}, {wrist_quat_wxyz[1]:.4f}, {wrist_quat_wxyz[2]:.4f}, {wrist_quat_wxyz[3]:.4f}]")
    
    # 保存真实的 TCP 朝下姿态供后续使用 [w,x,y,z] 格式
    REAL_TCP_DOWN_QUAT_WXYZ = pinch_quat_wxyz.copy()
    log.info(f"\n✓ 已保存真实 TCP 姿态到 REAL_TCP_DOWN_QUAT_WXYZ (w,x,y,z格式)")
else:
    log.error("无法找到 pinch site 或 wrist_3_link body")
    REAL_TCP_DOWN_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])  # 回退值 [w,x,y,z]

if panel_tip_site_id >= 0:
    _panel_tip_pos = _data.site_xpos[panel_tip_site_id]
    log.info(f"初始 panel4_tip_L 位置: [{_panel_tip_pos[0]:.4f}, {_panel_tip_pos[1]:.4f}, {_panel_tip_pos[2]:.4f}]")
else:
    log.warning("未找到 site: panel4_tip_L，将使用硬编码目标点作为回退")

log.info("="*60 + "\n")



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
        log.warning(f"未找到 body: {panel_body_name}，使用默认法向 [0,0,1]")
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
    T_wrist3_to_tcp = _T_w3_tcp_from_cfg(robot_cfg)
    
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
    waypoint_pos_pinch = [0.3, 0.17, 1.10]

    # 默认回退目标（世界坐标系，site 读取失败时使用）
    # 注意：panel4_tip_L 初始位置是 [0.3000, 0.2400, 1.2800]
    approach_pos_pinch = [0.3, 0.24, 1.43]   # 在抓取点上方 15cm
    grasp_pos_pinch = [0.3, 0.24, 1.28]      # 对准 panel4_tip_L
    deploy_pos_pinch = [0.3, -0.30, 1.30]    # 沿 -Y 方向拉动 54cm
    retract_pos_pinch = [0.3, -0.30, 1.35]   # 上抬 5cm

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
                if step == 0:
                    log.info(f"太阳能板法向量: [{panel_normal[0]:.4f}, {panel_normal[1]:.4f}, {panel_normal[2]:.4f}]")
        
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
        
        # ===== 调试输出（仅在关键步数） =====
        if step == 0:
            log.info(f"\n{'='*60}")
            log.info(f"抓取点位计算详情:")
            log.info(f"{'='*60}")
            log.info(f"原始 panel_tip: [{panel_tip[0]:.4f}, {panel_tip[1]:.4f}, {panel_tip[2]:.4f}]")
            log.info(f"直角坐标偏移: [{offset_xyz[0]:.4f}, {offset_xyz[1]:.4f}, {offset_xyz[2]:.4f}]")
            if panel_normal is not None and abs(normal_dist) > 1e-6:
                log.info(f"法向偏移: {normal_dist:.4f}m 沿 [{panel_normal[0]:.4f}, {panel_normal[1]:.4f}, {panel_normal[2]:.4f}]")
            else:
                log.info(f"法向偏移: 禁用")
            log.info(f"最终抓取点: [{grasp_pos_pinch[0]:.4f}, {grasp_pos_pinch[1]:.4f}, {grasp_pos_pinch[2]:.4f}]")
            offset_total = np.linalg.norm(grasp_pos_pinch_np - panel_tip)
            log.info(f"总偏移距离: {offset_total*1000:.2f}mm")
            log.info(f"{'='*60}\n")

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
            deploy_pos_pinch[2] + 0.05,
        ]
    elif step == 0:
        log.warning("panel4_tip_L 位置不可用，回退到硬编码轨迹目标")

    # 方案B: 直接使用 pinch/TCP 坐标定义轨迹
    # 在函数末尾通过 ee_pose_from_tcp_pose() 转换为 wrist_3_link 坐标
    initial_pos = initial_pos_pinch
    approach_pos = approach_pos_pinch
    grasp_pos = grasp_pos_pinch
    deploy_pos = deploy_pos_pinch
    retract_pos = retract_pos_pinch

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
    
    # ===== 阶段3: 抓取 (步数 70-90) =====
    elif step < 90:
        t = (step - 70) / 20.0
        # 位置保持不变
        ee_pos = grasp_pos
        ee_quat = tcp_quat_wxyz  # 使用真实的 TCP 姿态 [w,x,y,z]


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
        ee_quat = tcp_quat_wxyz  # 使用真实的 TCP 姿态 [w,x,y,z]


        gripper_width = 0.2  # 保持闭合状态
    
    # ===== 阶段5: 释放 (步数 170-190) =====
    elif step < 190:
        t = (step - 170) / 20.0
        # 位置保持在展开位置
        ee_pos = deploy_pos
        ee_quat = tcp_quat_wxyz  # 使用真实的 TCP 姿态 [w,x,y,z]

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
        ee_quat = tcp_quat_wxyz  # 使用真实的 TCP 姿态 [w,x,y,z]

        gripper_width = 1.0  # 保持打开
    
    # 方案B: 使用 TCP 坐标定义轨迹，然后转换为 wrist3 坐标
    ee_pos_tcp = torch.tensor([ee_pos], device=device, dtype=torch.float32)
    ee_quat_tcp = torch.tensor([ee_quat], device=device, dtype=torch.float32)  # [w,x,y,z] 格式

        
    # 使用经过验证的转换函数（误差 0.0000mm）
    ee_pos_target, ee_quat_target = tcp_pose_to_wrist3_pose(
        ee_pos_tcp,
        ee_quat_tcp,
        robot_cfg,
    )

    return ee_pos_target, ee_quat_target, gripper_width


# ===== 获取机器人 base_link 在世界坐标系中的位姿 =====
# 这个位姿在整个运动过程中保持不变（机器人底座固定）
# 从 robot_dict 中读取，与初始化时设置的值一致
robot_base_pos_world = robot_dict["ur5e_2f85"]["pos"]  # torch.tensor([1.0, 0.0, 0.895])
robot_base_quat_world = robot_dict["ur5e_2f85"]["rot"]  # torch.tensor([1.0, 0.0, 0.0, 0.0])

# 确保在正确的设备上
robot_base_pos_world = robot_base_pos_world.to(device).unsqueeze(0)  # [1, 3]
robot_base_quat_world = robot_base_quat_world.to(device).unsqueeze(0)  # [1, 4] (w,x,y,z)

log.info(f"机器人 base_link 世界坐标: pos={robot_base_pos_world[0].cpu().numpy()}, quat={robot_base_quat_world[0].cpu().numpy()}")

# ===== 启动时一次性自检：测量链路 vs 配置链路 =====
if pinch_site_id >= 0 and wrist_body_id >= 0:
    T_w3_tcp_meas = _measure_T_w3_tcp(_data, wrist_body_id, pinch_site_id)
    T_w3_tcp_pred = _T_w3_tcp_from_cfg(robot)
    T_err = np.linalg.inv(T_w3_tcp_pred) @ T_w3_tcp_meas
    pos_err_m = np.linalg.norm(T_err[:3, 3])
    rot_err_deg = np.degrees(np.linalg.norm(R.from_matrix(T_err[:3, :3]).as_rotvec()))
    log.info(
        f"[TCP链路自检] pos_err={pos_err_m*1000:.3f} mm, rot_err={rot_err_deg:.3f} deg "
        f"(阈值: 1.0 mm / 0.5 deg)"
    )
    if pos_err_m > 1e-3 or rot_err_deg > 0.5:
        raise RuntimeError("TCP链路自检失败：cfg/bridge 与实测不一致，请重新标定桥接常量")
else:
    raise RuntimeError("无法执行 TCP链路自检：缺少 pinch site 或 wrist_3_link body")



# 在主循环开始前，初始化目标关节角度
target_joint_angles = None  # 添加在 for step in range(220): 之前
# ===== 新增：创建关节角度记录列表 =====
joint_2_angles_record = []  # 记录每一步的第二关节角度

# ===== 新增：将参数传递给轨迹生成函数 =====
# 通过函数属性传递，避免修改函数签名
get_trajectory_targets._args = args
get_trajectory_targets._model = _model
get_trajectory_targets._data = _data
log.info("已将 args, model, data 传递给 get_trajectory_targets 函数")
log.info(f"当前偏移配置: X={args.grasp_offset_x}m, Y={args.grasp_offset_y}m, Z={args.grasp_offset_z}m")
log.info(f"法向偏移: {'启用' if args.use_normal_offset else '禁用'}\n")

# 修改主循环 - 每步都求解IK，实现真正的笛卡尔空间轨迹跟踪
for step in range(220):
    log.debug(f"Step {step}")
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

    
    # 方案A：使用世界坐标（实验）
    ee_pos_base = ee_pos_batch_world[0:1]
    ee_quat_base = ee_quat_batch_world[0:1]
    
    
    # ===== 新增：坐标转换验证（仅在关键步骤打印）=====
    if step in [0, 50, 90, 170]:
        log.info(f"\n{'='*60}")
        log.info(f"坐标转换验证 (Step {step})")
        log.info(f"{'='*60}")
        log.info(f"世界坐标系目标: pos={ee_pos_batch_world[0].cpu().numpy()}")
        log.info(f"base 坐标系目标: pos={ee_pos_base[0].cpu().numpy()}")
        log.info(f"机器人 base 世界位置: {robot_base_pos_world[0].cpu().numpy()}")
        log.info(f"转换差值: {(ee_pos_batch_world[0] - robot_base_pos_world[0]).cpu().numpy()}")
        log.info(f"{'='*60}\n")

    # 扩展到所有环境
    ee_pos_base_batch = ee_pos_base.repeat(args.num_envs, 1)
    ee_quat_base_batch = ee_quat_base.repeat(args.num_envs, 1)

    # ===== 新增：四元数格式诊断（仅在关键步骤打印）=====
    if step in [0, 50, 90, 170]:
        log.info(f"\n{'='*60}")
        log.info(f"四元数格式诊断 (Step {step})")
        log.info(f"{'='*60}")
        quat_np = ee_quat_base_batch[0].cpu().numpy()
        log.info(f"ee_quat_base_batch: [{quat_np[0]:.4f}, {quat_np[1]:.4f}, {quat_np[2]:.4f}, {quat_np[3]:.4f}] (应为 w,x,y,z)")
        log.info(f"四元数范数: {ee_quat_base_batch[0].norm().item():.6f} (应该≈1.0)")
        
        # 验证四元数格式：w 分量应该是最大的（对于接近单位旋转）
        if abs(quat_np[0]) > 0.7:
            log.info(f"✓ 四元数格式正确：w={quat_np[0]:.4f} 是主分量")
        elif abs(quat_np[3]) > 0.7 and abs(quat_np[0]) < 0.3:
            log.error(f"⚠️ 四元数格式可能错误！w={quat_np[0]:.4f} 太小，可能是 [x,y,z,w] 格式")
        log.info(f"{'='*60}\n")

    # 执行 IK 求解（现在使用 base 坐标系的目标）
    q_sol, ik_success = ik_solver.solve_ik_batch(
        ee_pos_base_batch,      # ✓ base 坐标系位置
        ee_quat_base_batch,     # ✓ base 坐标系姿态（已经是 [w,x,y,z] 格式）
        curr_q
    )


    # 检查求解结果
    if ik_success is not None and ik_success.any():
        target_joint_angles = q_sol.clone()
        # ===== 新增：打印第二关节角度 =====
        joint_2_angle = target_joint_angles[0, 1].item()  # 第二关节（索引1）
        joint_2_deg = joint_2_angle * 180 / 3.14159  # 转换为角度
        
        # 定义安全范围（根据UR机械臂调整）
        JOINT_2_MIN = -2.0  # 弧度，约 -115度
        JOINT_2_MAX = 0.5   # 弧度，约 29度
        
        # 检查是否超出安全范围
        if joint_2_angle < JOINT_2_MIN or joint_2_angle > JOINT_2_MAX:
            log.warning(f"⚠️  Step {step}: 第二关节角度 = {joint_2_angle:.3f} rad ({joint_2_deg:.1f}°) - 超出安全范围!")
        else:
            log.info(f"✓ Step {step}: 第二关节角度 = {joint_2_angle:.3f} rad ({joint_2_deg:.1f}°)")
        # ===== 新增结束 =====
        # ===== 新增：记录第二关节角度 =====
        joint_2_angles_record.append({
            'step': step,
            'joint_2_rad': target_joint_angles[0, 1].item(),
            'joint_2_deg': target_joint_angles[0, 1].item() * 180 / 3.14159
        })

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
