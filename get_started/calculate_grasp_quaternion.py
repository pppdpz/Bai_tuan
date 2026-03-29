"""
计算正确的抓取姿态四元数

坐标系变换链：
wrist_3_link → flange (quat="1 -1 0 0") → robotiq_2f85 → 2f85_base → base (quat="1 0 0 -1") → pinch

目标：让 pinch 点的 Z 轴指向太阳翼边缘的法向量方向

使用场景：
1. 理解 UR5e + Robotiq 2F-85 的坐标系变换链
2. 验证抓取姿态的理论计算
3. 调试时对比理论值与实测值

注意：此文件仅用于离线分析，实际运动规划使用 4_motion_planning_baituan.py
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def mujoco_to_scipy_quat(quat_wxyz):
    """MuJoCo [w,x,y,z] -> scipy [x,y,z,w]"""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def normalize_quat(quat):
    """归一化四元数"""
    return quat / np.linalg.norm(quat)


def build_rotation_frame(z_direction):
    """根据 Z 轴方向构建完整的旋转矩阵（右手坐标系）"""
    z_axis = z_direction / np.linalg.norm(z_direction)
    
    # 选择合理的 X 轴候选方向
    x_candidate = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.9 else np.array([1, 0, 0])
    
    # 构建正交基
    y_axis = np.cross(z_axis, x_candidate)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    
    return np.column_stack([x_axis, y_axis, z_axis])


def print_section(title):
    """打印分隔线"""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ===== 1. 解析 MJCF 坐标系变换 =====
print_section("1. 解析 MJCF 坐标系变换")

# flange 和 base 的旋转（MuJoCo 格式）
flange_quat_mj = np.array([1, -1, 0, 0])
base_quat_mj = np.array([1, 0, 0, -1])

# 转换为 scipy 格式并归一化
r_flange = R.from_quat(normalize_quat(mujoco_to_scipy_quat(flange_quat_mj)))
r_base = R.from_quat(normalize_quat(mujoco_to_scipy_quat(base_quat_mj)))

# 复合旋转：wrist_3_link → pinch
r_wrist_to_pinch = r_flange * r_base

print(f"flange 欧拉角 (XYZ, deg): {r_flange.as_euler('xyz', degrees=True)}")
print(f"base 欧拉角 (XYZ, deg): {r_base.as_euler('xyz', degrees=True)}")
print(f"复合旋转 (wrist→pinch) 欧拉角: {r_wrist_to_pinch.as_euler('xyz', degrees=True)}")

# ===== 2. 分析太阳翼坐标系 =====
print_section("2. 分析太阳翼坐标系")

# satellite_root 绕 Z 轴旋转 90°
r_satellite = R.from_euler('xyz', [0, 0, 1.570796])
sat_rot_matrix = r_satellite.as_matrix()

# 太阳翼表面法向量（局部 Z 轴）在世界坐标系中的方向
panel_normal_world = sat_rot_matrix @ np.array([0, 0, 1])

print(f"太阳翼旋转 (deg): {r_satellite.as_euler('xyz', degrees=True)}")
print(f"太阳翼法向量 (世界): {panel_normal_world}")

# ===== 3. 计算目标抓取姿态 =====
print_section("3. 计算目标抓取姿态")

# 抓取方向：从法向量反方向接近
grasp_direction = -panel_normal_world
print(f"抓取方向 (pinch Z轴): {grasp_direction}")

# 构建 pinch 的目标旋转矩阵
R_pinch_target = build_rotation_frame(grasp_direction)
r_pinch_target = R.from_matrix(R_pinch_target)

print(f"pinch 目标姿态 (欧拉角, deg): {r_pinch_target.as_euler('xyz', degrees=True)}")

# ===== 4. 反向计算 wrist_3_link 姿态 =====
print_section("4. 反向计算 wrist_3_link 姿态")

# r_pinch_target = r_wrist3 * r_wrist_to_pinch
# => r_wrist3 = r_pinch_target * r_wrist_to_pinch^(-1)
r_wrist3_target = r_pinch_target * r_wrist_to_pinch.inv()

quat_scipy = r_wrist3_target.as_quat()  # [x, y, z, w]
quat_mujoco = quat_scipy  # MuJoCo 也使用 [x, y, z, w] 格式

print(f"wrist_3_link 目标欧拉角 (deg): {r_wrist3_target.as_euler('xyz', degrees=True)}")
print(f"wrist_3_link 目标四元数 [x,y,z,w]: {quat_mujoco}")

# ===== 5. 验证计算结果 =====
print_section("5. 验证计算结果")

r_pinch_verify = r_wrist3_target * r_wrist_to_pinch
pinch_z_verify = r_pinch_verify.as_matrix() @ np.array([0, 0, 1])
error = np.linalg.norm(pinch_z_verify - grasp_direction / np.linalg.norm(grasp_direction))

print(f"验证 pinch Z轴: {pinch_z_verify}")
print(f"期望 pinch Z轴: {grasp_direction / np.linalg.norm(grasp_direction)}")
print(f"误差: {error:.6f}")

# ===== 6. 输出最终结果 =====
print_section("最终结果")

print(f"quat_grasp = [{quat_mujoco[0]:.6f}, {quat_mujoco[1]:.6f}, {quat_mujoco[2]:.6f}, {quat_mujoco[3]:.6f}]")
print("\n在 4_motion_planning_baituan.py 中使用:")
print(f"  REAL_TCP_DOWN_QUAT_WXYZ = np.array([{quat_mujoco[3]:.6f}, {quat_mujoco[0]:.6f}, {quat_mujoco[1]:.6f}, {quat_mujoco[2]:.6f}])")

# ===== 7. 对比当前姿态 =====
print_section("7. 对比当前姿态 [0, 1, 0, 0]")

r_wrist3_current = R.from_quat([1, 0, 0, 0])  # [x,y,z,w]
r_pinch_current = r_wrist3_current * r_wrist_to_pinch
pinch_z_current = r_pinch_current.as_matrix() @ np.array([0, 0, 1])

angle_diff = np.arccos(np.clip(np.dot(pinch_z_current, grasp_direction / np.linalg.norm(grasp_direction)), -1.0, 1.0))

print(f"当前 pinch Z轴: {pinch_z_current}")
print(f"期望 pinch Z轴: {grasp_direction / np.linalg.norm(grasp_direction)}")
print(f"夹角偏差: {np.degrees(angle_diff):.2f}°")
