"""
计算正确的抓取姿态四元数

坐标系变换链：
wrist_3_link → flange (quat="1 -1 0 0") → robotiq_2f85 → 2f85_base → base (quat="1 0 0 -1") → pinch

目标：让 pinch 点的 Z 轴指向太阳翼边缘的法向量方向
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# ===== 步骤1: 解析 MJCF 中的四元数 =====
# MuJoCo 四元数格式: [w, x, y, z]
# scipy 四元数格式: [x, y, z, w]

# flange 相对于 wrist_3_link 的旋转
flange_quat_mujoco = np.array([1, -1, 0, 0])
flange_quat_scipy = np.array([-1, 0, 0, 1])  # 转换为 [x, y, z, w]
flange_quat_scipy = flange_quat_scipy / np.linalg.norm(flange_quat_scipy)  # 归一化
r_flange = R.from_quat(flange_quat_scipy)

print("=" * 60)
print("步骤1: 解析 MJCF 四元数")
print("=" * 60)
print(f"flange 四元数 (MuJoCo [w,x,y,z]): {flange_quat_mujoco}")
print(f"flange 四元数 (scipy [x,y,z,w]): {flange_quat_scipy}")
print(f"flange 旋转矩阵:\n{r_flange.as_matrix()}")
print(f"flange 欧拉角 (XYZ, degrees): {r_flange.as_euler('xyz', degrees=True)}")

# base 相对于 2f85_base 的旋转
base_quat_mujoco = np.array([1, 0, 0, -1])
base_quat_scipy = np.array([0, 0, -1, 1])  # 转换为 [x, y, z, w]
base_quat_scipy = base_quat_scipy / np.linalg.norm(base_quat_scipy)  # 归一化
r_base = R.from_quat(base_quat_scipy)

print(f"\nbase 四元数 (MuJoCo [w,x,y,z]): {base_quat_mujoco}")
print(f"base 四元数 (scipy [x,y,z,w]): {base_quat_scipy}")
print(f"base 旋转矩阵:\n{r_base.as_matrix()}")
print(f"base 欧拉角 (XYZ, degrees): {r_base.as_euler('xyz', degrees=True)}")

# ===== 步骤2: 计算从 wrist_3_link 到 pinch 的复合旋转 =====
# pinch 相对于 wrist_3_link 的旋转 = flange * base
r_wrist_to_pinch = r_flange * r_base

print("\n" + "=" * 60)
print("步骤2: 复合旋转 (wrist_3_link → pinch)")
print("=" * 60)
print(f"复合旋转矩阵:\n{r_wrist_to_pinch.as_matrix()}")
print(f"复合欧拉角 (XYZ, degrees): {r_wrist_to_pinch.as_euler('xyz', degrees=True)}")

# ===== 步骤3: 分析太阳翼坐标系 =====
# satellite_root 的旋转: euler="0 0 1.570796" (绕 Z 轴旋转 90°)
satellite_euler = np.array([0, 0, 1.570796])  # radians
r_satellite = R.from_euler('xyz', satellite_euler)

print("\n" + "=" * 60)
print("步骤3: 太阳翼坐标系")
print("=" * 60)
print(f"satellite_root 欧拉角 (XYZ, radians): {satellite_euler}")
print(f"satellite_root 欧拉角 (XYZ, degrees): {np.degrees(satellite_euler)}")
print(f"satellite_root 旋转矩阵:\n{r_satellite.as_matrix()}")

# 太阳翼局部坐标系的 X, Y, Z 轴在世界坐标系中的方向
sat_x_world = r_satellite.as_matrix() @ np.array([1, 0, 0])
sat_y_world = r_satellite.as_matrix() @ np.array([0, 1, 0])
sat_z_world = r_satellite.as_matrix() @ np.array([0, 0, 1])

print(f"\n太阳翼局部 X 轴 → 世界坐标: {sat_x_world}")
print(f"太阳翼局部 Y 轴 → 世界坐标: {sat_y_world}")
print(f"太阳翼局部 Z 轴 → 世界坐标: {sat_z_world}")

# ===== 步骤4: 确定抓取方向 =====
print("\n" + "=" * 60)
print("步骤4: 确定抓取方向")
print("=" * 60)

# panel4_tip_L 在折叠状态下的位置: [0.15, 0.24, 1.28]
# 从 MJCF 结构分析：
# - panel4_L 的 hinge_4_L 轴是 axis="0 -1 0" (负 Y 轴)
# - 太阳翼表面法向量在局部坐标系中是 Z 轴方向
# - 经过 satellite_root 旋转后，太阳翼表面法向量在世界坐标系中是什么方向？

# 假设太阳翼表面法向量在卫星局部坐标系中是 +Z 方向
panel_normal_local = np.array([0, 0, 1])
panel_normal_world = r_satellite.as_matrix() @ panel_normal_local

print(f"太阳翼表面法向量 (局部): {panel_normal_local}")
print(f"太阳翼表面法向量 (世界): {panel_normal_world}")

# 抓取方向：夹爪应该从太阳翼表面法向量的反方向接近
# 即 pinch 点的 Z 轴应该指向 -panel_normal_world
grasp_direction_world = -panel_normal_world

print(f"\n期望的抓取方向 (世界坐标): {grasp_direction_world}")
print(f"  → 这意味着 pinch 的 Z 轴应该指向: {grasp_direction_world}")

# ===== 步骤5: 计算目标姿态 =====
print("\n" + "=" * 60)
print("步骤5: 计算 pinch 的目标姿态")
print("=" * 60)

# 构建目标旋转矩阵：pinch 的 Z 轴指向 grasp_direction_world
# 我们需要构建一个完整的旋转矩阵，需要定义 X, Y, Z 三个轴

# Z 轴 = grasp_direction_world
z_axis = grasp_direction_world / np.linalg.norm(grasp_direction_world)

# 选择一个合理的 X 轴方向（例如，尽量水平）
# 如果 Z 轴是水平的，X 轴可以选择竖直方向
# 如果 Z 轴是竖直的，X 轴可以选择水平方向
if abs(z_axis[2]) < 0.9:  # Z 轴不是竖直的
    # X 轴选择竖直方向
    x_axis_candidate = np.array([0, 0, 1])
else:
    # X 轴选择水平方向
    x_axis_candidate = np.array([1, 0, 0])

# Y 轴 = Z × X (右手坐标系)
y_axis = np.cross(z_axis, x_axis_candidate)
y_axis = y_axis / np.linalg.norm(y_axis)

# 重新计算 X 轴 = Y × Z
x_axis = np.cross(y_axis, z_axis)
x_axis = x_axis / np.linalg.norm(x_axis)

# 构建目标旋转矩阵
R_pinch_target = np.column_stack([x_axis, y_axis, z_axis])
r_pinch_target = R.from_matrix(R_pinch_target)

print(f"pinch 目标 X 轴 (世界): {x_axis}")
print(f"pinch 目标 Y 轴 (世界): {y_axis}")
print(f"pinch 目标 Z 轴 (世界): {z_axis}")
print(f"\npinch 目标旋转矩阵:\n{R_pinch_target}")
print(f"pinch 目标四元数 (scipy [x,y,z,w]): {r_pinch_target.as_quat()}")

# ===== 步骤6: 反向计算 wrist_3_link 的姿态 =====
print("\n" + "=" * 60)
print("步骤6: 反向计算 wrist_3_link 姿态")
print("=" * 60)

# r_pinch_target = r_wrist3 * r_wrist_to_pinch
# 因此: r_wrist3 = r_pinch_target * r_wrist_to_pinch^(-1)
r_wrist3_target = r_pinch_target * r_wrist_to_pinch.inv()

print(f"wrist_3_link 目标旋转矩阵:\n{r_wrist3_target.as_matrix()}")
print(f"wrist_3_link 目标欧拉角 (XYZ, degrees): {r_wrist3_target.as_euler('xyz', degrees=True)}")

# 转换为 MuJoCo 格式的四元数 [qx, qy, qz, qw]
quat_scipy = r_wrist3_target.as_quat()  # [x, y, z, w]
quat_mujoco = np.array([quat_scipy[0], quat_scipy[1], quat_scipy[2], quat_scipy[3]])  # [qx, qy, qz, qw]

print(f"\nwrist_3_link 目标四元数 (scipy [x,y,z,w]): {quat_scipy}")
print(f"wrist_3_link 目标四元数 (MuJoCo [qx,qy,qz,qw]): {quat_mujoco}")

# ===== 步骤7: 验证 =====
print("\n" + "=" * 60)
print("步骤7: 验证计算结果")
print("=" * 60)

# 验证：r_wrist3_target * r_wrist_to_pinch 应该等于 r_pinch_target
r_pinch_verify = r_wrist3_target * r_wrist_to_pinch
pinch_z_verify = r_pinch_verify.as_matrix() @ np.array([0, 0, 1])

print(f"验证：pinch 的 Z 轴方向 (世界): {pinch_z_verify}")
print(f"期望：pinch 的 Z 轴方向 (世界): {z_axis}")
print(f"误差: {np.linalg.norm(pinch_z_verify - z_axis):.6f}")

# ===== 步骤8: 输出最终结果 =====
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)
print(f"在代码中使用的四元数 (格式 [qx, qy, qz, qw]):")
print(f"quat_grasp = {list(quat_mujoco)}")
print("\n替换代码中的这一行:")
print(f"  quat_down = [0.0, 1.0, 0.0, 0.0]")
print(f"为:")
print(f"  quat_grasp = [{quat_mujoco[0]:.6f}, {quat_mujoco[1]:.6f}, {quat_mujoco[2]:.6f}, {quat_mujoco[3]:.6f}]")

# ===== 额外分析：当前姿态 =====
print("\n" + "=" * 60)
print("额外分析：当前姿态 quat_down = [0, 1, 0, 0]")
print("=" * 60)

quat_down_mujoco = np.array([0, 1, 0, 0])
quat_down_scipy = np.array([1, 0, 0, 0])  # [x, y, z, w]
r_wrist3_current = R.from_quat(quat_down_scipy)

print(f"当前 wrist_3_link 四元数: {quat_down_mujoco}")
print(f"当前 wrist_3_link 旋转矩阵:\n{r_wrist3_current.as_matrix()}")

# 当前 pinch 的姿态
r_pinch_current = r_wrist3_current * r_wrist_to_pinch
pinch_z_current = r_pinch_current.as_matrix() @ np.array([0, 0, 1])

print(f"\n当前 pinch 的 Z 轴方向 (世界): {pinch_z_current}")
print(f"期望 pinch 的 Z 轴方向 (世界): {z_axis}")

# 计算两个方向之间的夹角
angle_diff = np.arccos(np.clip(np.dot(pinch_z_current, z_axis), -1.0, 1.0))
print(f"\n当前方向与期望方向的夹角: {np.degrees(angle_diff):.2f}°")
print(f"  → 这就是为什么抓取姿态'相反'的原因！")
