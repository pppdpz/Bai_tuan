"""
深入分析姿态问题的根本原因

关键发现：
1. ee_pose_from_tcp_pose() 只转换位置，不转换姿态
2. TCP 四元数被直接传给 IK solver 作为 wrist_3_link 的目标
3. 但 wrist_3_link 和 pinch 之间有复合旋转差异
"""

import numpy as np
import torch

# ===== 第一部分：验证 matrix_from_quat 的四元数约定 =====
print("=" * 70)
print("第一部分：验证 matrix_from_quat 的四元数约定")
print("=" * 70)

def matrix_from_quat_numpy(quaternions):
    """
    复现 metasim/utils/math.py 中的 matrix_from_quat 逻辑
    四元数格式: (w, x, y, z)
    """
    r, i, j, k = quaternions[0], quaternions[1], quaternions[2], quaternions[3]
    two_s = 2.0 / (quaternions * quaternions).sum()
    
    o = np.array([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ])
    
    return o.reshape(3, 3)

# 测试几个已知的四元数
test_cases = [
    ("单位四元数 (无旋转)", np.array([1, 0, 0, 0])),
    ("绕X轴旋转90°", np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])),
    ("绕Y轴旋转90°", np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])),
    ("绕Z轴旋转90°", np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])),
    ("当前代码中的 quat_down", np.array([0, 1, 0, 0])),
]

for name, quat in test_cases:
    mat = matrix_from_quat_numpy(quat)
    print(f"\n{name}:")
    print(f"  四元数 (w,x,y,z): {quat}")
    print(f"  旋转矩阵:\n{mat}")
    print(f"  X轴 → {mat @ np.array([1, 0, 0])}")
    print(f"  Y轴 → {mat @ np.array([0, 1, 0])}")
    print(f"  Z轴 → {mat @ np.array([0, 0, 1])}")

# ===== 第二部分：分析 ee_pose_from_tcp_pose 的问题 =====
print("\n" + "=" * 70)
print("第二部分：分析 ee_pose_from_tcp_pose 的问题")
print("=" * 70)

print("""
函数签名：
def ee_pose_from_tcp_pose(robot_cfg, tcp_pos, tcp_quat):
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos)
    ee_pos = tcp_pos + torch.matmul(matrix_from_quat(tcp_quat), -tcp_rel_pos.unsqueeze(-1)).squeeze()
    return ee_pos, tcp_quat  # ← 注意：直接返回 tcp_quat！
    
关键问题：
1. 函数只转换了位置 (ee_pos)，使用了 tcp_quat 的旋转矩阵
2. 但姿态直接返回 tcp_quat，没有做任何转换
3. 这意味着 IK solver 收到的是 TCP 的姿态，而不是 wrist_3_link 的姿态

正确的逻辑应该是：
    ee_quat = tcp_quat * inv(tcp_rel_rot)  # 需要考虑相对旋转
    return ee_pos, ee_quat
""")

# ===== 第三部分：计算正确的 wrist_3_link 到 pinch 的旋转 =====
print("\n" + "=" * 70)
print("第三部分：从 MJCF 计算 wrist_3_link → pinch 的旋转")
print("=" * 70)

# 从 MJCF 解析的旋转
# flange: quat="1 -1 0 0" (MuJoCo格式 w,x,y,z)
flange_quat = np.array([1, -1, 0, 0])
flange_quat = flange_quat / np.linalg.norm(flange_quat)
R_flange = matrix_from_quat_numpy(flange_quat)

print(f"flange 四元数 (w,x,y,z): {flange_quat}")
print(f"flange 旋转矩阵:\n{R_flange}")

# base: quat="1 0 0 -1" (MuJoCo格式 w,x,y,z)
base_quat = np.array([1, 0, 0, -1])
base_quat = base_quat / np.linalg.norm(base_quat)
R_base = matrix_from_quat_numpy(base_quat)

print(f"\nbase 四元数 (w,x,y,z): {base_quat}")
print(f"base 旋转矩阵:\n{R_base}")

# 复合旋转：wrist_3_link → flange → base → pinch
R_wrist_to_pinch = R_flange @ R_base

print(f"\n复合旋转 (wrist_3_link → pinch):")
print(f"旋转矩阵:\n{R_wrist_to_pinch}")
print(f"\n含义：")
print(f"  wrist_3_link 的 X 轴 → pinch 的 {R_wrist_to_pinch @ np.array([1, 0, 0])}")
print(f"  wrist_3_link 的 Y 轴 → pinch 的 {R_wrist_to_pinch @ np.array([0, 1, 0])}")
print(f"  wrist_3_link 的 Z 轴 → pinch 的 {R_wrist_to_pinch @ np.array([0, 0, 1])}")

# ===== 第四部分：反推正确的 wrist_3_link 姿态 =====
print("\n" + "=" * 70)
print("第四部分：反推正确的 wrist_3_link 姿态")
print("=" * 70)

# 目标：pinch 的 Z 轴指向世界坐标的 -Z 方向（垂直向下）
print("目标：pinch 的 Z 轴指向世界坐标的 -Z 方向")

# 构建 pinch 的目标旋转矩阵
# pinch Z 轴 = [0, 0, -1]
# pinch X 轴 = [1, 0, 0] (保持水平)
# pinch Y 轴 = Z × X = [0, -1, 0]
R_pinch_target = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

print(f"\npinch 目标旋转矩阵:\n{R_pinch_target}")
print(f"  X 轴: {R_pinch_target[:, 0]}")
print(f"  Y 轴: {R_pinch_target[:, 1]}")
print(f"  Z 轴: {R_pinch_target[:, 2]}")

# 反推 wrist_3_link 的旋转矩阵
# R_pinch_target = R_wrist3 @ R_wrist_to_pinch
# 因此: R_wrist3 = R_pinch_target @ inv(R_wrist_to_pinch)
R_wrist3_target = R_pinch_target @ np.linalg.inv(R_wrist_to_pinch)

print(f"\nwrist_3_link 目标旋转矩阵:\n{R_wrist3_target}")

# 将旋转矩阵转换为四元数
def quat_from_matrix(R):
    """
    从旋转矩阵计算四元数 (w, x, y, z)
    使用 Shepperd's method
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])

quat_wrist3_target = quat_from_matrix(R_wrist3_target)
print(f"\nwrist_3_link 目标四元数 (w,x,y,z): {quat_wrist3_target}")

# 验证
R_verify = matrix_from_quat_numpy(quat_wrist3_target)
print(f"\n验证：从四元数重建的旋转矩阵:\n{R_verify}")
print(f"误差: {np.linalg.norm(R_verify - R_wrist3_target):.10f}")

# 验证 pinch 的最终姿态
R_pinch_verify = R_verify @ R_wrist_to_pinch
print(f"\n验证：pinch 的最终旋转矩阵:\n{R_pinch_verify}")
print(f"  Z 轴: {R_pinch_verify @ np.array([0, 0, 1])}")
print(f"  期望: [0, 0, -1]")

# ===== 第五部分：对比当前代码中的 quat_down =====
print("\n" + "=" * 70)
print("第五部分：对比当前代码中的 quat_down")
print("=" * 70)

quat_down = np.array([0, 1, 0, 0])  # (w, x, y, z)
R_down = matrix_from_quat_numpy(quat_down)

print(f"当前 quat_down (w,x,y,z): {quat_down}")
print(f"当前 wrist_3_link 旋转矩阵:\n{R_down}")

R_pinch_current = R_down @ R_wrist_to_pinch
print(f"\n当前 pinch 的旋转矩阵:\n{R_pinch_current}")
print(f"  Z 轴: {R_pinch_current @ np.array([0, 0, 1])}")

# 计算角度差异
pinch_z_current = R_pinch_current @ np.array([0, 0, 1])
pinch_z_target = np.array([0, 0, -1])
angle_diff = np.arccos(np.clip(np.dot(pinch_z_current, pinch_z_target), -1, 1))

print(f"\n当前 pinch Z 轴: {pinch_z_current}")
print(f"目标 pinch Z 轴: {pinch_z_target}")
print(f"角度差异: {np.degrees(angle_diff):.2f}°")

# ===== 第六部分：最终结论 =====
print("\n" + "=" * 70)
print("最终结论")
print("=" * 70)

print(f"""
问题根源：
1. ee_pose_from_tcp_pose() 只转换位置，不转换姿态
2. 代码中设置的 tcp_quat 被直接传给 IK solver 作为 wrist_3_link 的目标
3. 但 wrist_3_link 和 pinch 之间有复合旋转 R_wrist_to_pinch

当前错误：
  quat_down = [0, 1, 0, 0]  # (w,x,y,z)
  → pinch Z 轴指向: {pinch_z_current}
  → 与目标方向相差 {np.degrees(angle_diff):.2f}°

正确的四元数（两种方案）：

方案A：修正 wrist_3_link 的目标四元数
  在代码中使用：
  quat_correct = [{quat_wrist3_target[0]:.6f}, {quat_wrist3_target[1]:.6f}, {quat_wrist3_target[2]:.6f}, {quat_wrist3_target[3]:.6f}]  # (w,x,y,z)
  
  替换：
  ee_quat_tcp = torch.tensor([ee_quat], device=device, dtype=torch.float32)
  
  为：
  ee_quat_wrist = torch.tensor([quat_correct], device=device, dtype=torch.float32)
  
  然后直接传给 IK solver（不使用 ee_pose_from_tcp_pose）

方案B：修复 ee_pose_from_tcp_pose 函数
  让它同时转换位置和姿态：
  ee_quat = quat_multiply(tcp_quat, quat_inverse(tcp_rel_quat))
  
  其中 tcp_rel_quat 是从 R_wrist_to_pinch 计算出的四元数

推荐方案A，因为：
1. 不需要修改框架代码
2. 更直观，直接设置 wrist_3_link 的目标姿态
3. 避免了四元数乘法的复杂性
""")

print("\n" + "=" * 70)
print("代码修改建议")
print("=" * 70)
print(f"""
在 get_trajectory_targets() 函数中：

# 当前代码（错误）：
quat_down = [0.0, 1.0, 0.0, 0.0]  # 这是 wrist_3_link 的姿态，但会导致 pinch 指向错误方向
ee_quat_tcp = torch.tensor([quat_down], device=device, dtype=torch.float32)
ee_pos_target, ee_quat_target = ee_pose_from_tcp_pose(robot_cfg, ee_pos_tcp, ee_quat_tcp)

# 修正方案（正确）：
quat_wrist3 = [{quat_wrist3_target[0]:.6f}, {quat_wrist3_target[1]:.6f}, {quat_wrist3_target[2]:.6f}, {quat_wrist3_target[3]:.6f}]  # (w,x,y,z)
ee_pos_tcp = torch.tensor([ee_pos], device=device, dtype=torch.float32)
ee_quat_wrist = torch.tensor([quat_wrist3], device=device, dtype=torch.float32)

# 只转换位置，姿态已经是 wrist_3_link 的
tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(device)
ee_pos_target = ee_pos_tcp + torch.matmul(matrix_from_quat(ee_quat_wrist), -tcp_rel_pos.unsqueeze(-1)).squeeze()
ee_quat_target = ee_quat_wrist

return ee_pos_target, ee_quat_target
""")
