"""
机器人姿态问题诊断工具

问题：ee_pose_from_tcp_pose() 只转换位置不转换姿态，导致 IK solver 收到错误的目标姿态
解决：计算正确的 wrist_3_link 四元数，使 pinch 垂直向下
"""

import numpy as np


def matrix_from_quat(q):
    """四元数 (w,x,y,z) 转旋转矩阵"""
    w, x, y, z = q
    two_s = 2.0 / (q @ q)
    return np.array([
        [1 - two_s * (y*y + z*z), two_s * (x*y - z*w), two_s * (x*z + y*w)],
        [two_s * (x*y + z*w), 1 - two_s * (x*x + z*z), two_s * (y*z - x*w)],
        [two_s * (x*z - y*w), two_s * (y*z + x*w), 1 - two_s * (x*x + y*y)]
    ])


def quat_from_matrix(R):
    """旋转矩阵转四元数 (w,x,y,z)，使用 Shepperd's method"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s, (R[2,1] - R[1,2]) * s, (R[0,2] - R[2,0]) * s, (R[1,0] - R[0,1]) * s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1] - R[1,2]) / s, 0.25 * s, (R[0,1] + R[1,0]) / s, (R[0,2] + R[2,0]) / s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2] - R[2,0]) / s, (R[0,1] + R[1,0]) / s, 0.25 * s, (R[1,2] + R[2,1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0] - R[0,1]) / s, (R[0,2] + R[2,0]) / s, (R[1,2] + R[2,1]) / s, 0.25 * s])


# 从 MJCF 计算 wrist_3_link → pinch 的复合旋转
flange_quat = np.array([1, -1, 0, 0]) / np.linalg.norm([1, -1, 0, 0])
base_quat = np.array([1, 0, 0, -1]) / np.linalg.norm([1, 0, 0, -1])
R_wrist_to_pinch = matrix_from_quat(flange_quat) @ matrix_from_quat(base_quat)

# 目标：pinch Z 轴指向 [0, 0, -1]（垂直向下）
R_pinch_target = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

# 反推 wrist_3_link 的目标姿态
R_wrist3_target = R_pinch_target @ np.linalg.inv(R_wrist_to_pinch)
quat_wrist3_target = quat_from_matrix(R_wrist3_target)

# 验证当前错误值
quat_down = np.array([0, 1, 0, 0])
R_pinch_current = matrix_from_quat(quat_down) @ R_wrist_to_pinch
pinch_z_current = R_pinch_current @ np.array([0, 0, 1])
angle_diff = np.degrees(np.arccos(np.clip(pinch_z_current @ np.array([0, 0, -1]), -1, 1)))

# 输出结果
print("=" * 70)
print("姿态问题诊断结果")
print("=" * 70)
print(f"\n当前错误值: quat_down = [0, 1, 0, 0]")
print(f"  → pinch Z 轴指向: {pinch_z_current}")
print(f"  → 与目标方向 [0, 0, -1] 相差: {angle_diff:.2f}°")
print(f"\n正确的 wrist_3_link 四元数 (w,x,y,z):")
print(f"  {quat_wrist3_target}")
print(f"\n验证：")
R_verify = matrix_from_quat(quat_wrist3_target) @ R_wrist_to_pinch
print(f"  pinch Z 轴: {R_verify @ np.array([0, 0, 1])}")
print(f"  误差: {np.linalg.norm(R_verify @ np.array([0, 0, 1]) - np.array([0, 0, -1])):.10f}")

print("\n" + "=" * 70)
print("代码修改方案")
print("=" * 70)
print(f"""
在 get_trajectory_targets() 中替换：

# 错误写法：
quat_down = [0.0, 1.0, 0.0, 0.0]
ee_quat_tcp = torch.tensor([quat_down], device=device, dtype=torch.float32)
ee_pos_target, ee_quat_target = ee_pose_from_tcp_pose(robot_cfg, ee_pos_tcp, ee_quat_tcp)

# 正确写法：
quat_wrist3 = [{quat_wrist3_target[0]:.6f}, {quat_wrist3_target[1]:.6f}, 
               {quat_wrist3_target[2]:.6f}, {quat_wrist3_target[3]:.6f}]
ee_quat_wrist = torch.tensor([quat_wrist3], device=device, dtype=torch.float32)
tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(device)
ee_pos_target = ee_pos_tcp + torch.matmul(matrix_from_quat(ee_quat_wrist), 
                                           -tcp_rel_pos.unsqueeze(-1)).squeeze()
ee_quat_target = ee_quat_wrist
""")
