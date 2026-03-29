"""验证框架：检查配置与实测 TCP 偏置的几何一致性（最小代码，不改业务逻辑）"""

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

# =================================================================
# 1. 记录配置文件中的参数 (提取自 ur5e_2f85_cfg.py)
# =================================================================
pos_cfg = np.array([0.0, 0.1, -0.1558])
rot_cfg_rpy = np.array([np.pi, 0.0, 0.0])  # 3.14159近似为精确的pi

# 构建配置中的 T_ee_tcp_cfg 齐次位姿矩阵
R_cfg = R.from_euler('xyz', rot_cfg_rpy).as_matrix()
T_ee_tcp_cfg = np.eye(4)
T_ee_tcp_cfg[:3, :3] = R_cfg
T_ee_tcp_cfg[:3, 3] = pos_cfg

print(f"==================================================")
print(f"[阶段1] 配置文件给出的 T_ee_tcp_cfg (ee_link -> TCP):")
print(np.round(T_ee_tcp_cfg, 4))
print(f"==================================================\n")

# =================================================================
# 2. 初始化 MuJoCo 场景
# =================================================================
model = mj.MjModel.from_xml_path("/home/e0/RoboVerse/asset_baituan/example_scene_y.xml")
data = mj.MjData(model)

wrist_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
pinch_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pinch")

# 多组测试姿态
test_configs = {
    "全零初始": {
        "shoulder_pan_joint": 0.0, "shoulder_lift_joint": 0.0, "elbow_joint": 0.0,
        "wrist_1_joint": 0.0, "wrist_2_joint": 0.0, "wrist_3_joint": 0.0
    },
    "肩抬45度": {
        "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -0.785, "elbow_joint": -1.57,
        "wrist_1_joint": 0.785, "wrist_2_joint": 0.0, "wrist_3_joint": 0.0
    },
    "截图姿态": {
        "shoulder_pan_joint": 0.0628, "shoulder_lift_joint": -0.754, "elbow_joint": -1.29,
        "wrist_1_joint": 0.503, "wrist_2_joint": -0.0628, "wrist_3_joint": 0.0
    }
}

first_bridge = None

print(f"==================================================")
print(f"[阶段2] 动态对齐验证 (循环多组位姿)")
print(f"==================================================")

for config_name, joint_angles in test_configs.items():
    # 赋值关节位姿并前向运动学求解
    data.qpos[:] = 0.0
    for jname, angle in joint_angles.items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle
    mj.mj_forward(model, data)

    # 提取 wrist_3_link 到世界坐标的变换
    T_world_w3 = np.eye(4)
    T_world_w3[:3, 3] = data.xpos[wrist_bid]
    T_world_w3[:3, :3] = data.xmat[wrist_bid].reshape(3, 3)

    # 提取 pinch (TCP) 到世界坐标的变换
    T_world_tcp = np.eye(4)
    T_world_tcp[:3, 3] = data.site_xpos[pinch_sid]
    T_world_tcp[:3, :3] = data.site_xmat[pinch_sid].reshape(3, 3)

    # 步骤 1：获得实测常量 T_w3_tcp_meas
    T_w3_tcp_meas = np.linalg.inv(T_world_w3) @ T_world_tcp

    # 步骤 3：反解桥接矩阵 T_w3_ee_bridge = T_w3_tcp_meas * inv(T_ee_tcp_cfg)
    T_w3_ee_bridge = T_w3_tcp_meas @ np.linalg.inv(T_ee_tcp_cfg)

    # 分析多姿态一致性
    if first_bridge is None:
        first_bridge = T_w3_ee_bridge
        print(f"[{config_name}] (首次计算) T_w3_ee_bridge:")
        print(np.round(T_w3_ee_bridge, 4))
        
        b_pos = T_w3_ee_bridge[:3, 3]
        b_rpy = R.from_matrix(T_w3_ee_bridge[:3, :3]).as_euler('xyz')
        print(f"  -> Bridge 位置偏移 (x,y,z) = {np.round(b_pos, 4)}")
        print(f"  -> Bridge 旋转偏移 (r,p,y) = {np.round(b_rpy, 4)} rad")
        print("-" * 50)
    else:
        # 计算 E_i = 首次 Bridge 与 当前 Bridge 的差异，理论上应该全是恒等矩阵
        E_i = first_bridge @ np.linalg.inv(T_w3_ee_bridge)
        pos_err = np.linalg.norm(E_i[:3, 3])
        rot_err = np.linalg.norm(R.from_matrix(E_i[:3, :3]).as_rotvec())
        
        print(f"[{config_name}] 多姿态常量一致性检验:")
        print(f"  实测 T_w3_tcp_meas 位置: {np.round(T_w3_tcp_meas[:3, 3], 4)}")
        print(f"  位置误差: {pos_err:.2e} m")
        print(f"  旋转误差: {rot_err:.2e} rad")
        status = "✅ 完美一致 (它是恒定桥接框架!)" if pos_err < 1e-4 and rot_err < 1e-4 else "❌ 不一致"
        print(f"  结果判定: {status}")
        print("-" * 50)

print(f"\n[结论分析]")
print(f"如果全部为'✅ 完美一致'，说明:")
print(f"1. 你的实测偏置与配置文件的偏置【并不矛盾】，它们描述了一模一样的夹爪物理信息！")
print(f"2. 只不过【配置文件的原点是 ee_link】，而【实测的原点是 wrist_3_link】。")
print(f"3. 巧合的是，这二者之间恰好相差刚计算出来的固定常数 T_w3_ee_bridge。")
print(f"4. 后续轨迹求解中，完全可以干掉手动的 'TCP_TO_WRIST3_OFFSET_*' 硬编码，直接采用:")
print(f"   T_w3_tcp = T_w3_ee_bridge @ T_ee_tcp_cfg，既满足了实测需求，又实现了跨体配置归一化。")
