"""诊断脚本：直接用 MuJoCo 加载场景，计算不同关节角度下的 TCP 位置"""

import mujoco as mj
import numpy as np

# 直接加载场景 XML
model = mj.MjModel.from_xml_path("/home/e0/RoboVerse/asset_baituan/example_scene_y.xml")
data = mj.MjData(model)

robot_base = np.array([1.0, 0.0, 0.895])

# 获取关键 body/site ID
wrist_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
pinch_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pinch")
tcp_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "tcp_link")
panel_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "panel4_tip_L")
panel_r_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "panel4_tip")

print(f"Body/Site IDs: wrist={wrist_bid}, pinch={pinch_sid}, tcp={tcp_bid}, panel_L={panel_sid}, panel_R={panel_r_sid}")

# ===== 测试多组关节角度 =====
test_configs = {
    "截图工作姿态": {
        "shoulder_pan_joint": 0.0628,
        "shoulder_lift_joint": -0.754,
        "elbow_joint": -1.29,
        "wrist_1_joint": 0.503,
        "wrist_2_joint": -0.0628,
        "wrist_3_joint": 0.0,
    },
    "全零初始": {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": 0.0,
        "elbow_joint": 0.0,
        "wrist_1_joint": 0.0,
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    },
    "建议初始(紧凑)": {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.0,
        "elbow_joint": -1.5,
        "wrist_1_joint": 0.5,
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    },
    "肩抬45度": {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -0.785,
        "elbow_joint": -1.57,
        "wrist_1_joint": 0.785,
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    },
}

print("=" * 70)
print(f"UR5e 工作空间诊断 - 机器人底座: {robot_base}")
print("=" * 70)

for config_name, joint_angles in test_configs.items():
    # 重置所有 qpos
    data.qpos[:] = 0.0
    
    # 设置关节角度
    for jname, angle in joint_angles.items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle
    
    mj.mj_forward(model, data)
    
    wrist_pos = data.xpos[wrist_bid].copy()
    pinch_pos = data.site_xpos[pinch_sid].copy()
    tcp_pos = data.xpos[tcp_bid].copy()
    
    pinch_base = pinch_pos - robot_base
    dist_pinch = np.linalg.norm(pinch_base)
    
    print(f"\n--- {config_name} ---")
    print(f"  joints: sh_lift={joint_angles['shoulder_lift_joint']:.3f}, elbow={joint_angles['elbow_joint']:.3f}")
    print(f"  wrist3 世界: [{wrist_pos[0]:.4f}, {wrist_pos[1]:.4f}, {wrist_pos[2]:.4f}]")
    print(f"  pinch  世界: [{pinch_pos[0]:.4f}, {pinch_pos[1]:.4f}, {pinch_pos[2]:.4f}]")
    print(f"  pinch  base: [{pinch_base[0]:.4f}, {pinch_base[1]:.4f}, {pinch_base[2]:.4f}]")
    print(f"  pinch 距base: {dist_pinch:.4f}m")

# ===== 太阳能板位置 =====
print("\n" + "=" * 70)
print("太阳能板关键位置")
print("=" * 70)

# 未折叠
data.qpos[:] = 0.0
mj.mj_forward(model, data)
unfolded_pos = data.site_xpos[panel_sid].copy()
unfolded_r = data.site_xpos[panel_r_sid].copy()
print(f"展开 panel4_tip_L: [{unfolded_pos[0]:.4f}, {unfolded_pos[1]:.4f}, {unfolded_pos[2]:.4f}]")
print(f"展开 panel4_tip:   [{unfolded_r[0]:.4f}, {unfolded_r[1]:.4f}, {unfolded_r[2]:.4f}]")
print(f"  base坐标: [{(unfolded_pos-robot_base)[0]:.4f}, {(unfolded_pos-robot_base)[1]:.4f}, {(unfolded_pos-robot_base)[2]:.4f}]")
print(f"  距base: {np.linalg.norm(unfolded_pos-robot_base):.4f}m")

# 折叠
fold_angles = {
    "hinge_1": -1.5708, "hinge_2": 3.1416, "hinge_3": -3.1416, "hinge_4": 1.5708,
    "hinge_1_L": -1.5708, "hinge_2_L": 3.1416, "hinge_3_L": -3.1416, "hinge_4_L": 1.5708,
}
for jname, angle in fold_angles.items():
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = angle
mj.mj_forward(model, data)
folded_pos = data.site_xpos[panel_sid].copy()
folded_r = data.site_xpos[panel_r_sid].copy()
print(f"\n折叠 panel4_tip_L: [{folded_pos[0]:.4f}, {folded_pos[1]:.4f}, {folded_pos[2]:.4f}]")
print(f"折叠 panel4_tip:   [{folded_r[0]:.4f}, {folded_r[1]:.4f}, {folded_r[2]:.4f}]")
print(f"  base坐标: [{(folded_pos-robot_base)[0]:.4f}, {(folded_pos-robot_base)[1]:.4f}, {(folded_pos-robot_base)[2]:.4f}]")
print(f"  距base: {np.linalg.norm(folded_pos-robot_base):.4f}m")

# ===== 建议 waypoints =====
# 恢复截图姿态读 pinch
data.qpos[:] = 0.0
for jname, angle in test_configs["截图工作姿态"].items():
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = angle
mj.mj_forward(model, data)
screenshot_pinch = data.site_xpos[pinch_sid].copy()

print("\n" + "=" * 70)
print("建议 waypoints (pinch 坐标)")
print("=" * 70)
print(f"截图姿态 pinch: [{screenshot_pinch[0]:.4f}, {screenshot_pinch[1]:.4f}, {screenshot_pinch[2]:.4f}]")
print(f"折叠 tip_L:     [{folded_pos[0]:.4f}, {folded_pos[1]:.4f}, {folded_pos[2]:.4f}]")
print(f"展开 tip_L:     [{unfolded_pos[0]:.4f}, {unfolded_pos[1]:.4f}, {unfolded_pos[2]:.4f}]")

# 中间点: 在截图姿态和抓取位置之间找合理的过渡
mid_y = (screenshot_pinch[1] + folded_pos[1]) / 2
mid_z = max(folded_pos[2], screenshot_pinch[2]) + 0.10

print(f"\n推荐 waypoint 值:")
print(f"  initial_pos_pinch  = [{screenshot_pinch[0]:.4f}, {screenshot_pinch[1]:.4f}, {screenshot_pinch[2]:.4f}]")
print(f"  waypoint_pos_pinch = [{folded_pos[0]:.2f}, {mid_y:.2f}, {mid_z:.2f}]")
print(f"  approach_pos_pinch = [{folded_pos[0]:.2f}, {folded_pos[1]:.2f}, {folded_pos[2]+0.15:.2f}]")
print(f"  grasp_pos_pinch    = [{folded_pos[0]:.4f}, {folded_pos[1]:.4f}, {folded_pos[2]:.4f}]")
print(f"  deploy_pos_pinch   = [{unfolded_pos[0]:.4f}, {unfolded_pos[1]:.4f}, {unfolded_pos[2]:.4f}]")
print(f"  retract_pos_pinch  = [{unfolded_pos[0]:.2f}, {unfolded_pos[1]:.2f}, {unfolded_pos[2]+0.20:.2f}]")

print("\n诊断完成!")
