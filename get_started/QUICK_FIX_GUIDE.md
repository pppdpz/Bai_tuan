# 快速修复指南：IK 求解器未到达最外端

## 问题总结

根据 FK 分析：
- **Y=0.24** 是正确的最外端目标（panel_4_L 悬空尖端）
- **Y=0.55** 是错误的（靠近主体根部）
- IK 求解器可能在 Y≈0.46 处停止（内侧铰链群位置）

## 已添加的诊断功能

我已经在 `4_motion_planning_baituan.py` 中添加了 IK 收敛性验证，会在关键步骤（0, 50, 70, 90, 120, 200, 220, 250）输出：

```
IK 收敛性验证 (Step 70)
目标 TCP 位置: [0.3000, 0.2400, 1.2800]
IK 求解到达: [0.3000, 0.4600, 1.2800]  ← 如果这里 Y≠0.24，说明失败
位置误差: 220.00 mm
Y 轴误差: 220.00 mm
❌ Y 轴误差过大！未到达最外端悬空点
```

## 运行诊断

```bash
cd RoboVerse/get_started
python 4_motion_planning_baituan.py --solver pyroki
```

## 如果 IK 确实失败了，尝试以下方案：

### 方案 1：切换到 curobo 求解器（推荐）

```bash
python 4_motion_planning_baituan.py --solver curobo
```

curobo 通常有更好的收敛性和更多的优化选项。

### 方案 2：调整抓取高度

如果 Y=0.24 在当前高度 Z=1.28 无法到达，尝试调整 Z：

```bash
# 降低 2cm
python 4_motion_planning_baituan.py --grasp_offset_z -0.04

# 或升高 2cm
python 4_motion_planning_baituan.py --grasp_offset_z 0.02
```

### 方案 3：调整 Y 位置（微调）

如果 Y=0.24 完全无法到达，可以尝试稍微内移：

```bash
# 内移 2cm（Y 从 0.24 → 0.26）
python 4_motion_planning_baituan.py --grasp_offset_y 0.02

# 内移 5cm（Y 从 0.24 → 0.29）
python 4_motion_planning_baituan.py --grasp_offset_y 0.05
```

**注意**：根据 FK 分析，Y=0.42 是内侧铰链群，所以不要超过 0.10 的偏移（即不要超过 Y=0.34）。

### 方案 4：检查关节限位

查看 MJCF 文件中的关节限位：

```bash
grep -A 2 "shoulder_lift_joint\|elbow_joint" asset_baituan/example_scene_y.xml
```

如果限位过严（例如 shoulder_lift 只能到 -0.5 rad），可能需要调整。

### 方案 5：分阶段逼近（手动实现）

如果单步跨度太大，可以在代码中添加更多中间点。编辑 `get_trajectory_targets` 函数，在阶段 1 和阶段 2 之间添加更多 waypoints。

## 预期结果

如果 IK 收敛良好，你应该看到：

```
IK 收敛性验证 (Step 70)
目标 TCP 位置: [0.3000, 0.2400, 1.2800]
IK 求解到达: [0.3000, 0.2398, 1.2802]
位置误差: 2.24 mm
Y 轴误差: 0.20 mm
✓ IK 收敛良好，误差在可接受范围内
```

## 理解坐标系

```
Y 轴方向（从机器人视角）：
  0.60 ← 卫星主体
  0.52 ← 连接根部
  0.42 ← 内侧铰链群（Panel 1-3）
  0.24 ← 最外端悬空点（Panel 4 尖端）⭐ 目标
  0.00 ← 机器人 base
```

## 下一步

1. 运行诊断脚本
2. 查看 "IK 收敛性验证" 输出
3. 如果 Y 轴误差 >5cm，尝试上述方案
4. 如果所有方案都失败，可能需要：
   - 调整机器人 base 位置（更靠近卫星）
   - 使用轨迹优化而非逐步 IK
   - 检查是否有碰撞检测阻止到达

## 联系支持

如果问题持续，请提供：
- 完整的诊断输出（特别是 Step 70 的 IK 收敛性验证）
- 使用的求解器（pyroki 或 curobo）
- 任何错误消息或警告
