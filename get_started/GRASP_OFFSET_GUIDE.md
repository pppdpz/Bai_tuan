# 抓取点位偏移功能使用指南

## 功能概述

已为 `4_motion_planning_baituan.py` 添加了抓取点位偏移功能，支持：
- ✅ X/Y/Z 三轴独立偏移
- ✅ 沿太阳能板法向偏移（可选）
- ✅ 命令行参数快速调整
- ✅ 详细的调试日志输出

## 快速开始

### 1. 默认配置（推荐首次测试）

```bash
cd RoboVerse
python get_started/4_motion_planning_baituan.py --headless True
```

**默认参数：**
- `grasp_offset_x = 0.0` (无X轴偏移)
- `grasp_offset_y = 0.0` (无Y轴偏移)
- `grasp_offset_z = -0.02` (向下偏移2cm)
- `use_normal_offset = False` (禁用法向偏移)

### 2. 调整 Z 轴偏移

```bash
# 向下偏移 3cm（增加偏移）
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.03

# 向下偏移 1cm（减少偏移）
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.01

# 无偏移（直接对准 panel_tip）
python get_started/4_motion_planning_baituan.py --grasp_offset_z 0.0

# 向上偏移 1cm（正值）
python get_started/4_motion_planning_baituan.py --grasp_offset_z 0.01
```

### 3. 组合 X/Y/Z 偏移

```bash
# 向右1cm、向前1cm、向下2cm
python get_started/4_motion_planning_baituan.py \
    --grasp_offset_x 0.01 \
    --grasp_offset_y -0.01 \
    --grasp_offset_z -0.02
```

### 4. 启用法向偏移（高级功能）

```bash
# 沿太阳能板法向外侧偏移 1cm
python get_started/4_motion_planning_baituan.py \
    --use_normal_offset True \
    --normal_offset_dist 0.01

# 组合使用：Z轴偏移 + 法向偏移
python get_started/4_motion_planning_baituan.py \
    --grasp_offset_z -0.02 \
    --use_normal_offset True \
    --normal_offset_dist 0.015
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--grasp_offset_x` | float | 0.0 | X轴偏移（米），正值向右 |
| `--grasp_offset_y` | float | 0.0 | Y轴偏移（米），正值向前 |
| `--grasp_offset_z` | float | -0.02 | Z轴偏移（米），负值向下 |
| `--use_normal_offset` | bool | False | 是否启用法向偏移 |
| `--normal_offset_dist` | float | 0.01 | 法向偏移距离（米） |

## 日志输出解读

运行时会在步数 0 输出详细的抓取点位计算信息：

```
============================================================
抓取点位计算详情:
============================================================
原始 panel_tip: [0.3000, 0.2400, 1.2800]
直角坐标偏移: [0.0000, 0.0000, -0.0200]
法向偏移: 禁用
最终抓取点: [0.3000, 0.2400, 1.2600]
总偏移距离: 20.00mm
============================================================
```

**关键信息：**
- **原始 panel_tip**: 从 MuJoCo 读取的 `panel4_tip_L` 位置
- **直角坐标偏移**: 你设置的 X/Y/Z 偏移量
- **法向偏移**: 是否启用及偏移方向
- **最终抓取点**: 计算后的实际抓取目标
- **总偏移距离**: 偏移的欧氏距离

## 调试流程

### 步骤 1: 验证基础功能

```bash
# 测试无偏移
python get_started/4_motion_planning_baituan.py --grasp_offset_z 0.0
```

观察：
- 机器人是否能到达 panel_tip 位置
- 夹爪是否对准太阳能板边缘

### 步骤 2: 调整 Z 轴偏移

```bash
# 从小到大测试不同偏移量
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.01
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.02
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.03
```

观察：
- 夹爪是否能准确夹住太阳能板
- 是否有碰撞或滑脱

### 步骤 3: 微调 X/Y 偏移（如需要）

```bash
# 如果夹爪左右偏移，调整 X
python get_started/4_motion_planning_baituan.py \
    --grasp_offset_x 0.005 \
    --grasp_offset_z -0.02

# 如果夹爪前后偏移，调整 Y
python get_started/4_motion_planning_baituan.py \
    --grasp_offset_y -0.005 \
    --grasp_offset_z -0.02
```

### 步骤 4: 启用法向偏移（可选）

```bash
# 测试法向偏移
python get_started/4_motion_planning_baituan.py \
    --use_normal_offset True \
    --normal_offset_dist 0.01
```

观察日志中的 "太阳能板法向量" 输出，确认方向是否正确。

## 常见问题

### Q1: 如何知道偏移量是否合适？

**A:** 观察以下指标：
1. 步数 70 时，TCP 是否准确到达目标位置（查看日志）
2. 步数 90 时，夹爪是否成功夹住太阳能板
3. 步数 170 时，太阳能板是否顺利展开

### Q2: 法向偏移的方向如何确定？

**A:** 
- 正值：沿法向外侧偏移（远离太阳能板表面）
- 负值：沿法向内侧偏移（靠近太阳能板表面）
- 如果方向相反，修改 `normal_offset_dist` 的符号

### Q3: 偏移后机器人无法到达目标？

**A:** 可能原因：
1. 偏移量过大，超出机器人工作空间
2. IK 求解失败（查看日志中的 "✗ IK求解失败" 提示）
3. 关节角度超出安全范围（查看第二关节角度警告）

**解决方法：**
- 减小偏移量
- 调整机器人初始位置
- 检查关节限位

### Q4: 如何保存最佳参数？

**A:** 修改代码中的默认值（第 66-70 行）：

```python
grasp_offset_x: float = 0.01     # 你的最佳X偏移
grasp_offset_y: float = -0.005   # 你的最佳Y偏移
grasp_offset_z: float = -0.025   # 你的最佳Z偏移
```

## 高级技巧

### 1. 批量测试不同偏移量

创建测试脚本：

```bash
#!/bin/bash
for z_offset in 0.0 -0.01 -0.02 -0.03 -0.04; do
    echo "测试 Z 偏移: $z_offset"
    python get_started/4_motion_planning_baituan.py \
        --grasp_offset_z $z_offset \
        --headless True
    mv get_started/output/4_motion_planning_baituan_mujoco.mp4 \
       get_started/output/test_z${z_offset}.mp4
done
```

### 2. 动态调整偏移（代码修改）

如果需要在运行时根据传感器数据动态调整偏移，修改 `get_trajectory_targets` 函数：

```python
# 在第 400 行附近添加
if step == 50:  # 在接近阶段
    # 读取传感器数据
    # sensor_data = ...
    # 动态调整偏移
    offset_xyz[2] += sensor_correction
```

### 3. 可视化偏移效果

在 MuJoCo 中添加可视化标记（需要修改 XML）：

```xml
<site name="grasp_target" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
```

然后在代码中更新位置：

```python
grasp_target_site_id = mj.mj_name2id(_model, mj.mjtObj.mjOBJ_SITE, "grasp_target")
_data.site_xpos[grasp_target_site_id] = grasp_pos_pinch_np
```

## 总结

**推荐的调试顺序：**
1. ✅ 先禁用法向偏移（`use_normal_offset=False`）
2. ✅ 仅调整 Z 轴偏移，找到最佳值
3. ✅ 如需要，微调 X/Y 偏移
4. ✅ 最后考虑启用法向偏移（高级功能）

**关键指标：**
- 位置误差 < 5mm
- 抓取成功率 > 90%
- 展开过程平滑无碰撞

祝调试顺利！🚀
