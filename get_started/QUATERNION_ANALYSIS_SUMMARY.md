# 抓取姿态问题深度分析报告

## 问题现象

机械臂抓取小卫星太阳翼时，姿态与预期相差 **90°**，导致夹爪方向不正确。

## 根本原因

### 1. 核心问题：`ee_pose_from_tcp_pose()` 函数的设计缺陷

```python
def ee_pose_from_tcp_pose(robot_cfg, tcp_pos, tcp_quat):
    """Calculate the end-effector (EE) pose from the tool center point (TCP) pose.
    
    Note that currently only the translation is considered.  # ← 关键注释！
    """
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos)
    ee_pos = tcp_pos + torch.matmul(matrix_from_quat(tcp_quat), -tcp_rel_pos.unsqueeze(-1)).squeeze()
    return ee_pos, tcp_quat  # ← 姿态直接返回，没有转换！
```

**问题**：
- 函数只转换了**位置** (ee_pos)
- 姿态 (tcp_quat) 被**原封不动**地返回
- 这导致 IK solver 收到的是 **TCP/pinch 的姿态**，而不是 **wrist_3_link 的姿态**

### 2. 坐标系变换链

从 MJCF 文件解析的坐标系层级：

```
wrist_3_link (IK solver 的目标)
  └─ flange (quat="1 -1 0 0")
      └─ robotiq_2f85
          └─ 2f85_base
              └─ base (quat="1 0 0 -1")
                  └─ pinch (pos="0 0 0.145")
```

**复合旋转矩阵** (wrist_3_link → pinch)：

```
R_wrist_to_pinch = R_flange @ R_base = 
[[0, 1, 0],
 [0, 0, 1],
 [1, 0, 0]]
```

**含义**：
- wrist_3_link 的 X 轴 → pinch 的 Z 轴
- wrist_3_link 的 Y 轴 → pinch 的 X 轴
- wrist_3_link 的 Z 轴 → pinch 的 Y 轴

### 3. 当前代码的错误

```python
quat_down = [0.0, 1.0, 0.0, 0.0]  # (w, x, y, z)
```

这个四元数对应的旋转矩阵：
```
[[1,  0,  0],
 [0, -1,  0],
 [0,  0, -1]]
```

**当前 pinch 的 Z 轴方向**：
```
R_pinch_current = R_down @ R_wrist_to_pinch
pinch_z = [0, -1, 0]  # 指向世界坐标的 -Y 方向（水平）
```

**期望 pinch 的 Z 轴方向**：
```
pinch_z_target = [0, 0, -1]  # 指向世界坐标的 -Z 方向（垂直向下）
```

**角度差异：90°** ← 这就是问题所在！

## 正确的解决方案

### 方案 A：直接设置 wrist_3_link 的姿态（推荐）

**正确的四元数**：
```python
quat_wrist3 = [-0.5, 0.5, -0.5, 0.5]  # (w, x, y, z)
```

**验证**：
- wrist_3_link 旋转矩阵：
  ```
  [[0,  0,  1],
   [-1, 0,  0],
   [0, -1,  0]]
  ```
- pinch 最终旋转矩阵：
  ```
  [[1,  0,  0],
   [0, -1,  0],
   [0,  0, -1]]
  ```
- pinch Z 轴：`[0, 0, -1]` ✓ 正确！

**代码修改**：

```python
def get_trajectory_targets(step, device, robot_cfg):
    # ... 位置计算保持不变 ...
    
    # 修改前（错误）：
    # quat_down = [0.0, 1.0, 0.0, 0.0]
    # ee_quat = quat_down
    
    # 修改后（正确）：
    quat_wrist3 = [-0.5, 0.5, -0.5, 0.5]  # (w, x, y, z)
    ee_quat = quat_wrist3
    
    # ... 阶段判断保持不变 ...
    
    # 方案A：直接设置 wrist_3_link 的姿态
    ee_pos_tcp = torch.tensor([ee_pos], device=device, dtype=torch.float32)
    ee_quat_wrist = torch.tensor([ee_quat], device=device, dtype=torch.float32)
    
    # 只转换位置，姿态已经是 wrist_3_link 的
    from metasim.utils.math import matrix_from_quat
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(device)
    ee_pos_target = ee_pos_tcp + torch.matmul(
        matrix_from_quat(ee_quat_wrist), 
        -tcp_rel_pos.unsqueeze(-1)
    ).squeeze()
    ee_quat_target = ee_quat_wrist
    
    return ee_pos_target, ee_quat_target, gripper_width
```

### 方案 B：修复 `ee_pose_from_tcp_pose()` 函数

如果要保持"TCP 坐标系思维"，需要修改框架函数：

```python
def ee_pose_from_tcp_pose(robot_cfg, tcp_pos, tcp_quat):
    """Calculate the end-effector (EE) pose from the tool center point (TCP) pose."""
    
    # 位置转换（原有逻辑）
    tcp_rel_pos = torch.tensor(robot_cfg.curobo_tcp_rel_pos).unsqueeze(0).to(tcp_pos.device)
    ee_pos = tcp_pos + torch.matmul(matrix_from_quat(tcp_quat), -tcp_rel_pos.unsqueeze(-1)).squeeze()
    
    # 姿态转换（新增逻辑）
    # 需要从 MJCF 计算 tcp_rel_quat
    tcp_rel_quat = torch.tensor(robot_cfg.curobo_tcp_rel_quat).unsqueeze(0).to(tcp_quat.device)
    ee_quat = quat_multiply(tcp_quat, quat_inverse(tcp_rel_quat))
    
    return ee_pos, ee_quat
```

**缺点**：
- 需要修改框架代码
- 需要在 `robot_cfg` 中添加 `curobo_tcp_rel_quat` 配置
- 需要实现四元数乘法和求逆函数

## 四元数格式约定

根据 `metasim/utils/math.py` 中的 `matrix_from_quat()` 函数：

```python
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.
    
    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
    """
    r, i, j, k = torch.unbind(quaternions, -1)  # r=w, i=x, j=y, k=z
    # ...
```

**格式**：`[w, x, y, z]` (scalar-first)

## 关键发现总结

1. ✅ **复合旋转矩阵正确**：`[[0,1,0],[0,0,1],[1,0,0]]`
2. ✅ **卫星坐标系分析正确**：Z 轴朝上不变
3. ✅ **目标方向正确**：pinch Z 轴需要朝 -Z（从上方接近）
4. ❌ **之前计算的四元数错误**：`[0.5, -0.5, 0.5, -0.5]` 不正确
5. ✅ **正确的四元数**：`[-0.5, 0.5, -0.5, 0.5]` (w, x, y, z)
6. ⚠️ **根本原因**：`ee_pose_from_tcp_pose()` 只转换位置，不转换姿态

## 验证步骤

1. 使用 `quat_wrist3 = [-0.5, 0.5, -0.5, 0.5]`
2. 计算旋转矩阵并验证 pinch Z 轴指向 `[0, 0, -1]`
3. 运行仿真，观察夹爪是否从上方垂直接近太阳翼
4. 检查抓取过程中的姿态是否稳定

## 推荐行动

1. **立即修改**：使用方案 A，直接设置 wrist_3_link 的姿态
2. **长期优化**：考虑实现方案 B，修复框架函数以支持完整的 TCP 坐标系转换
3. **文档更新**：在代码中添加注释，说明四元数格式和坐标系关系

## 附录：四元数计算脚本

详见：
- `calculate_grasp_quaternion.py` - 初步计算（有误）
- `analyze_quaternion_issue.py` - 深入分析（正确）
