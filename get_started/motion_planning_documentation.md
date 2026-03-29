# 4_motion_planning_baituan.py 运动规划说明文档

## 1. 项目一句话概述
这是一个基于 MuJoCo 物理仿真的 UR5e 机械臂 + Robotiq 2F-85 夹爪的运动规划演示程序，用于模拟太阳能卫星板的抓取、移动和部署任务。

## 2. 项目整体定位
**项目目标**
- 实现机械臂在零重力环境下对太阳能板的精确抓取和部署
- 演示完整的运动规划流程：接近 → 抓取 → 移动 → 释放 → 撤离
- 验证 IK（逆运动学）求解器在实际任务中的可靠性

**适用场景**
- 空间机器人操作仿真
- 太阳能板展开/折叠任务验证
- 机械臂运动规划算法测试
- 多物理引擎兼容性验证（支持 MuJoCo、Isaac Sim、Genesis 等）

**目标用户**
- 机器人研究人员
- 空间机器人工程师
- 运动规划算法开发者
- 仿真系统集成工程师

## 3. 技术栈概览
**编程语言**
- Python 3.x

**核心框架/库**
- **MuJoCo**: 物理仿真引擎
- **PyTorch**: 张量计算和设备管理
- **NumPy**: 数值计算
- **SciPy**: 旋转变换（Rotation）
- **Tyro**: 命令行参数解析
- **Loguru + Rich**: 日志和终端美化

**自定义模块**
- **metasim**: 场景配置、相机、观测保存
- **inline_scene_adapter**: 场景补丁安装
- **IK Solver**: 支持 CuRobo 和 PyRoki 两种求解器

**构建与部署**
- 无需编译，直接运行 Python 脚本
- 支持 GPU 加速（CUDA）

## 4. 项目目录结构说明
```text
RoboVerse/get_started/
├── 4_motion_planning_baituan.py       # 【核心文件】运动规划主程序
├── inline_scene_adapter.py            # 场景适配器补丁
├── asset_baituan/
│   └── example_scene_y.xml            # MuJoCo 场景描述文件（卫星+机械臂）
└── output/
    └── 4_motion_planning_baituan_*.mp4  # 生成的仿真视频
```
**关键文件说明**
- `4_motion_planning_baituan.py`: 主程序，包含轨迹规划、IK求解、仿真循环
- `example_scene_y.xml`: 定义机械臂、太阳能板、夹爪等物理模型
- `inline_scene_adapter.py`: 修复场景加载兼容性问题

## 5. 核心模块拆解

### 5.1 配置模块
**`Args`**: 命令行参数配置类
- `robot`: 机器人型号（默认 ur5e_2f85）
- `sim`: 仿真器选择（mujoco/isaacgym/genesis等）
- `solver`: IK求解器（curobo/pyroki）
- `grasp_offset_*`: 抓取点偏移参数
- `use_normal_offset`: 是否使用法向量偏移

**`BaituanSceneCfg`**: 场景配置类，继承自 `SceneCfg`

### 5.2 坐标变换模块
提供完整的 3D 空间变换工具链：

| 函数名 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `quat_xyzw_to_wxyz` | 四元数格式转换 | [x,y,z,w] | [w,x,y,z] |
| `matrix_to_quat_wxyz` | 旋转矩阵转四元数 | 3x3矩阵 | [w,x,y,z] |
| `build_transform_matrix` | 构建4x4变换矩阵 | pos + quat/rpy | 4x4矩阵 |
| `world_pose_to_base_pose` | 世界坐标转base坐标 | 世界位姿 + base位姿 | base系位姿 |
| `tcp_pose_to_wrist3_pose` | TCP坐标转wrist3坐标 | TCP位姿 + 变换矩阵 | wrist3位姿 |

### 5.3 场景感知模块
- `get_site_world_pos`: 读取 MuJoCo site 的世界坐标
- `get_panel_normal_vector`: 计算太阳能板表面法向量
- `compute_grasp_point_with_offset`: 计算带偏移的抓取点
- `measure_T_w3_tcp`: 测量 wrist3 到 TCP 的变换矩阵（自检）

### 5.4 轨迹规划模块
`get_trajectory_targets` 是核心函数，定义了 9 个阶段的运动轨迹：

| 阶段 | 步数范围 | 起点 | 终点 | 夹爪状态 | 说明 |
|---|---|---|---|---|---|
| 1 | 0-30 | 初始位置 | 路径点 | 打开(1.0) | 移动到中间点 |
| 2 | 30-50 | 路径点 | 接近点 | 打开 | 接近太阳能板 |
| 3 | 50-70 | 接近点 | 抓取点 | 打开 | 精确定位 |
| 4 | 70-90 | 抓取点 | 抓取点 | 渐闭(1→0) | 闭合夹爪 |
| 5 | 90-120 | 抓取点 | 抓取点 | 闭合(0.0) | 稳定抓取 |
| 6 | 120-320 | 抓取点 | 部署点 | 闭合 | S曲线平滑移动 |
| 7 | 320-340 | 部署点 | 部署点 | 渐开(0→1) | 释放夹爪 |
| 8 | 340-370 | 部署点 | 部署点 | 打开 | 稳定释放 |
| 9 | 370-400 | 部署点 | 撤离点 | 打开 | 撤离 |

**关键特性:**
- 使用 S 曲线插值（`smooth_interpolation`）实现平滑加减速
- 动态计算抓取点（基于 site 实时位置）
- 支持法向量偏移（避免碰撞）

### 5.5 IK求解与执行模块
- `setup_ik_solver`: 初始化 IK 求解器（CuRobo 或 PyRoki）
- `solve_ik_batch`: 批量求解逆运动学
- **坐标系转换**: URDF ↔ MJCF 关节角度转换（关键差异在 joint 1 和 3）
- `process_gripper_command`: 处理夹爪指令

## 6. 核心业务流程 / 执行流程

**主流程图**
```text
[1] 初始化
    ├─ 加载场景配置（XML）
    ├─ 创建物理引擎 handler
    ├─ 初始化 IK 求解器
    └─ 折叠太阳翼到初始状态

[2] 自检与测量
    ├─ 获取关键 site/body ID
    ├─ 测量 TCP 变换矩阵（T_W3_TCP）
    └─ 验证测量有效性

[3] 主循环（450步）
    ├─ 读取太阳能板 tip 位置
    ├─ 计算当前阶段目标位姿（TCP坐标系）
    ├─ TCP → wrist3 坐标转换
    ├─ 世界坐标 → base 坐标转换
    ├─ MJCF → URDF 关节角转换
    ├─ IK 求解目标关节角
    ├─ URDF → MJCF 关节角转换
    ├─ 组合关节+夹爪动作
    ├─ 执行仿真步
    └─ 保存观测数据

[4] 清理与保存
    ├─ 保存视频（MP4）
    └─ 释放资源
```

**关键数据流**
```text
panel_tip_world_pos (MuJoCo site)
    ↓
grasp_pos (世界坐标)
    ↓
ee_pos_target (TCP坐标系)
    ↓
ee_pos_wrist3 (wrist3坐标系)
    ↓
ee_pos_base (base坐标系)
    ↓
q_sol_urdf (URDF关节角)
    ↓
q_sol_mjcf (MJCF关节角)
    ↓
actions (执行指令)
```

## 7. 关键文件重点解读

**入口文件**
`4_motion_planning_baituan.py` 的 `main()` 函数

**配置常量**
```python
# TCP到末端执行器的桥接变换
W3_TO_EE_BRIDGE_POS = [0.1, 0.1, 0.0]
W3_TO_EE_BRIDGE_RPY = [π/2, -π/2, 0.0]

# 太阳翼折叠角度（初始状态）
FOLD_ANGLES = {
    "hinge_1": -1.5708,  # -90°
    "hinge_2": 3.1416,   # 180°
    ...
}

# 机器人初始配置
ROBOT_INIT_CONFIG = {
    "pos": [1.0, 0.0, 0.895],
    "rot": [1.0, 0.0, 0.0, 0.0],  # wxyz
    "dof_pos": {...}
}
```

**关键坐标系转换**
```python
# URDF ↔ MJCF 关节角转换
curr_q_urdf[:, 1] -= π/2  # shoulder_lift
curr_q_urdf[:, 3] -= π/2  # wrist_1

q_sol_mjcf[:, 1] += π/2
q_sol_mjcf[:, 3] += π/2
```

**轨迹硬编码位置（可调整）**
```python
initial_pos = [0.2844, 0.1867, 1.1826]
waypoint_pos = [0.3, 0.07, 1.20]
approach_pos = [0.3, 0.24, 1.43]
grasp_pos = [0.3, 0.24, 1.28]  # 动态计算
deploy_pos = [0.3, -0.30, 1.30]
retract_pos = [0.3, -0.30, 1.40]
```

## 8. 项目运行方式

**环境依赖**
```bash
# 核心依赖
pip install mujoco numpy torch scipy tyro loguru rich

# 自定义模块（需在项目根目录）
# metasim, inline_scene_adapter
```

**启动方式**
```bash
# 基础运行（默认 MuJoCo + PyRoki）
python 4_motion_planning_baituan.py

# 指定仿真器
python 4_motion_planning_baituan.py --sim mujoco

# 使用 CuRobo 求解器（需 GPU）
python 4_motion_planning_baituan.py --solver curobo

# 调整抓取偏移
python 4_motion_planning_baituan.py \
    --grasp_offset_x 0.01 \
    --grasp_offset_y -0.02 \
    --grasp_offset_z 0.0

# 启用法向量偏移（避免碰撞）
python 4_motion_planning_baituan.py \
    --use_normal_offset \
    --normal_offset_dist 0.02

# 多环境并行（需支持的仿真器）
python 4_motion_planning_baituan.py --num_envs 4

# 无头模式（不显示窗口）
python 4_motion_planning_baituan.py --headless
```

**常见配置项**
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--robot` | ur5e_2f85 | 机器人型号 |
| `--sim` | mujoco | 仿真器类型 |
| `--solver` | pyroki | IK求解器 |
| `--num_envs` | 1 | 并行环境数 |
| `--headless` | True | 无头模式 |
| `--grasp_offset_x/y/z` | 0.0 | 抓取点偏移(m) |
| `--use_normal_offset` | False | 启用法向量偏移 |
| `--normal_offset_dist` | 0.01 | 法向量偏移距离(m) |

**输出文件**
- 视频: `4_motion_planning_baituan_mujoco.mp4`
- 日志: 终端实时输出（Rich格式）

## 9. 阅读建议

**推荐阅读顺序**
1. 先看常量定义（第 40-60 行）
   - 理解坐标系关系
   - 了解初始配置
2. 再看工具函数（第 63-180 行）
   - 坐标变换逻辑
   - 四元数/矩阵转换
3. 重点看轨迹规划（`get_trajectory_targets` 函数）
   - 9 个阶段的定义
   - S 曲线插值实现
4. 最后看主循环（`main` 函数）
   - 初始化流程
   - IK 求解流程
   - 坐标系转换链路

**理解该项目需要重点关注**
- **多坐标系转换**
  - 世界坐标系 → base 坐标系
  - TCP 坐标系 → wrist3 坐标系
  - URDF 关节角 ↔ MJCF 关节角
- **四元数约定**
  - MuJoCo 使用 wxyz 格式
  * SciPy 使用 xyzw 格式
  - 需要频繁转换
- **IK 求解器差异**
  - PyRoki: 纯 Python，速度较慢
  - CuRobo: GPU 加速，需 CUDA
- **轨迹平滑技术**
  - S 曲线插值（3t²-2t³）
  - 避免突变和震荡

## 10. 项目亮点与可改进点

**架构亮点**
- ✅ **多仿真器兼容**: 支持 7+ 种物理引擎，统一接口
- ✅ **模块化设计**: 坐标变换、轨迹规划、IK求解完全解耦
- ✅ **自检机制**: 启动时测量 TCP 变换矩阵，验证链路正确性
- ✅ **动态抓取点**: 基于 site 实时位置计算，适应场景变化

**工程亮点**
- ✅ **S 曲线插值**: 实现平滑加减速，避免机械冲击
- ✅ **法向量偏移**: 支持沿表面法向量偏移，避免碰撞
- ✅ **批量 IK 求解**: 支持多环境并行（虽然当前默认 1 个）
- ✅ **视频自动保存**: 使用 ObsSaver 自动录制仿真过程

**潜在问题**
- ⚠️ **硬编码轨迹**: 路径点写死在代码中，缺乏灵活性
- ⚠️ **无碰撞检测**: 未集成碰撞检测，可能出现穿模
- ⚠️ **IK 失败处理**: 当 IK 无解时，直接使用当前关节角（可能导致卡死）
- ⚠️ **坐标系转换复杂**: 多次转换增加出错风险

**可优化方向**
- 🔧 **引入路径规划器**: 使用 RRT/PRM 等算法自动生成路径
- 🔧 **添加碰撞检测**: 集成 FCL 或 MuJoCo 内置碰撞检测
- 🔧 **参数化轨迹**: 将路径点移到配置文件或 YAML
- 🔧 **增强 IK 鲁棒性**: 添加多解选择、奇异点处理
- 🔧 **可视化调试**: 添加 TCP 轨迹、抓取点的可视化标记
- 🔧 **力控支持**: 当前仅位置控制，可增加力/力矩控制

## 11. 总结
这是一个工程化程度较高的机械臂运动规划演示项目，核心价值在于：
- **完整的任务流程**: 从接近到抓取再到部署，覆盖典型操作场景
- **严谨的坐标变换**: 处理了多个坐标系之间的复杂转换关系
- **可扩展的架构**: 支持多种仿真器和 IK 求解器，便于算法对比
- **实用的工具函数**: 提供了丰富的 3D 变换工具，可复用到其他项目

**维护重点:**
- 定期验证 TCP 变换矩阵的准确性（自检机制）
- 调整轨迹参数时注意关节限位和奇异点
- 更换仿真器时需重新校准坐标系转换

适合人群: 需要快速搭建机械臂仿真原型的研究人员，或需要理解运动规划完整流程的学习者。

