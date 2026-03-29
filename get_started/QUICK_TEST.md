# 快速测试命令

## 立即开始测试

### 测试 1: 默认配置（Z轴向下偏移2cm）

```bash
cd RoboVerse
python get_started/4_motion_planning_baituan.py --headless True
```

**预期输出：**
```
抓取偏移参数: X=0.0, Y=0.0, Z=-0.02
法向偏移: 禁用, 距离=0.01m
...
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

---

### 测试 2: 增加偏移到3cm

```bash
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.03 --headless True
```

**观察：** 抓取点是否向下移动了1cm

---

### 测试 3: 减少偏移到1cm

```bash
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.01 --headless True
```

**观察：** 抓取点是否更接近 panel_tip

---

### 测试 4: 无偏移（对照组）

```bash
python get_started/4_motion_planning_baituan.py --grasp_offset_z 0.0 --headless True
```

**观察：** 抓取点应该与 panel_tip 完全重合

---

### 测试 5: 组合偏移

```bash
python get_started/4_motion_planning_baituan.py \
    --grasp_offset_x 0.01 \
    --grasp_offset_y -0.01 \
    --grasp_offset_z -0.02 \
    --headless True
```

**观察：** 抓取点在三个方向都有偏移

---

## 查看结果

### 1. 查看视频

```bash
# 视频保存位置
ls -lh get_started/output/4_motion_planning_baituan_mujoco.mp4

# 使用视频播放器打开
# Linux: vlc get_started/output/4_motion_planning_baituan_mujoco.mp4
# macOS: open get_started/output/4_motion_planning_baituan_mujoco.mp4
```

### 2. 查看关节角度日志

```bash
# CSV 文件保存位置
cat get_started/output/joint_2_angles_log_mujoco.csv

# 查看统计信息（在运行日志末尾）
```

### 3. 关键步数的调试信息

运行时会在以下步数自动打印详细信息：
- **步数 0**: 抓取点位计算详情
- **步数 50**: 阶段2开始-精确定位
- **步数 70**: 阶段3开始-抓取
- **步数 90**: 阶段4开始-展开
- **步数 170**: 阶段5开始-释放
- **步数 190**: 阶段6开始-撤离

---

## 对比不同偏移量

### 方法 1: 手动对比

```bash
# 运行测试1
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.01 --headless True
mv get_started/output/4_motion_planning_baituan_mujoco.mp4 \
   get_started/output/test_z_minus_0.01.mp4

# 运行测试2
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.02 --headless True
mv get_started/output/4_motion_planning_baituan_mujoco.mp4 \
   get_started/output/test_z_minus_0.02.mp4

# 运行测试3
python get_started/4_motion_planning_baituan.py --grasp_offset_z -0.03 --headless True
mv get_started/output/4_motion_planning_baituan_mujoco.mp4 \
   get_started/output/test_z_minus_0.03.mp4

# 对比视频
ls -lh get_started/output/test_*.mp4
```

### 方法 2: 使用测试脚本

```bash
# 给脚本添加执行权限
chmod +x get_started/test_grasp_offset.sh

# 运行批量测试
./get_started/test_grasp_offset.sh
```

---

## 判断标准

### ✅ 成功的抓取

1. **步数 70**: TCP 准确到达目标位置（误差 < 5mm）
2. **步数 90**: 夹爪成功夹住太阳能板（无滑脱）
3. **步数 170**: 太阳能板顺利展开（无卡顿）
4. **关节角度**: 第二关节在安全范围内（-2.0 ~ 0.5 rad）

### ❌ 失败的抓取

1. **IK 求解失败**: 日志中出现 "✗ IK求解失败"
2. **位置偏差过大**: 实际位置与目标位置差距 > 1cm
3. **夹爪未夹住**: 太阳能板在展开时脱落
4. **关节超限**: 出现 "⚠️ 危险" 警告

---

## 下一步

找到最佳偏移量后：

1. **保存参数**: 修改代码中的默认值
2. **测试稳定性**: 多次运行验证成功率
3. **优化轨迹**: 调整其他阶段的参数（如抓取时间、展开速度）

详细说明请查看: `GRASP_OFFSET_GUIDE.md`
