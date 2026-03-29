#!/bin/bash
# 测试抓取点位偏移参数的脚本

echo "=========================================="
echo "抓取点位偏移测试脚本"
echo "=========================================="
echo ""

# 测试1: 默认配置（Z轴向下偏移2cm，法向偏移禁用）
echo "测试1: 默认配置 (Z=-0.02m, 法向偏移禁用)"
echo "------------------------------------------"
python RoboVerse/get_started/4_motion_planning_baituan.py \
    --headless True \
    --sim mujoco

echo ""
echo "测试1完成！视频保存在: get_started/output/4_motion_planning_baituan_mujoco.mp4"
echo ""

# 测试2: 增加Z轴偏移到3cm
echo "测试2: 增加Z轴偏移 (Z=-0.03m)"
echo "------------------------------------------"
python RoboVerse/get_started/4_motion_planning_baituan.py \
    --headless True \
    --sim mujoco \
    --grasp_offset_z -0.03

echo ""
echo "测试2完成！"
echo ""

# 测试3: 减少Z轴偏移到1cm
echo "测试3: 减少Z轴偏移 (Z=-0.01m)"
echo "------------------------------------------"
python RoboVerse/get_started/4_motion_planning_baituan.py \
    --headless True \
    --sim mujoco \
    --grasp_offset_z -0.01

echo ""
echo "测试3完成！"
echo ""

# 测试4: 无偏移（直接对准panel_tip）
echo "测试4: 无偏移 (Z=0.0m)"
echo "------------------------------------------"
python RoboVerse/get_started/4_motion_planning_baituan.py \
    --headless True \
    --sim mujoco \
    --grasp_offset_z 0.0

echo ""
echo "测试4完成！"
echo ""

# 测试5: 组合偏移（X/Y/Z）
echo "测试5: 组合偏移 (X=0.01m, Y=-0.01m, Z=-0.02m)"
echo "------------------------------------------"
python RoboVerse/get_started/4_motion_planning_baituan.py \
    --headless True \
    --sim mujoco \
    --grasp_offset_x 0.01 \
    --grasp_offset_y -0.01 \
    --grasp_offset_z -0.02

echo ""
echo "测试5完成！"
echo ""

echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo ""
echo "提示："
echo "1. 查看日志中的 '抓取点位计算详情' 部分"
echo "2. 对比不同偏移量下的抓取效果"
echo "3. 根据实际效果调整偏移参数"
echo ""
