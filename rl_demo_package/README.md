# RL Decision Environment Demo

强化学习决策环境演示项目

## 项目结构

- `env.py` - 自定义Gymnasium环境（上下左右平移动作）
- `train.py` - 训练PPO模型
- `evaluate.py` - 评估模型性能
- `analyze_performance.py` - 详细性能分析
- `visualize_trajectories.py` - 可视化轨迹
- `demo_env.py` - 环境演示脚本
- `requirements.txt` - 依赖包列表
- `ppo_decision.zip` - 训练好的模型文件

## 环境说明

- **观察空间**: 6维 (ego位置(2) + destination位置(2) + obs位置(2))
- **动作空间**: 4个离散动作 (上、下、左、右)
- **移动方式**: 每次动作固定移动1.0单位距离
- **目标**: 从起点到达目标点，同时避开障碍物

## 使用方法

1. 安装依赖: `pip install -r requirements.txt`
2. 训练模型: `python train.py`
3. 评估模型: `python evaluate.py`
4. 分析性能: `python analyze_performance.py`
5. 可视化轨迹: `python visualize_trajectories.py`

## 模型性能

- 成功率: 100%
- 碰撞率: 0%
- 超时率: 0%

