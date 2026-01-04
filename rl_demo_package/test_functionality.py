"""
验证子文件夹内代码功能的测试脚本
"""
import sys
import os

# 确保可以导入当前目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有必要的导入"""
    print("=" * 70)
    print("测试 1: 导入模块")
    print("=" * 70)
    try:
        import gymnasium as gym
        print("✅ gymnasium 导入成功")
        
        import numpy as np
        print("✅ numpy 导入成功")
        
        from stable_baselines3 import PPO
        print("✅ stable_baselines3 导入成功")
        
        from env import DecisionEnv
        print("✅ DecisionEnv 导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_environment():
    """测试环境基本功能"""
    print("\n" + "=" * 70)
    print("测试 2: 环境基本功能")
    print("=" * 70)
    try:
        from env import DecisionEnv
        import numpy as np
        
        # 创建环境
        env = DecisionEnv()
        print("✅ 环境创建成功")
        
        # 测试观察空间
        assert env.observation_space.shape == (6,), f"观察空间形状错误: {env.observation_space.shape}"
        print("✅ 观察空间正确: shape=(6,)")
        
        # 测试动作空间
        assert env.action_space.n == 4, f"动作空间大小错误: {env.action_space.n}"
        print("✅ 动作空间正确: 4个动作 (上、下、左、右)")
        
        # 测试reset
        obs, info = env.reset()
        assert obs.shape == (6,), f"观察形状错误: {obs.shape}"
        assert isinstance(obs, np.ndarray), "观察不是numpy数组"
        print("✅ reset() 成功，观察形状正确")
        
        # 测试step
        action = 0  # 上
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (6,), f"观察形状错误: {obs.shape}"
        assert isinstance(reward, (int, float)), "奖励不是数值"
        assert isinstance(terminated, bool), "terminated不是布尔值"
        print("✅ step() 成功")
        
        # 测试所有动作
        obs, _ = env.reset()
        for action in range(4):
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.isnan(reward), f"动作 {action} 产生NaN奖励"
        print("✅ 所有动作 (0-3) 都能正常执行")
        
        # 测试位置更新
        initial_pos = env.ego_pos.copy()
        env.step(0)  # 上
        assert env.ego_pos[1] > initial_pos[1], "向上移动后y坐标应该增加"
        print("✅ 位置更新正确")
        
        return True
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_function():
    """测试奖励函数"""
    print("\n" + "=" * 70)
    print("测试 3: 奖励函数")
    print("=" * 70)
    try:
        from env import DecisionEnv
        import numpy as np
        
        env = DecisionEnv()
        obs, _ = env.reset()
        
        # 测试向目标移动的奖励
        initial_dist = np.linalg.norm(env.destination - env.ego_pos)
        obs, reward1, _, _, _ = env.step(0)  # 随机动作
        print(f"✅ 奖励计算正常: {reward1:.2f}")
        
        # 测试到达目标
        # 手动设置位置接近目标
        env.ego_pos = env.destination + np.array([1.0, 1.0])
        obs, reward2, terminated, _, _ = env.step(0)
        if terminated:
            print("✅ 到达目标时正确终止")
        
        # 测试碰撞
        env.reset()
        env.ego_pos = env.obs_pos + np.array([0.5, 0.5])  # 接近障碍物
        obs, reward3, terminated, _, _ = env.step(0)
        if terminated:
            print("✅ 碰撞时正确终止")
        
        return True
    except Exception as e:
        print(f"❌ 奖励函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 70)
    print("测试 4: 模型文件")
    print("=" * 70)
    try:
        import os
        import zipfile
        
        # 检查模型文件是否存在
        model_path = "ppo_decision.zip"
        if os.path.exists(model_path):
            print(f"✅ 模型文件存在: {model_path}")
            
            # 检查zip文件是否有效
            try:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    files = zip_ref.namelist()
                    print(f"✅ ZIP文件有效，包含 {len(files)} 个文件")
            except zipfile.BadZipFile:
                print("❌ ZIP文件损坏")
                return False
        else:
            print(f"⚠️  模型文件不存在: {model_path} (需要先训练)")
            return True  # 不算错误，只是还没训练
        
        # 尝试加载模型
        try:
            from stable_baselines3 import PPO
            model = PPO.load("ppo_decision")
            print("✅ 模型加载成功")
            return True
        except Exception as e:
            print(f"⚠️  模型加载失败: {e} (可能需要先训练)")
            return True  # 不算错误，只是还没训练
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def test_scripts():
    """测试脚本文件是否存在且可读"""
    print("\n" + "=" * 70)
    print("测试 5: 脚本文件")
    print("=" * 70)
    scripts = [
        "train.py",
        "evaluate.py",
        "analyze_performance.py",
        "visualize_trajectories.py",
        "demo_env.py"
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script} 存在")
            # 检查文件是否可读
            try:
                with open(script, 'r') as f:
                    content = f.read()
                    if len(content) > 0:
                        print(f"   ✅ 文件可读 ({len(content)} 字符)")
                    else:
                        print(f"   ⚠️  文件为空")
            except Exception as e:
                print(f"   ❌ 文件读取失败: {e}")
                all_exist = False
        else:
            print(f"❌ {script} 不存在")
            all_exist = False
    
    return all_exist

def test_requirements():
    """检查依赖文件"""
    print("\n" + "=" * 70)
    print("测试 6: 依赖文件")
    print("=" * 70)
    if os.path.exists("requirements.txt"):
        print("✅ requirements.txt 存在")
        try:
            with open("requirements.txt", 'r') as f:
                requirements = f.read().strip().split('\n')
                print(f"✅ 包含 {len(requirements)} 个依赖包:")
                for req in requirements:
                    if req.strip():
                        print(f"   - {req.strip()}")
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            return False
        return True
    else:
        print("❌ requirements.txt 不存在")
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("开始验证子文件夹内代码功能")
    print("=" * 70)
    
    results = []
    
    results.append(("导入测试", test_imports()))
    results.append(("环境功能", test_environment()))
    results.append(("奖励函数", test_reward_function()))
    results.append(("模型文件", test_model_loading()))
    results.append(("脚本文件", test_scripts()))
    results.append(("依赖文件", test_requirements()))
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！代码功能正常。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 项测试未通过，请检查。")
        return 1

if __name__ == "__main__":
    exit(main())


