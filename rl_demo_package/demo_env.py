"""
演示环境如何生成数据的脚本
展示环境的状态、动作、奖励等数据生成过程
"""
import numpy as np
from env import DecisionEnv

def demo_environment():
    """演示环境的数据生成过程"""
    print("=" * 60)
    print("强化学习环境数据生成演示")
    print("=" * 60)
    
    # 创建环境
    env = DecisionEnv()
    
    # 演示1: 环境重置 - 生成初始状态
    print("\n【演示1】环境重置 - 生成初始随机状态")
    print("-" * 60)
    obs, info = env.reset(seed=42)  # 使用固定种子以便复现
    print(f"初始观察 (observation): {obs}")
    print(f"观察空间形状: {obs.shape}")
    print(f"观察空间范围: [-100, 100]")
    print(f"当前自车速度: {env.ego_speed:.2f}")
    print(f"当前障碍物距离: {env.obs_dist:.2f}")
    
    # 演示2: 执行多个动作 - 展示状态转换
    print("\n【演示2】执行动作 - 状态动态变化")
    print("-" * 60)
    print(f"{'步骤':<6} {'动作':<8} {'速度':<8} {'距离':<10} {'奖励':<8} {'终止':<6}")
    print("-" * 60)
    
    actions = [0, 0, 1, 2, 0, 1, 3]  # 预定义的动作序列
    action_names = {0: "go", 1: "yield", 2: "slow", 3: "stop"}
    
    for step, action in enumerate(actions, 1):
        action_name = action_names[action]
        obs_before = obs.copy()
        speed_before = env.ego_speed
        dist_before = env.obs_dist
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"{step:<6} {action_name:<8} {env.ego_speed:>6.2f}  {env.obs_dist:>8.2f}  {reward:>6.2f}  {terminated!s:<6}")
        
        if terminated:
            print(f"\n⚠️  环境终止！原因: ", end="")
            if env.obs_dist < 2:
                print("碰撞（距离 < 2）")
            elif env.obs_dist > 30:
                print("成功通过（距离 > 30）")
            break
    
    # 演示3: 多次重置 - 展示随机性
    print("\n【演示3】多次重置 - 展示初始状态的随机性")
    print("-" * 60)
    print(f"{'重置次数':<10} {'初始速度':<12} {'初始距离':<12}")
    print("-" * 60)
    
    for i in range(5):
        obs, _ = env.reset()
        print(f"{i+1:<10} {env.ego_speed:>10.2f}  {env.obs_dist:>10.2f}")
    
    # 演示4: 完整episode - 展示完整的数据流
    print("\n【演示4】完整Episode - 展示完整的数据生成流程")
    print("-" * 60)
    
    obs, _ = env.reset(seed=123)
    total_reward = 0
    step_count = 0
    
    print(f"初始状态: 速度={env.ego_speed:.2f}, 距离={env.obs_dist:.2f}")
    print("\n执行随机动作序列:")
    
    while step_count < 50:  # 最多50步
        # 随机选择动作（实际训练中由策略网络决定）
        action = env.action_space.sample()
        action_name = action_names[action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"  步骤 {step_count}: 动作={action_name}, 速度={env.ego_speed:.2f}, "
              f"距离={env.obs_dist:.2f}, 奖励={reward:.2f}")
        
        if terminated or truncated:
            print(f"\n✅ Episode结束！")
            print(f"   总步数: {step_count}")
            print(f"   总奖励: {total_reward:.2f}")
            print(f"   终止原因: ", end="")
            if env.obs_dist < 2:
                print("碰撞")
            elif env.obs_dist > 30:
                print("成功通过")
            else:
                print("其他")
            break
    
    # 演示5: 数据统计
    print("\n【演示5】数据统计 - 展示环境生成的数据特征")
    print("-" * 60)
    
    speeds = []
    distances = []
    rewards = []
    episode_lengths = []
    
    for episode in range(10):
        obs, _ = env.reset()
        speeds.append(env.ego_speed)
        distances.append(env.obs_dist)
        episode_reward = 0
        episode_steps = 0
        
        while episode_steps < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        episode_lengths.append(episode_steps)
    
    print(f"初始速度统计 (10个episode):")
    print(f"  平均值: {np.mean(speeds):.2f}")
    print(f"  范围: [{np.min(speeds):.2f}, {np.max(speeds):.2f}]")
    
    print(f"\n初始距离统计 (10个episode):")
    print(f"  平均值: {np.mean(distances):.2f}")
    print(f"  范围: [{np.min(distances):.2f}, {np.max(distances):.2f}]")
    
    print(f"\n奖励统计 (所有步骤):")
    print(f"  平均值: {np.mean(rewards):.2f}")
    print(f"  最小值: {np.min(rewards):.2f}")
    print(f"  最大值: {np.max(rewards):.2f}")
    
    print(f"\nEpisode长度统计:")
    print(f"  平均值: {np.mean(episode_lengths):.2f} 步")
    print(f"  范围: [{np.min(episode_lengths)}, {np.max(episode_lengths)}] 步")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n💡 关键点:")
    print("1. 每次 reset() 都会生成新的随机初始状态")
    print("2. 每次 step() 都会根据动作生成新的状态和奖励")
    print("3. 环境本身就是数据生成器，无需额外的虚拟数据")
    print("4. 训练时，PPO算法会与环境交互，自动收集这些数据")

if __name__ == "__main__":
    demo_environment()

