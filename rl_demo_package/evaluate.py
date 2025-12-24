"""
评估训练好的模型性能
"""
from stable_baselines3 import PPO
from env import DecisionEnv
import numpy as np

def evaluate_model(num_episodes=20, verbose=True):
    """评估模型性能"""
    env = DecisionEnv()
    model = PPO.load("ppo_decision")
    
    # 统计指标
    episode_rewards = []
    episode_lengths = []
    success_count = 0  # 成功到达目标
    collision_count = 0  # 碰撞障碍物
    timeout_count = 0  # 超时（达到最大步数）
    
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    print("=" * 70)
    print("模型评估结果")
    print("=" * 70)
    print(f"评估 {num_episodes} 个 episodes...\n")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_actions = []
        
        while steps < 200:  # 增加最大步数
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # 确保是标量
            obs, reward, done, truncated, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            episode_actions.append(action_names[action])
            
            if done or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 计算最终距离
        dist_to_dest = np.linalg.norm(env.destination - env.ego_pos)
        dist_to_obs = np.linalg.norm(env.obs_pos - env.ego_pos)
        
        # 判断episode结果
        if dist_to_obs < 2.0:
            collision_count += 1
            result = "❌ 碰撞"
        elif dist_to_dest < 8.0:  # 更新到达阈值
            success_count += 1
            result = "✅ 成功"
        else:
            timeout_count += 1
            result = "⏱️  超时"
        
        if verbose:
            print(f"Episode {episode + 1:2d}: {result:8s} | "
                  f"奖励: {total_reward:6.2f} | "
                  f"步数: {steps:3d} | "
                  f"到目标: {dist_to_dest:5.2f}")
            if verbose and episode < 3:  # 只显示前3个episode的详细动作序列
                print(f"  动作序列: {' -> '.join(episode_actions[:15])}")
                if len(episode_actions) > 15:
                    print(f"            ... (共{len(episode_actions)}步)")
    
    # 打印统计结果
    print("\n" + "=" * 70)
    print("总体统计")
    print("=" * 70)
    print(f"总Episodes: {num_episodes}")
    print(f"  ✅ 成功到达: {success_count} ({success_count/num_episodes*100:.1f}%)")
    print(f"  ❌ 碰撞:     {collision_count} ({collision_count/num_episodes*100:.1f}%)")
    print(f"  ⏱️  超时:     {timeout_count} ({timeout_count/num_episodes*100:.1f}%)")
    print()
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  最小: {np.min(episode_rewards):.2f}")
    print(f"  最大: {np.max(episode_rewards):.2f}")
    print()
    print(f"平均Episode长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f} 步")
    print(f"  最短: {np.min(episode_lengths)} 步")
    print(f"  最长: {np.max(episode_lengths)} 步")
    print("=" * 70)
    
    # 性能评估
    print("\n性能评估:")
    success_rate = success_count / num_episodes
    if success_rate >= 0.8:
        print("  🎉 优秀！模型表现很好")
    elif success_rate >= 0.5:
        print("  👍 良好！模型表现不错")
    elif success_rate >= 0.3:
        print("  ⚠️  一般，可能需要更多训练")
    else:
        print("  ❌ 较差，建议增加训练时间或调整超参数")
    
    return {
        'success_rate': success_rate,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'success_count': success_count,
        'collision_count': collision_count,
        'timeout_count': timeout_count
    }

if __name__ == "__main__":
    # 运行评估
    results = evaluate_model(num_episodes=20, verbose=True)
    
    # 可选：运行一次详细演示
    print("\n" + "=" * 70)
    print("详细演示 - 单个Episode")
    print("=" * 70)
    env = DecisionEnv()
    model = PPO.load("ppo_decision")
    
    obs, _ = env.reset(seed=42)
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    dist_to_dest = np.linalg.norm(env.destination - env.ego_pos)
    print(f"\n初始状态: 位置={env.ego_pos}, 到目标={dist_to_dest:.2f}\n")
    print(f"{'步骤':<6} {'动作':<8} {'位置':<20} {'到目标':<10} {'奖励':<8} {'累计奖励':<10}")
    print("-" * 80)
    
    total_reward = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)  # 确保是标量
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        dist_to_dest = np.linalg.norm(env.destination - env.ego_pos)
        pos_str = f"({env.ego_pos[0]:.1f},{env.ego_pos[1]:.1f})"
        
        print(f"{step+1:<6} {action_names[action]:<8} {pos_str:<20} {dist_to_dest:>8.2f}  "
              f"{reward:>6.2f}  {total_reward:>8.2f}")
        
        if done or truncated:
            print(f"\nEpisode结束！最终奖励: {total_reward:.2f}")
            dist_to_obs = np.linalg.norm(env.obs_pos - env.ego_pos)
            if dist_to_obs < 2.0:
                print("结果: ❌ 碰撞障碍物")
            elif dist_to_dest < 8.0:  # 更新到达阈值
                print("结果: ✅ 成功到达目标")
            else:
                print("结果: ⏱️  超时")
            break
