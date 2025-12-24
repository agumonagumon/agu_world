"""
分析模型表现，找出问题原因
"""
from stable_baselines3 import PPO
from env import DecisionEnv
import numpy as np
import matplotlib.pyplot as plt

def analyze_performance(num_episodes=20):
    """详细分析模型表现"""
    env = DecisionEnv()
    model = PPO.load("ppo_decision")
    
    # 统计数据
    episodes_data = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # up, down, left, right
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    print("=" * 80)
    print("详细性能分析")
    print("=" * 80)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        trajectory = [env.ego_pos.copy()]
        initial_dist = np.linalg.norm(env.destination - env.ego_pos)
        episode_actions = []
        episode_rewards = []
        episode_distances = [initial_dist]
        episode_speeds = []
        
        total_reward = 0
        steps = 0
        
        while steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, done, truncated, _ = env.step(action)
            
            trajectory.append(env.ego_pos.copy())
            episode_actions.append(action)
            episode_rewards.append(reward)
            action_counts[action] += 1
            
            dist_to_dest = np.linalg.norm(env.destination - env.ego_pos)
            dist_to_obs = np.linalg.norm(env.obs_pos - env.ego_pos)
            # 移除速度相关，因为新模型不考虑速度
            speed = 1.0  # 固定速度，因为每次移动固定距离
            
            episode_distances.append(dist_to_dest)
            episode_speeds.append(speed)
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # 判断结果
        final_dist = np.linalg.norm(env.destination - env.ego_pos)
        final_dist_obs = np.linalg.norm(env.obs_pos - env.ego_pos)
        
        if final_dist_obs < 2.0:
            result = "collision"
        elif final_dist < 8.0:  # 更新阈值以匹配环境
            result = "success"
        else:
            result = "timeout"
        
        episodes_data.append({
            'result': result,
            'total_reward': total_reward,
            'steps': steps,
            'initial_dist': initial_dist,
            'final_dist': final_dist,
            'min_dist': min(episode_distances),
            'actions': episode_actions,
            'rewards': episode_rewards,
            'distances': episode_distances,
            'speeds': episode_speeds,
            'trajectory': np.array(trajectory)
        })
    
    # 打印统计
    print(f"\n总体统计 ({num_episodes} episodes):")
    print("-" * 80)
    success = [e for e in episodes_data if e['result'] == 'success']
    collision = [e for e in episodes_data if e['result'] == 'collision']
    timeout = [e for e in episodes_data if e['result'] == 'timeout']
    
    print(f"成功: {len(success)} ({len(success)/num_episodes*100:.1f}%)")
    print(f"碰撞: {len(collision)} ({len(collision)/num_episodes*100:.1f}%)")
    print(f"超时: {len(timeout)} ({len(timeout)/num_episodes*100:.1f}%)")
    
    # 动作使用统计
    print(f"\n动作使用统计:")
    print("-" * 80)
    total_actions = sum(action_counts.values())
    for action_id, count in action_counts.items():
        percentage = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action_names[action_id]:12s}: {count:5d} ({percentage:5.1f}%)")
    
    # 距离分析
    print(f"\n距离分析:")
    print("-" * 80)
    all_min_dists = [e['min_dist'] for e in episodes_data]
    all_final_dists = [e['final_dist'] for e in episodes_data]
    print(f"最小距离 (所有episodes):")
    print(f"  平均: {np.mean(all_min_dists):.2f}")
    print(f"  最小: {np.min(all_min_dists):.2f}")
    print(f"  最大: {np.max(all_min_dists):.2f}")
    print(f"  中位数: {np.median(all_min_dists):.2f}")
    
    print(f"\n最终距离 (所有episodes):")
    print(f"  平均: {np.mean(all_final_dists):.2f}")
    print(f"  最小: {np.min(all_final_dists):.2f}")
    print(f"  最大: {np.max(all_final_dists):.2f}")
    print(f"  中位数: {np.median(all_final_dists):.2f}")
    
    # 速度分析
    print(f"\n速度分析:")
    print("-" * 80)
    all_speeds = []
    for e in episodes_data:
        all_speeds.extend(e['speeds'])
    print(f"平均速度: {np.mean(all_speeds):.2f}")
    print(f"速度范围: {np.min(all_speeds):.2f} - {np.max(all_speeds):.2f}")
    print(f"速度中位数: {np.median(all_speeds):.2f}")
    
    # 分析接近目标时的行为
    print(f"\n接近目标时的行为分析 (距离 < 10):")
    print("-" * 80)
    near_target_actions = {0: 0, 1: 0, 2: 0, 3: 0}
    near_target_count = 0
    for e in episodes_data:
        for i, dist in enumerate(e['distances']):
            if dist < 10.0 and i < len(e['actions']):
                action = e['actions'][i]
                near_target_actions[action] += 1
                near_target_count += 1
    
    if near_target_count > 0:
        print(f"在距离 < 10 时的动作分布:")
        for action_id, count in near_target_actions.items():
            percentage = count / near_target_count * 100 if near_target_count > 0 else 0
            print(f"  {action_names[action_id]:12s}: {count:5d} ({percentage:5.1f}%)")
    
    # 分析成功接近但未到达的episodes
    print(f"\n接近目标但未到达的episodes (最小距离 < 10 但未成功):")
    print("-" * 80)
    close_but_failed = [e for e in episodes_data if e['min_dist'] < 10.0 and e['result'] != 'success']
    print(f"数量: {len(close_but_failed)}")
    if len(close_but_failed) > 0:
        print(f"平均最小距离: {np.mean([e['min_dist'] for e in close_but_failed]):.2f}")
        print(f"平均最终距离: {np.mean([e['final_dist'] for e in close_but_failed]):.2f}")
        
        # 分析这些episodes在接近时的动作
        print(f"\n这些episodes在接近目标时的动作:")
        near_actions = {0: 0, 1: 0, 2: 0, 3: 0}
        near_count = 0
        for e in close_but_failed:
            for i, dist in enumerate(e['distances']):
                if dist < 10.0 and i < len(e['actions']):
                    action = e['actions'][i]
                    near_actions[action] += 1
                    near_count += 1
        
        if near_count > 0:
            for action_id, count in near_actions.items():
                percentage = count / near_count * 100 if near_count > 0 else 0
                print(f"  {action_names[action_id]:12s}: {count:5d} ({percentage:5.1f}%)")
    
    # 问题诊断
    print(f"\n" + "=" * 80)
    print("问题诊断")
    print("=" * 80)
    
    issues = []
    
    # 1. 检查速度（新模型固定速度，此检查已不适用）
    # if np.mean(all_speeds) < 3.0:
    #     issues.append("⚠️  平均速度过低，可能限制了移动能力")
    
    # 2. 检查动作使用（新模型：上下左右）
    # 检查是否有动作使用过度集中
    max_action_ratio = max(action_counts.values()) / total_actions if total_actions > 0 else 0
    if max_action_ratio > 0.8:
        issues.append("⚠️  某个动作使用率过高，模型可能动作选择不够多样化")
    
    # 3. 检查接近目标时的行为（新模型：检查是否合理使用动作）
    if near_target_count > 0:
        # 检查动作分布是否合理
        action_diversity = len([c for c in near_target_actions.values() if c > 0]) / len(near_target_actions)
        if action_diversity < 0.5:
            issues.append("⚠️  接近目标时动作选择过于单一，可能导致无法有效到达")
    
    # 4. 检查到达阈值
    if len(close_but_failed) > 0:
        avg_min_dist = np.mean([e['min_dist'] for e in close_but_failed])
        if avg_min_dist < 6.0:
            issues.append(f"⚠️  很多episodes接近目标(最小距离{avg_min_dist:.2f})但未到达，到达阈值(5.0)可能过小")
    
    # 5. 检查奖励信号
    if len(success) == 0:
        issues.append("❌  成功率0%，模型可能没有学习到正确的策略")
    
    if len(collision) > len(success):
        issues.append("⚠️  碰撞率高于成功率，避障策略可能有问题")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ 未发现明显问题")
    
    # 建议
    print(f"\n" + "=" * 80)
    print("改进建议")
    print("=" * 80)
    
    suggestions = []
    
    if len(close_but_failed) > 0 and near_target_count > 0:
        # 检查接近目标时的动作分布
        action_diversity = len([c for c in near_target_actions.values() if c > 0]) / len(near_target_actions)
        if action_diversity < 0.5:
            suggestions.append("1. 在接近目标时(距离<10)增加朝向目标方向动作的奖励")
            suggestions.append("2. 优化奖励函数，鼓励模型在接近目标时选择更直接的动作")
    
    if np.mean(all_speeds) < 4.0:
        suggestions.append("3. 考虑提高最大速度限制(当前5.0)，或增加速度奖励")
    
    if len(close_but_failed) > 0:
        avg_min_dist = np.mean([e['min_dist'] for e in close_but_failed])
        if avg_min_dist < 6.0:
            suggestions.append("4. 考虑增大到达目标的阈值(当前5.0)，或添加更平滑的到达奖励")
    
    # 动作使用建议（新模型：上下左右）
    max_action_ratio = max(action_counts.values()) / total_actions if total_actions > 0 else 0
    if max_action_ratio > 0.7:
        suggestions.append("5. 增加朝向目标方向动作的奖励，鼓励向目标移动")
    
    if len(success) == 0:
        suggestions.append("6. 增加训练时间或调整学习率")
        suggestions.append("7. 检查奖励函数是否平衡，确保成功奖励足够大")
    
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    return episodes_data, action_counts

if __name__ == "__main__":
    episodes_data, action_counts = analyze_performance(num_episodes=20)

