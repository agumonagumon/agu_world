"""
可视化20个episode的路线行为
"""
from stable_baselines3 import PPO
from env import DecisionEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_episodes(num_episodes=20):
    """可视化多个episode的轨迹"""
    env = DecisionEnv()
    model = PPO.load("ppo_decision")
    
    # 存储所有episode的数据
    episodes_data = []
    
    print(f"Running {num_episodes} episodes and recording trajectories...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        trajectory = [env.ego_pos.copy()]  # 记录轨迹
        destination = env.destination.copy()
        obs_pos = env.obs_pos.copy()
        initial_pos = env.ego_pos.copy()
        
        total_reward = 0
        steps = 0
        
        while steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, done, truncated, _ = env.step(action)
            
            trajectory.append(env.ego_pos.copy())
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # 判断episode结果
        final_pos = env.ego_pos.copy()
        dist_to_dest = np.linalg.norm(destination - final_pos)
        dist_to_obs = np.linalg.norm(obs_pos - final_pos)
        
        if dist_to_obs < 2.0:
            result = "collision"
        elif dist_to_dest < 8.0:  # 更新到达阈值，与env.py和evaluate.py一致
            result = "success"
        else:
            result = "timeout"
        
        episodes_data.append({
            'trajectory': np.array(trajectory),
            'destination': destination,
            'obs_pos': obs_pos,
            'initial_pos': initial_pos,
            'result': result,
            'total_reward': total_reward,
            'steps': steps
        })
        
        print(f"Episode {episode + 1:2d}: {result:8s} | Reward: {total_reward:6.2f} | Steps: {steps:3d}")
    
    # 绘制轨迹图
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 统计
    success_count = sum(1 for ep in episodes_data if ep['result'] == 'success')
    collision_count = sum(1 for ep in episodes_data if ep['result'] == 'collision')
    timeout_count = sum(1 for ep in episodes_data if ep['result'] == 'timeout')
    
    # 绘制每个episode的轨迹
    for i, ep_data in enumerate(episodes_data):
        trajectory = ep_data['trajectory']
        result = ep_data['result']
        
        # 根据结果选择颜色和样式
        if result == 'success':
            color = 'green'
            alpha = 0.6
            linewidth = 1.5
            label = 'Success' if i == 0 or sum(1 for j in range(i) if episodes_data[j]['result'] == 'success') == 0 else ''
        elif result == 'collision':
            color = 'red'
            alpha = 0.6
            linewidth = 1.5
            label = 'Collision' if i == 0 or sum(1 for j in range(i) if episodes_data[j]['result'] == 'collision') == 0 else ''
        else:  # timeout
            color = 'orange'
            alpha = 0.4
            linewidth = 1.0
            label = 'Timeout' if i == 0 or sum(1 for j in range(i) if episodes_data[j]['result'] == 'timeout') == 0 else ''
        
        # 绘制轨迹线
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
                color=color, alpha=alpha, linewidth=linewidth, label=label)
        
        # 标记起点
        ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                  color='blue', marker='o', s=50, alpha=0.7, zorder=5)
        
        # 标记终点
        end_marker = '^' if result == 'success' else ('X' if result == 'collision' else 's')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                  color=color, marker=end_marker, s=100, alpha=0.8, zorder=5)
    
    # 绘制目标和障碍物（使用第一个episode的位置作为参考，因为它们可能不同）
    # 实际上每个episode的目标和障碍物位置可能不同，我们绘制所有
    for i, ep_data in enumerate(episodes_data):
        # 目标点
        ax.scatter(ep_data['destination'][0], ep_data['destination'][1], 
                  color='gold', marker='*', s=200, alpha=0.8, zorder=6,
                  label='Destination' if i == 0 else '')
        
        # 障碍物
        circle = patches.Circle(ep_data['obs_pos'], radius=2.0, 
                               color='darkred', alpha=0.3, zorder=4)
        ax.add_patch(circle)
        ax.scatter(ep_data['obs_pos'][0], ep_data['obs_pos'][1], 
                  color='darkred', marker='s', s=100, alpha=0.8, zorder=6,
                  label='Obstacle' if i == 0 else '')
    
    # 设置图形属性
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'20 Episodes Trajectories\n'
                f'Success: {success_count} | Collision: {collision_count} | Timeout: {timeout_count}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    # 添加说明
    textstr = f'Total Episodes: {num_episodes}\n'
    textstr += f'Success Rate: {success_count/num_episodes*100:.1f}%\n'
    textstr += f'Collision Rate: {collision_count/num_episodes*100:.1f}%\n'
    textstr += f'Timeout Rate: {timeout_count/num_episodes*100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('trajectories_20_episodes.png', dpi=150, bbox_inches='tight')
    print(f"\nTrajectory plot saved as: trajectories_20_episodes.png")
    plt.show()

if __name__ == "__main__":
    visualize_episodes(num_episodes=20)

