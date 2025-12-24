"""快速运行测试"""
from env import DecisionEnv
import numpy as np

# 测试环境运行
print("测试环境运行...")
env = DecisionEnv()
obs, _ = env.reset()
print(f"✅ 环境初始化成功，观察形状: {obs.shape}")

total_reward = 0
for i in range(10):
    action = np.random.randint(0, 4)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        print(f"Episode在第{i+1}步结束")
        break

print(f"✅ 运行10步成功，累计奖励: {total_reward:.2f}")
print("✅ 环境功能验证完成")

