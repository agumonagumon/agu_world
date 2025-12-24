from stable_baselines3 import PPO
from env import DecisionEnv

env = DecisionEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=512,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.01,  # 增加探索率，鼓励尝试避障动作
    gamma=0.99,  # 折扣因子，重视长期奖励
    gae_lambda=0.95,  # GAE lambda，平衡偏差和方差
    clip_range=0.2,  # PPO clip range
    vf_coef=0.5,  # 价值函数系数
    max_grad_norm=0.5  # 梯度裁剪
)

# 增加训练时间，让模型更好地学习避障策略
model.learn(total_timesteps=300_000)  # 从200k增加到300k
model.save("ppo_decision")
