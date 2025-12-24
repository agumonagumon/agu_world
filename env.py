import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DecisionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        # 观察空间：ego位置(2) + destination位置(2) + obs位置(2) = 6维（移除速度）
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.step_size = 1.0  # 每次移动的固定距离
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # ego初始位置设为原点
        self.ego_pos = np.array([0.0, 0.0], dtype=np.float32)
        
        # destination目标点：在初始位置前方
        dest_distance = np.random.uniform(50, 80)
        dest_angle = np.random.uniform(-0.2, 0.2)  # 目标点稍微偏移
        self.destination = np.array([
            dest_distance * np.cos(dest_angle),
            dest_distance * np.sin(dest_angle)
        ], dtype=np.float32)
        
        # 障碍物位置：在初始位置和destination之间
        obs_ratio = np.random.uniform(0.3, 0.5)  # 在路径的30%-50%处
        obs_base_x = obs_ratio * self.destination[0]
        obs_base_y = obs_ratio * self.destination[1]
        
        # 障碍物横向偏移（在路径两侧）
        lateral_offset = np.random.uniform(-5, 5)
        # 计算垂直于路径的方向
        path_direction = self.destination / (np.linalg.norm(self.destination) + 1e-6)
        perpendicular = np.array([-path_direction[1], path_direction[0]])
        self.obs_pos = np.array([
            obs_base_x + lateral_offset * perpendicular[0],
            obs_base_y + lateral_offset * perpendicular[1]
        ], dtype=np.float32)
        
        return self._get_obs(), {}

    def _get_obs(self):
        """返回观察：ego位置、destination位置、obs位置（移除速度）"""
        return np.concatenate([
            self.ego_pos,
            self.destination,
            self.obs_pos
        ], dtype=np.float32)

    def step(self, action):
        reward = -0.01  # 每步小惩罚
        terminated = False
        
        # 动作执行：上下左右平移
        # action: 0=上(y+), 1=下(y-), 2=左(x-), 3=右(x+)
        if action == 0:      # 上 - 向y正方向移动
            self.ego_pos += np.array([0.0, self.step_size], dtype=np.float32)
        elif action == 1:    # 下 - 向y负方向移动
            self.ego_pos += np.array([0.0, -self.step_size], dtype=np.float32)
        elif action == 2:    # 左 - 向x负方向移动
            self.ego_pos += np.array([-self.step_size, 0.0], dtype=np.float32)
        elif action == 3:    # 右 - 向x正方向移动
            self.ego_pos += np.array([self.step_size, 0.0], dtype=np.float32)
        
        # 计算到destination的距离
        dist_to_dest = np.linalg.norm(self.destination - self.ego_pos)
        
        # 计算到障碍物的距离
        dist_to_obs = np.linalg.norm(self.obs_pos - self.ego_pos)
        
        # 奖励设计：基于到终点的距离
        # 初始距离
        if not hasattr(self, 'initial_dist_to_dest'):
            self.initial_dist_to_dest = np.linalg.norm(self.destination)
            self.last_dist_to_dest = self.initial_dist_to_dest
        
        # 1. 进度奖励：基于距离减少（鼓励向目标前进）
        progress = self.last_dist_to_dest - dist_to_dest
        progress_reward = progress * 2.0  # 每减少1单位距离，奖励2.0
        reward += progress_reward
        self.last_dist_to_dest = dist_to_dest
        
        # 2. 距离奖励：越接近目标，奖励越高（非线性，越近奖励增长越快）
        normalized_dist = dist_to_dest / (self.initial_dist_to_dest + 1e-6)
        # 如果超过初始距离，直接给负奖励
        if normalized_dist > 1.0:
            distance_reward = -(normalized_dist - 1.0) * 2.0  # 超过部分每单位惩罚2.0
        else:
            # 使用平方函数，使得接近目标时奖励增长更快
            distance_reward = (1.0 - normalized_dist) ** 2 * 1.0
        reward += distance_reward
        
        # 3. 动作奖励：根据距离和动作类型给予不同奖励
        # 计算到目标的方向，鼓励向目标方向移动
        to_dest = self.destination - self.ego_pos
        to_dest_norm = np.linalg.norm(to_dest)
        if to_dest_norm > 1e-6:
            to_dest_dir = to_dest / to_dest_norm
            # 计算动作方向
            action_dirs = {
                0: np.array([0.0, 1.0]),   # 上
                1: np.array([0.0, -1.0]),  # 下
                2: np.array([-1.0, 0.0]),  # 左
                3: np.array([1.0, 0.0])    # 右
            }
            action_dir = action_dirs[action]
            # 计算动作方向与目标方向的一致性（点积）
            direction_alignment = np.dot(action_dir, to_dest_dir)
            # 如果动作方向朝向目标，给予奖励
            if direction_alignment > 0:
                direction_reward = direction_alignment * 0.5  # 最多0.5的奖励
                reward += direction_reward
        
        # 5. 渐进式碰撞警告：距离障碍物越近，惩罚越大（增强版本）
        if dist_to_obs < 25.0:  # 提前开始警告（从25单位开始）
            # 距离越近，惩罚越大（增强）
            obs_penalty = (25.0 - dist_to_obs) / 10.0 * 1.5  # 最多3.75的惩罚
            reward -= obs_penalty
        
        # 5.1 避障动作奖励：当接近障碍物时，强烈鼓励远离障碍物（增强版本）
        if dist_to_obs < 15.0:  # 距离障碍物15单位内（扩大范围）
            to_obs = self.obs_pos - self.ego_pos
            to_obs_norm = np.linalg.norm(to_obs)
            if to_obs_norm > 1e-6:
                to_obs_dir = to_obs / to_obs_norm
                # 计算动作方向
                action_dirs = {
                    0: np.array([0.0, 1.0]),   # 上
                    1: np.array([0.0, -1.0]),  # 下
                    2: np.array([-1.0, 0.0]),  # 左
                    3: np.array([1.0, 0.0])    # 右
                }
                action_dir = action_dirs[action]
                # 计算动作方向与远离障碍物方向的一致性（负点积，因为要远离）
                avoidance_alignment = -np.dot(action_dir, to_obs_dir)
                # 如果动作方向远离障碍物，给予奖励（增强）
                if avoidance_alignment > 0:
                    avoidance_bonus = avoidance_alignment * (15.0 - dist_to_obs) / 15.0 * 2.0  # 最多2.0的奖励
                    reward += avoidance_bonus
                # 如果动作方向朝向障碍物，给予惩罚（增强）
                elif avoidance_alignment < 0:
                    approach_penalty = -avoidance_alignment * (15.0 - dist_to_obs) / 15.0 * 1.5  # 最多1.5的惩罚
                    reward -= approach_penalty
        
        # 5.2 紧急避障：非常接近障碍物时，强烈鼓励避障
        if dist_to_obs < 8.0:  # 距离障碍物8单位内（紧急情况）
            to_obs = self.obs_pos - self.ego_pos
            to_obs_norm = np.linalg.norm(to_obs)
            if to_obs_norm > 1e-6:
                to_obs_dir = to_obs / to_obs_norm
                action_dirs = {
                    0: np.array([0.0, 1.0]),   # 上
                    1: np.array([0.0, -1.0]),  # 下
                    2: np.array([-1.0, 0.0]),  # 左
                    3: np.array([1.0, 0.0])    # 右
                }
                action_dir = action_dirs[action]
                avoidance_alignment = -np.dot(action_dir, to_obs_dir)
                # 紧急情况下，远离障碍物的奖励大幅增强
                if avoidance_alignment > 0:
                    emergency_bonus = avoidance_alignment * (8.0 - dist_to_obs) / 8.0 * 3.0  # 最多3.0的奖励
                    reward += emergency_bonus
                # 紧急情况下，朝向障碍物的惩罚大幅增强
                elif avoidance_alignment < 0:
                    emergency_penalty = -avoidance_alignment * (8.0 - dist_to_obs) / 8.0 * 2.5  # 最多2.5的惩罚
                    reward -= emergency_penalty
        
        # 6. 到达目标（大幅奖励）
        if dist_to_dest < 8.0:  # 增大到达阈值
            reward += 100.0  # 到达奖励
            terminated = True
        
        # 7. 碰撞障碍物（大幅惩罚，增强以避免碰撞）
        if dist_to_obs < 2.0:
            reward -= 200.0  # 碰撞惩罚（增强，从160增加到200）
            terminated = True
        
        return self._get_obs(), reward, terminated, False, {}
