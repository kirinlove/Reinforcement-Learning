import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import gym
from gym import spaces
import os
from scipy.integrate import solve_ivp
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 超参数配置
BUFFER_SIZE = int(1e5)  # 經驗回放緩衝區大小
BATCH_SIZE = 64        # 小批量採樣大小
GAMMA = 1              # 折扣因子
TAU = 0.005            # 目標網路軟更新係數
LR_ACTOR = 1e-4        # Actor學習率
LR_CRITIC = 1e-3       # Critic學習率
EXPLORATION_NOISE = 1.5 # 動作探索噪聲
TARGET_NOISE = 0.4     # 目標動作噪聲
NOISE_CLIP = 0.5       # 噪聲裁減範圍
POLICY_UPDATE_FREQ = 10 # 策略延遲更新頻率

# 設備配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor網路定義（策略網路）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.output_layer(x)

# Critic網路定義（雙Q網路）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1網路
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 1)
        
        # Q2網路
        self.layer5 = nn.Linear(state_dim + action_dim, 256)
        self.layer6 = nn.Linear(256, 256)
        self.layer7 = nn.Linear(256, 256)
        self.layer8 = nn.Linear(256, 1)
        
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Q1前向傳播
        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        q1 = torch.relu(self.layer3(q1))
        q1 = self.layer4(q1)
        
        # Q2前向傳播
        q2 = torch.relu(self.layer5(sa))
        q2 = torch.relu(self.layer6(q2))
        q2 = torch.relu(self.layer7(q2))
        q2 = self.layer8(q2)
        return q1, q2

# 經驗回放緩衝區
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, copy=True),
            np.array(action, copy=True),
            reward,
            np.array(next_state, copy=True),
            done
        ))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

# TD3 Agent類
class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR, betas=(0.8,0.7), weight_decay=0.0001, amsgrad=True)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, betas=(0.8,0.7), weight_decay=0.0001, amsgrad=True)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.total_iterations = 0
        
    def select_action(self, state, add_noise=True):
        # 對狀態進行 mod 操作
        modified_state = np.array([
            state[0] % (2 * np.pi),  # x mod 2*pi
            state[1] % (2 * np.pi),  # y mod 2*pi
            state[2] % ((2 * np.pi) / env.omega)  # time mod (2*pi)/w
        ])
        state = torch.FloatTensor(modified_state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if add_noise:
            noise = np.random.normal(0, EXPLORATION_NOISE, size=action.shape)
            action = action + noise
        return action
    
    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # 從緩衝區採樣小批量數據
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        # 對狀態進行 mod 操作
        states_mod = torch.stack([
            states[:, 0] % (2 * np.pi),  # x mod 2*pi
            states[:, 1] % (2 * np.pi),  # y mod 2*pi
            states[:, 2] % ((2 * np.pi) / env.omega)  # time mod (2*pi)/w
        ], dim=1)
        
        next_states_mod = torch.stack([
            next_states[:, 0] % (2 * np.pi),  # x mod 2*pi
            next_states[:, 1] % (2 * np.pi),  # y mod 2*pi
            next_states[:, 2] % ((2 * np.pi) / env.omega)  # time mod (2*pi)/w
        ], dim=1)
    
        # 計算目標Q值（帶噪聲裁剪）
        with torch.no_grad():
            next_actions = self.actor_target(next_states_mod)
            noise = torch.randn_like(next_actions) * TARGET_NOISE
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = next_actions + noise
                
            target_Q1, target_Q2 = self.critic_target(next_states_mod, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * GAMMA * target_Q
            
        # 更新Critic網路
        current_Q1, current_Q2 = self.critic(states_mod, actions)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
            
        # 延遲策略更新
        if self.total_iterations % POLICY_UPDATE_FREQ == 0:
            # 更新Actor網路
            actor_loss = -self.critic(states_mod, self.actor(states_mod))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
                
            # 軟更新目標網路
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        self.total_iterations += 1


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.dt = 0.125
        self.T = 5
        self.steps = int(self.T / self.dt)
        self.current_step = 0
        self.x = 0
        self.y = 0
        self.u = 0.0  
        self.v = 0.0  
        self.prev_x = 0.0
        self.A = 4  # 振幅
        self.omega = 3.0
        self.time = 0  # 當前時間

        # 初始化歷史軌跡
        self.x_history = []
        self.y_history = []

        # Action space: u, v
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        # Observation space: x, y, time
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.x = 0
        self.y = 0
        self.u = 0.0  
        self.v = 0.0  
        self.prev_x = 0.0
        self.time = 0  # 重置時間

        # 重置歷史軌跡
        self.x_history = [self.x]
        self.y_history = [self.y]

        return np.array([self.x, self.y, self.time])

    def step(self, action):
        self.u, self.v = action  # 更新 u, v
        direction = np.array([self.u, self.v])
        norm = np.sqrt(self.u**2 + self.v**2) + (1e-8)  # 防止除以零
        direction = direction / norm 
        # 解 dx/dt = V(t, x) + direction using solve_ivp
        def ode(t, s):
            x, y = s
            Vx = self.A * (np.cos(y) + np.sin(y) * np.cos(self.omega * t))
            Vy = self.A * (np.cos(x) + np.sin(x) * np.cos(self.omega * t))
            return [Vx + direction[0], Vy + direction[1]]

        s0 = [self.x, self.y]
        sol = solve_ivp(ode, [0, self.dt], s0, method='RK45', t_eval=[self.dt])

        self.x, self.y = sol.y[:, -1]  # 取最後一點（也剛好是唯一的一點）
        self.time += self.dt  # 更新時間

        # 更新历史轨迹
        self.x_history.append(self.x)
        self.y_history.append(self.y)

        reward = 2 * (self.x - self.prev_x)
        self.prev_x = self.x
        self.current_step += 1
        done = self.current_step >= self.steps
        if done:
            reward = reward + self.x  # 最後再加上 x 
        
        return np.array([self.x, self.y, self.time]), reward, done, {}

    
# 主程序
if __name__ == "__main__":
    # 打印使用的設備
    print(f"Using device: {device}")
    
    # 初始化環境與Agent
    env = CustomEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    best_agent = TD3Agent(state_dim, action_dim)  # 新增：最佳Agent
    best_reward = -np.inf  # 新增：最佳獎勵
    best_model_path = "best_model.pth"
    
    # 訓練
    max_episodes = 3000
    max_steps = 40
    test_steps = 400
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
        
        # 更新最佳Agent
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_agent.actor.load_state_dict(agent.actor.state_dict())  # 更新最佳Agent的Actor網絡
            best_agent.critic.load_state_dict(agent.critic.state_dict())  # 更新最佳Agent的Critic網絡
            torch.save(agent.actor.state_dict(), best_model_path)  # 保存最佳模型
    
    # 繪製原始Agent的軌跡
    env.steps = test_steps
    env.dt = env.T / test_steps
    state = env.reset()
    original_x_history = [state[0]]
    original_y_history = [state[1]]
    action_history = []  # 儲存動作歷史
    time_history = [0]  # 儲存時間歷史
    
    for step in range(test_steps):
        action = agent.select_action(state, add_noise=False)  # 不添加噪聲
        next_state, _, done, _ = env.step(action)
        original_x_history.append(next_state[0])
        original_y_history.append(next_state[1])
        action_history.append(action)  # 記錄動作
        time_history.append(time_history[-1] + env.dt)  # 累加時間
        state = next_state
        if done:
            break

    u_history = [a[0] for a in action_history]
    v_history = [a[1] for a in action_history]
    t_history = time_history[:-1]  # 時間點對應動作
    
    # 繪製最佳Agent的軌跡
    env.steps = test_steps
    env.dt = env.T / test_steps
    state = env.reset()
    best_x_history = [state[0]]
    best_y_history = [state[1]]
    best_action_history = []  # 儲存動作歷史
    best_time_history = [0]  # 儲存時間歷史
    
    for step in range(test_steps):
        action = best_agent.select_action(state, add_noise=False)  # 不添加噪聲
        next_state, _, done, _ = env.step(action)
        best_x_history.append(next_state[0])
        best_y_history.append(next_state[1])
        best_action_history.append(action)  # 記錄動作
        state = next_state
        if done:
            break

    best_u_history = [a[0] for a in best_action_history]
    best_v_history = [a[1] for a in best_action_history]
    
    # 繪製兩個Agent的軌跡
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(original_x_history, original_y_history, label="Original Agent Trajectory", color="blue")
    plt.scatter(original_x_history[0], original_y_history[0], color="green", label="Start")
    plt.scatter(original_x_history[-1], original_y_history[-1], color="red", label="End")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Original Agent Trajectory")
    plt.legend()
    plt.grid()
    plt.axis('equal')  # 尺度相同
    
    plt.subplot(1, 2, 2)
    plt.plot(best_x_history, best_y_history, label="Best Agent Trajectory", color="orange")
    plt.scatter(best_x_history[0], best_y_history[0], color="green", label="Start")
    plt.scatter(best_x_history[-1], best_y_history[-1], color="red", label="End")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Best Agent Trajectory")
    plt.legend()
    plt.grid()
    plt.axis('equal')  # 尺度相同

    plt.tight_layout()
    plt.show()
