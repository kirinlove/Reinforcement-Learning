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
BATCH_SIZE = 256       # 小批量採樣大小
GAMMA = 1              # 折扣因子
TAU = 0.005            # 目標網路軟更新係數
LR_ACTOR = 3e-4       # Actor學習率
LR_CRITIC = 3e-4       # Critic學習率
EXPLORATION_NOISE = 2.9 # 動作探索噪聲
TARGET_NOISE = 0.4     # 目標動作噪聲
NOISE_CLIP = 0.5       # 噪聲裁減範圍
POLICY_UPDATE_FREQ = 2 # 策略延遲更新頻率

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
        
        self.activate = nn.ReLU(0.01)
    def forward(self, state):
        x = self.activate(self.layer1(state))
        x = self.activate(self.layer2(x))
        x = self.activate(self.layer3(x))
        return self.output_layer(x)

# Critic網路定義（雙Q網路）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1網路
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, 256)
        self.layer7 = nn.Linear(256, 1)
        
        # Q2網路
        self.layer8 = nn.Linear(state_dim + action_dim, 256)
        self.layer9 = nn.Linear(256, 256)
        self.layer10 = nn.Linear(256, 256)
        self.layer11 = nn.Linear(256, 256)
        self.layer12 = nn.Linear(256, 256)
        self.layer13 = nn.Linear(256, 256)
        self.layer14 = nn.Linear(256, 1)
        
        self.activate = nn.ReLU(0.01)
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Q1前向傳播
        q1 = self.activate(self.layer1(sa))
        q1 = self.activate(self.layer2(q1))
        q1 = self.activate(self.layer3(q1))
        q1 = self.activate(self.layer4(q1))
        q1 = self.activate(self.layer5(q1))
        q1 = self.activate(self.layer6(q1))
        q1 = self.layer7(q1)
        
        # Q2前向傳播
        q2 = self.activate(self.layer8(sa))
        q2 = self.activate(self.layer9(q2))
        q2 = self.activate(self.layer10(q2))
        q2 = self.activate(self.layer11(q2))
        q2 = self.activate(self.layer12(q2))
        q2 = self.activate(self.layer13(q2))
        q2 = self.layer14(q2)
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

class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR,betas=(0.8,0.7), weight_decay=0.0001, amsgrad=False)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC,betas=(0.8,0.7), weight_decay=0.0001, amsgrad=False)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.total_iterations = 0
        
    def select_action(self, state, add_noise=True):
        # 對狀態進行 mod 1 操作
        state = np.mod(state, 1.0)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if add_noise:
            # 更合理的噪聲策略
            noise_std = EXPLORATION_NOISE * max(0.1, 1.0 - self.total_iterations / 50000)
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
        return action
    
    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        # 從緩衝區採樣小批量數據
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        # 對狀態進行 mod 1 操作
        states = torch.fmod(states, 1.0)
        next_states = torch.fmod(next_states, 1.0)
    
        # 計算目標Q值（帶噪聲裁剪）
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = torch.randn_like(next_actions) * TARGET_NOISE
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = next_actions + noise
                
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1-dones) * GAMMA * target_Q
            
        # 更新Critic網路
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
            
        # 延遲策略更新
        if self.total_iterations % POLICY_UPDATE_FREQ == 0:
            # 更新Actor網路
            actor_loss = -self.critic(states, self.actor(states))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
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
        self.dt = 0.025
        self.T = 1
        self.steps = int(self.T / self.dt)
        self.current_step = 0
        self.x = 0.25
        self.y = 0.25
        self.u = 0.0  # 新增：u
        self.v = 0.0  # 新增：v
        self.prev_x = 0.0
        self.time = 0  # 當前時間

        # 初始化歷史軌跡
        self.x_history = []
        self.y_history = []

        # Action space: (u, v)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        # Observation space: x, y
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.x = 0.25
        self.y = 0.25
        self.initial_x = 0.25
        self.u = 0.0  # 重置 u
        self.v = 0.0  # 重置 v
        self.prev_x = 0.0
        self.time = 0  # 重置時間
        

        # 重置歷史軌跡
        self.x_history = [self.x]
        self.y_history = [self.y]

        return np.array([self.x, self.y])

    def step(self, action):
        self.u, self.v = action
        direction = np.array([self.u, self.v])
        norm = np.sqrt(self.u**2 + self.v**2) + (1e-8)  # 防止除以零
        direction = direction / norm 

        # 解 dx/dt = V(t, x) + direction using solve_ivp
        def ode(t, s):
            x, y = s
            Vx = -16 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
            Vy = 16 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
            return [Vx + direction[0], Vy + direction[1]]

        s0 = [self.x, self.y]
        sol = solve_ivp(ode, [0, self.dt], s0, method='RK45', t_eval=[self.dt])

        self.x, self.y = sol.y[:, -1]  # 取最後一點（也剛好是唯一的一點）

        self.time += self.dt  # 更新時間
        
        # 更新歷史軌跡
        self.x_history.append(self.x)
        self.y_history.append(self.y)

        progress_reward = 10 * (self.x - self.prev_x)
            
        # 控制成本
        control_cost = -0.05 * np.linalg.norm(action)

        reward = progress_reward + control_cost

        self.prev_x = self.x
        self.current_step += 1
        done = self.current_step >= self.steps
        if done:
            reward += 0 * self.x  # 最後再加上 x 
            total_progress = self.x - self.initial_x
            final_reward = total_progress * 20  # 獎勵總體向右移動
            reward += final_reward
        
        return np.array([self.x, self.y]), reward, done, {}

def fill_buffer_with_flow_grid(agent, env, grid_size=40, num_episodes=1):
    xs = np.linspace(0, 0.5, grid_size, endpoint=False)
    ys = np.linspace(0, 0.5, grid_size, endpoint=False)
    for episode in range(num_episodes):
        for x in xs:
            for y in ys:
                Vx = -16 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
                Vy =  16 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
                norm = np.sqrt(Vx**2 + Vy**2) + 1e-8
                action = np.array([Vx / norm, Vy / norm])
                env.x, env.y = x, y
                env.prev_x = x
                env.current_step = 0
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.add(np.array([x, y]), action, reward, next_state, done)

def plot_training_progress(agent, env, episode, test_steps=400):
    """測試當前agent並繪製軌跡"""
    # 暫存原始環境設定
    original_steps = env.steps
    original_dt = env.dt
    
    # 設定測試環境
    env.steps = test_steps
    env.dt = env.T / test_steps
    
    # 測試agent
    state = env.reset()
    x_history = [state[0]]
    y_history = [state[1]]
    
    for step in range(test_steps):
        action = agent.select_action(state, add_noise=False)  # 不添加噪聲
        next_state, _, done, _ = env.step(action)
        x_history.append(next_state[0])
        y_history.append(next_state[1])
        state = next_state
        if done:
            break
    
    # 繪圖
    plt.figure(figsize=(8, 6))
    plt.plot(x_history, y_history, label=f"Episode {episode} Trajectory", color="blue", linewidth=2)
    plt.scatter(x_history[0], y_history[0], color="green", s=30, label="Start", zorder=5)
    plt.scatter(x_history[-1], y_history[-1], color="red", s=30, label="End", zorder=5)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Agent Trajectory at Episode {episode}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 恢復原始環境設定
    env.steps = original_steps
    env.dt = original_dt
    
    print(f"Episode {episode}: Final X position = {x_history[-1]:.4f}")
    
# 主程序
if __name__ == "__main__":
    # 初始化環境和Agent
    env = CustomEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    
    fill_buffer_with_flow_grid(agent, env, grid_size=40, num_episodes=1)
    print(f"Pre-filled buffer size: {len(agent.replay_buffer)}")
    
    # 訓練
    max_episodes = 7000
    max_steps = 40
    plot_interval = 250  # 每250個episode畫一次圖
    
    # 用來記錄reward歷史
    reward_history = []
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        reward_history.append(episode_reward)
        
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
        
        # 每隔250個episode畫一次圖
        if (episode + 1) % plot_interval == 0:
            print(f"\n=== Training Progress at Episode {episode+1} ===")
            plot_training_progress(agent, env, episode+1, test_steps=400)
            
            # 可以額外畫reward曲線
            if len(reward_history) >= plot_interval:
                plt.figure(figsize=(10, 4))
                plt.plot(reward_history, alpha=0.7, linewidth=1)
                # 計算移動平均
                window_size = min(50, len(reward_history))
                moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(reward_history)), moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title(f"Training Reward History (up to Episode {episode+1})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

    # 最終測試和保存
    print("\n=== Final Training Result ===")
    plot_training_progress(agent, env, max_episodes, test_steps=400)
