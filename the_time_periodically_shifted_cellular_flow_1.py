import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym
from gym import spaces
import os
from scipy.integrate import solve_ivp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ============== 超參數配置 ==============
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 1.0 
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
EXPLORATION_NOISE = 1.5
TARGET_NOISE = 0.4
NOISE_CLIP = 0.5
POLICY_UPDATE_FREQ = 10

# 設備配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============== 網路定義 ==============
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, action_dim)

        self.norm = nn.LayerNorm(256)
        self.activate = nn.ReLU()
        
        # Xavier 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = self.activate(self.norm((self.layer1(state))))
        x = self.activate(self.norm((self.layer2(x))))
        x = self.activate(self.norm((self.layer3(x))))
        a = self.output_layer(x)
        a = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-8)
        return a

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

        self.norm = nn.LayerNorm(256)
        self.activate = nn.ReLU()

        # Xavier 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        # Q1前向傳播
        q1 = self.activate(self.norm((self.layer1(sa))))
        q1 = self.activate(self.norm((self.layer2(q1))))
        q1 = self.activate(self.norm((self.layer3(q1))))
        q1 = self.activate(self.norm((self.layer4(q1))))
        q1 = self.activate(self.norm((self.layer5(q1))))
        q1 = self.activate(self.norm((self.layer6(q1))))
        q1 = self.layer7(q1)
        
        # Q2前向傳播
        q2 = self.activate(self.norm((self.layer8(sa))))
        q2 = self.activate(self.norm((self.layer9(q2))))
        q2 = self.activate(self.norm((self.layer10(q2))))
        q2 = self.activate(self.norm((self.layer11(q2))))
        q2 = self.activate(self.norm((self.layer12(q2))))
        q2 = self.activate(self.norm((self.layer13(q2))))
        q2 = self.layer14(q2)
        return q1, q2


# ============== Replay Buffer ==============
class ReplayBuffer:
    """優化的經驗回放緩衝區"""
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # 預分配數組
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        )
    
    def __len__(self):
        return self.size


# ============== TD3 Agent ==============
class TD3Agent:
    def __init__(self, state_dim, action_dim):
        # Actor 網路
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.AdamW(  # AdamW 比 Adam 更好
            self.actor.parameters(), 
            lr=LR_ACTOR,
            betas=(0.9, 0.999),  # 使用標準 betas
            weight_decay=1e-4
        )
        
        # Critic 網路
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), 
            lr=LR_CRITIC,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim, device)
        self.total_iterations = 0
    
    def select_action(self, state, add_noise=True):
        """選擇動作 - 優化版本"""
        # 直接用 torch 處理
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        
        with torch.no_grad():  # 推理時不需要梯度
            action = self.actor(state.unsqueeze(0))
        
        action = action.cpu().numpy().flatten()
        
        if add_noise:
            # 自適應噪聲衰減
            noise_std = EXPLORATION_NOISE * max(0.1, 1.0 - self.total_iterations / 50000)
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
        action = action / (np.linalg.norm(action) + 1e-8)
        return action
    
    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None, None
        
        # 採樣
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        # ===== 更新 Critic =====
        with torch.no_grad():
            # 目標動作 + 噪聲
            next_actions = self.actor_target(next_states)
            noise = torch.randn_like(next_actions) * TARGET_NOISE
            noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = next_actions + noise
            
            # 雙 Q 學習
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * GAMMA * target_Q
        
        # 當前 Q 值
        current_Q1, current_Q2 = self.critic(states, actions)
        
        # Critic 損失（使用 Huber loss 更穩定）
        critic_loss = nn.SmoothL1Loss()(current_Q1, target_Q) + \
                      nn.SmoothL1Loss()(current_Q2, target_Q)
        
        # 優化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        actor_loss = None
        
        # ===== 延遲更新 Actor =====
        if self.total_iterations % POLICY_UPDATE_FREQ == 0:
            # Actor 損失
            actor_loss = -self.critic(states, self.actor(states))[0].mean()
            
            # 優化 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            # 軟更新目標網路
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
            actor_loss = actor_loss.item()
            
        self.total_iterations += 1
        
        return critic_loss.item(), actor_loss
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_iterations': self.total_iterations
        }, filepath)
    
    def load(self, filepath):
        """載入模型"""
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_iterations = checkpoint['total_iterations']


# ============== 環境定義 ==============
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.dt = 0.125
        self.T = 5
        self.steps = int(self.T / self.dt)
        self.current_step = 0
        self.x = 0.0
        self.y = 0.0
        self.u = 0.0
        self.v = 0.0
        self.prev_x = 0.0
        self.time = 0
        self.initial_x = 0.0
        self.omega = 3.0
        self.B = 1.0
        
        self.x_history = []
        self.y_history = []
        
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        self.x = 0.0
        self.y = 0.0
        self.initial_x = 0.0
        self.u = 0.0
        self.v = 0.0
        self.prev_x = 0.0
        self.time = 0
        
        self.x_history = [self.x]
        self.y_history = [self.y]
        
        return np.array([self.x, self.y], dtype=np.float32)
    
    def step(self, action):
        self.u, self.v = action
        direction = np.array([self.u, self.v])
        norm = np.linalg.norm(direction) + 1e-8
        direction = direction / norm
        
        # ODE 求解
        def ode(t, s):
            x, y = s
            Vx = 4 * (- np.sin(2 * np.pi * x + self.B * np.sin(2 * np.pi * self.omega * t)) * np.cos(2 * np.pi * y))
            Vy = 4 * ( np.cos(2 * np.pi * x + self.B * np.sin(2 * np.pi * self.omega * t)) * np.sin(2 * np.pi * y))
            return [Vx + direction[0], Vy + direction[1]]
        
        s0 = [self.x, self.y]
        sol = solve_ivp(ode, [0, self.dt], s0, method='RK45', t_eval=[self.dt])
        
        self.x, self.y = sol.y[:, -1]
        self.time += self.dt
        
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        
        # 獎勵設計
        progress_reward = 2 * (self.x - self.prev_x)
        reward = progress_reward 
        
        self.prev_x = self.x
        self.current_step += 1
        done = self.current_step >= self.steps
        
        if done:
            total_progress = self.x - self.initial_x
            reward += total_progress * 1
        
        return np.array([self.x, self.y], dtype=np.float32), reward, done, {}

def plot_training_progress(agent, env, episode, test_steps=400):
    """測試並繪製軌跡"""
    original_steps = env.steps
    original_dt = env.dt
    
    env.steps = test_steps
    env.dt = env.T / test_steps
    
    state = env.reset()
    x_history = [state[0]]
    y_history = [state[1]]
    
    for step in range(test_steps):
        action = agent.select_action(state, add_noise=False)
        next_state, _, done, _ = env.step(action)
        x_history.append(next_state[0])
        y_history.append(next_state[1])
        state = next_state
        if done:
            break
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_history, y_history, label=f"Episode {episode}", color="black", linewidth=2)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Agent Trajectory at Episode {episode}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"trajectory_ep{episode}.png", dpi=150)
    plt.show()
    
    env.steps = original_steps
    env.dt = original_dt
    
    print(f"Episode {episode}: Final X = {x_history[-1]:.4f}")
    return x_history[-1]


# ============== 主訓練循環 ==============
def main():
    env = CustomEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    
    # 訓練參數
    max_episodes = 30000
    max_steps = 40
    plot_interval = 250
    
    # 記錄
    reward_history = []
    critic_loss_history = []
    actor_loss_history = []
    final_x_history = []
    
    print(f"\n{'='*50}")
    print(f"Starting training on {device}")
    print(f"{'='*50}\n")
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # 訓練
            critic_loss, actor_loss = agent.train()
            if critic_loss is not None:
                critic_loss_history.append(critic_loss)
            if actor_loss is not None:
                actor_loss_history.append(actor_loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        reward_history.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episode {episode+1:4d} | Reward: {episode_reward:7.2f} | Avg(100): {avg_reward:7.2f}")
        
        # 定期測試和繪圖
        if (episode + 1) % plot_interval == 0:
            print(f"\n{'='*50}")
            print(f"Testing at Episode {episode+1}")
            print(f"{'='*50}")
            
            final_x = plot_training_progress(agent, env, episode+1, test_steps=400)
            final_x_history.append(final_x)
            
            # 繪製訓練曲線
            if len(reward_history) >= plot_interval:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Reward
                axes[0, 0].plot(reward_history, alpha=0.3, linewidth=0.5)
                window = min(50, len(reward_history))
                moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(reward_history)), moving_avg, 'r-', linewidth=2)
                axes[0, 0].set_xlabel("Episode")
                axes[0, 0].set_ylabel("Reward")
                axes[0, 0].set_title("Training Reward")
                axes[0, 0].grid(True, alpha=0.3)
                
                # Critic Loss
                if critic_loss_history:
                    axes[0, 1].plot(critic_loss_history, alpha=0.5, linewidth=0.5)
                    axes[0, 1].set_xlabel("Training Step")
                    axes[0, 1].set_ylabel("Critic Loss")
                    axes[0, 1].set_title("Critic Loss")
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Actor Loss
                if actor_loss_history:
                    axes[1, 0].plot(actor_loss_history, alpha=0.5, linewidth=0.5)
                    axes[1, 0].set_xlabel("Training Step")
                    axes[1, 0].set_ylabel("Actor Loss")
                    axes[1, 0].set_title("Actor Loss")
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Final X Position
                if final_x_history:
                    axes[1, 1].plot(range(plot_interval, episode+2, plot_interval), final_x_history, 'o-', linewidth=2)
                    axes[1, 1].set_xlabel("Episode")
                    axes[1, 1].set_ylabel("Final X Position")
                    axes[1, 1].set_title("Performance Over Time")
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"training_progress_ep{episode+1}.png", dpi=150)
                plt.show()
            
            # 保存模型
            agent.save(f"td3_checkpoint_ep{episode+1}.pth")
            print(f"Model saved to td3_checkpoint_ep{episode+1}.pth\n")
    
    # 最終測試
    print(f"\n{'='*50}")
    print("Final Training Result")
    print(f"{'='*50}")
    plot_training_progress(agent, env, max_episodes, test_steps=400)
    agent.save("td3_final.pth")
    print("Training completed!")


if __name__ == "__main__":
    main()
