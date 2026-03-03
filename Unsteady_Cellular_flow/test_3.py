import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from Unsteady_Cellular_flow.Unsteady_Cellular_flow_3 import CustomEnv, TD3Agent

# ================ 載入訓練好的 Actor =================
env = CustomEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = TD3Agent(state_dim, action_dim, omega=env.omega)
agent.load("td3_checkpoint_ep_time500.pth")   # ← 使用你訓練時存的最佳模型
agent.actor.eval()                       # 關閉 dropout、BN 等

def rollout_trajectory(agent, env, n_steps=400):
    agent.actor.eval()

    # 固定總時間
    env.dt = env.T / n_steps
    env.steps = n_steps

    state = env.reset()

    x_history = [state[0]]
    y_history = [state[1]]

    for _ in range(n_steps):
        action = agent.select_action(state, add_noise=False)
        state, reward, done, _ = env.step(action)

        x_history.append(state[0])
        y_history.append(state[1])

    # 繪圖
    plt.figure(figsize=(6, 6))
    plt.plot(x_history, y_history, 'k-', lw=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Final x = {x_history[-1]:.4f}")
    return x_history,y_history


rollout_trajectory(agent, env, n_steps=400)
