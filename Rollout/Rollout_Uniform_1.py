import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

# ================ 載入訓練好的 Actor =================
env = CustomEnv()
env.steps = 400        
env.dt = env.T / 400   # 調整 dt，確保總時間不變
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = TD3Agent(state_dim, action_dim)
agent.load("td3_checkpoint_ep8750.pth")   # ← 使用你訓練時存的最佳模型
agent.actor.eval()                       # 關閉 dropout、BN 等

print("Loaded actor from checkpoint 8750")

def flow_velocity(x, y):
    Vx = -6 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    Vy = 6 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return np.array([Vx, Vy], dtype=np.float32)

def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)   # 回傳 radians

def rollout_action_selection(env, agent, state,
                             num_candidates=30,
                             rollout_horizon=500):
    
    x, y = state
    V = flow_velocity(x, y)
    normV = np.linalg.norm(V)
    
    # -------- compute angle limit from article --------
    # θ_max = π - arccos(1/||V||)
    cos_arg = np.clip(1.0 / max(normV, 1.0), -1.0, 1.0)
    theta_max = np.pi - np.arccos(cos_arg)
    
    # avoid degenerate cases
    if normV < 1e-6:
        V = np.array([1.0, 0.0])  # arbitrary but safe direction
        theta_max = np.pi        # allow all angles
    
    base_action = agent.select_action(state, add_noise=False)
    
    # -------- generate raw candidates --------
    raw_candidates = [base_action.copy()]
    for _ in range(num_candidates - 1):
        theta_0 = np.random.uniform(0, 2*np.pi)
        a = np.array([np.cos(theta_0), np.sin(theta_0)], dtype=np.float32)
        raw_candidates.append(a)

    # -------- angle filtering --------
    candidates = []
    for a in raw_candidates:
        theta = angle_between(a, V)
        if theta <= theta_max:
            candidates.append(a)
    
    # fallback: no candidate passes filter
    if len(candidates) == 0:
        candidates = [base_action]

    # -------- rollout evaluation --------
    best_final_x = -999
    best_action = candidates[0]

    for action in candidates:
        sim_env = copy.deepcopy(env)
        
        s, _, done, _ = sim_env.step(action)

        steps = 0
        while not done and steps < rollout_horizon:
            a2 = agent.select_action(s, add_noise=False)
            s, _, done, _ = sim_env.step(a2)
            steps += 1
        
        final_x = sim_env.x
        
        if final_x > best_final_x:
            best_final_x = final_x
            best_action = action
    
    return best_action
    
state = env.reset()
xs = [state[0]]
ys = [state[1]]
actions = []

for step in range(400):
    action = rollout_action_selection(env, agent, state,
                                      num_candidates=500,
                                      rollout_horizon=1000)
    
    next_state, reward, done, _ = env.step(action)
    xs.append(next_state[0])
    ys.append(next_state[1])
    actions.append(action)
    state = next_state
    
    if done:
        break

# 畫圖

plt.figure(figsize=(8,8))
plt.plot(xs, ys, linewidth=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectory with Rollout (Checkpoint 4500)")
plt.axis('equal')
plt.grid(True)
plt.show()

print(f"Final X = {xs[-1]}")
