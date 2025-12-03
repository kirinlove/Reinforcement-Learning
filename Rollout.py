import torch
import numpy as np
import copy


# ================ 載入訓練好的 Actor =================
env = CustomEnv()
env.steps = 400        
env.dt = env.T / 400   # 調整 dt，確保總時間不變
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = TD3Agent(state_dim, action_dim)
agent.load("td3_checkpoint_ep4500.pth")   # ← 使用你訓練時存的最佳模型
agent.actor.eval()                       # 關閉 dropout、BN 等

print("Loaded actor from checkpoint 4500")

def rollout_action_selection(env, agent, state,
                             num_candidates=30,
                             noise_std=0.15,
                             rollout_horizon=500):
    """
    參數說明：
    - num_candidates：產生多少個候選動作（每回合）
    - noise_std：高斯擾動強度
    - rollout_horizon：後續模擬長度，直到 episode 結束
    """
    
    base_action = agent.select_action(state, add_noise=False)
    
    # 候選動作
    candidates = [base_action.copy()]
    for _ in range(num_candidates - 1):
        noise = np.random.normal(0, noise_std, size=base_action.shape)
        a = base_action + noise
        a = a / (np.linalg.norm(a) + 1e-8)
        candidates.append(a)

    best_final_x = -999
    best_action = candidates[0]

    # 對每個候選動作做 rollout 模擬
    for action in candidates:
        sim_env = copy.deepcopy(env)
        
        # 第一步採用候選動作
        s, _, done, _ = sim_env.step(action)
        
        # 後續使用舊策略（actor）
        steps = 0
        while not done and steps < rollout_horizon:
            a2 = agent.select_action(s, add_noise=False)
            s, _, done, _ = sim_env.step(a2)
            steps += 1
        
        # 結束後看 final x
        final_x = sim_env.x
        
        if final_x > best_final_x:
            best_final_x = final_x
            best_action = action
    
    return best_action

state = env.reset()
xs = [state[0]]
ys = [state[1]]

for step in range(400):
    action = rollout_action_selection(env, agent, state,
                                      num_candidates=50,
                                      noise_std=0.25,
                                      rollout_horizon=1000)
    
    next_state, reward, done, _ = env.step(action)
    xs.append(next_state[0])
    ys.append(next_state[1])
    state = next_state
    
    if done:
        break

# 畫圖
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
plt.plot(xs, ys, linewidth=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectory with Rollout (Checkpoint 4500)")
plt.axis('equal')
plt.grid(True)
plt.show()

print(f"Final X = {xs[-1]}")
