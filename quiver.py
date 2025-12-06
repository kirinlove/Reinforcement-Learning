import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,8))

# ---- (A) 畫軌跡線 ----
plt.plot(xs, ys, linewidth=2, color='black', alpha=0.6, label='Trajectory')

# ---- (B) 畫箭頭（每次動作）----
ax = np.array([a[0] for a in actions])
ay = np.array([a[1] for a in actions])

plt.quiver(xs[:-1], ys[:-1], ax, ay, scale=50,
           color='red')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectory with Actions (Rollout)")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
