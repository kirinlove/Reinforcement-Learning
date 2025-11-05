\documentclass{report}
\usepackage[top=23mm, bottom=35mm, left=30mm, right=25mm]{geometry}  % 調整邊界
\usepackage{amsmath, amssymb}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{caption} 
\usepackage{subcaption}
\linespread{1.4} % 1.4倍行距
\selectfont      
\begin{document}

\chapter{Introduction}

Optimal control has numerous applications in engineering, aeronautics and finance. For instance, robotic manipulators, trajectory 
tracking and quantitative investment. In theory of partial differential equation, a class of the optimal control problem is 
corresponded to Hamilton-Jacobi-Bellman (HJB) equation.

\bigskip

In this article, we discuss a trajectory optimization problem:

\bigskip

\textit{Consider the ODE}
\begin{subequations}\label{eq:system}
\begin{equation}
\begin{cases}
\dot{\bm{x}}(s) = \bm{V}(s, \bm{x}(s)) + \bm{\alpha}(s) \\
\bm{x}(0) = \bm{x},
\end{cases}
\label{eq:ode}
\end{equation}

\textit{the optimal problem is given by}
\begin{equation}
\begin{aligned}
    &\text{maximize } \bm{x}(T) \cdot \bm{e}_1 \\
    &\text{subject to } ||\bm{\alpha}(\cdot)|| \leq 1.
\end{aligned}
\label{eq:opt}
\end{equation}
\end{subequations}

The particle $\bm{x}$ travels along the flow velocity $\bm{V}$ and the control $\bm{\alpha}$. The admissible control has 
magnitude less than 1 within any direction. Given the initial position $\bm{x} \in \mathbb{R}^n$, we would like to find the best 
control such that the corresponding trajectory travels in $x$-direction as far as possible in time $[0, T]$.

\bigskip

This problem corresponds to the study of turbulent flame speed of G-equation in combustion theory:
\begin{equation}
G_t = \bm{V}(\bm{x}, t) \cdot \nabla G + |\nabla G|,
\label{eq:gequation}
\end{equation}
a level-set type HJB equation that describes the motion of flame front by flow velocity $\boldsymbol{V}$ and unit laminar speed. 
However, finding the best trajectory among all possible admissible controls suffers from the \textit{curse of dimensionality}. 
This issue commonly arises in the domain of financial engineering and data science (machine learning, neural network). Thanks for 
the advancement of computers and the inventions of the algorithms, we expect that the optimization problem (1.1) can be approached 
by \textbf{Reinforcement Learning}.

\bigskip

In chapter 2, We reviewed various definitions in reinforcement learning and introduced the algorithm that we used to solve 
optimization problem of (1.1).

\bigskip

In chapter 3, we present results of optimization problem (1.1) with various choices of the flow velocity $\boldsymbol{V}$, 
including the cellular flow, Rayleigh-Benard advection, and the time periodically shifted cellular flow. Then we compare the computed turbulent flame speed obtained by 
solving optimization problem (1.1).

\chapter{Reinforcement Learning}

\section{Basic Definitions}

\textbf{Reinforcement Learning (RL)} is a machine learning approach where an \textbf{agent} learns to make optimal decisions by interacting 
with an \textbf{environment}. At each time step, the agent observes a \textbf{state}, selects an \textbf{action} based on its 
current \textbf{policy}, receives a \textbf{reward} from the environment, and transitions to a new state. The agent's goal is to learn a 
policy that maximizes the total accumulated reward over time, which includes the following elements:

\begin{itemize}
\item \textbf{Agent}: In reinforcement learning, an agent is the decision-making entity that interacts with the environment. 
The goal of the agent is to learn an optimal policy through repeated interactions with the environment, enabling it to select 
the best actions in various states.

The agent operates as follows:
\begin{enumerate}
    \item At each time step, the agent observes the current state of the environment.
    \item It selects an action based on its policy.
    \item The environment responds to the action by returning a new state and a reward.
    \item The agent uses this feedback to improve its policy, aiming to maximize long-term cumulative reward.
\end{enumerate}
An agent can be a robot, a software controller, or a simulated intelligent entity. In this work, the agent is not a specific 
physical entity; instead, it is an abstract system that includes the policy (neural network), the learning algorithm (TD3), and 
interaction with the environment.
\item \textbf{State ($s$)}: A representation of the current situation of the environment. For example, in robot control tasks, it 
may include the robot’s position and velocity. In this work, the state was initially defined as time. However, using time alone 
as the input caused convergence problems during training. Therefore, we redefined the state as the position $(x,y)$ to improve 
learning stability. Furthermore, by using position as the input, we can subsequently create phase portraits.
\item \textbf{Action ($a$)}: The actions represent the possible choices or controls available to the agent, such as moving in a certain direction or applying a force. 
In this work, the action represents the direction. Initially, we considered using the angle $\theta$ as the action and represented the direction as $(cos\theta,sin\theta)$. 
However, this approach did not perform well, so we adopted a two-dimensional action (u,v), which was later normalized to obtain the direction.
\item \textbf{Reward ($r$)}: A scalar feedback signal provided by the environment in response to the agent’s action, which guides 
the learning process. Reward design is very important, as it largely affects how well the model converges. In this work, based on 
optimization problem (1.1), our goal is to maximize $x(T)$, so we simply used $x(T)$ as the reward. However, this approach provides 
a reward only at the final time step, while all intermediate steps yield no feedback. This is known as a \textit{sparse reward}, 
which can make training extremely difficult in reinforcement learning. To improve this, we added 2*($x$ - previous $x$) as an extra 
reward at each time step to encourage movement in the $x$-direction.
\item \textbf{Environment}: The system that reacts to the agent’s actions, determines state transitions, and provides rewards. 
All entities or factors outside the agent.
In this work, it represents the complete information of the ordinary differential equation (ODE) and the flow velocity $\bm{V}$.
\item \textbf{Policy}: In reinforcement learning, a policy is the mechanism by which an agent decides what action to 
take in a given state. In simple terms, the policy is the agent’s behavior function. Policies can be either 
\textit{deterministic} or \textit{stochastic}:
\begin{itemize}
  \item \textbf{Deterministic Policy}: Maps each state $s$ to a specific action $a$, denoted by $\mu(s) = a$.
  \item \textbf{Stochastic Policy}: Defines a probability distribution over actions given a state, denoted by $\pi(a|s)$, which 
  specifies the probability of taking action $a$ in state $s$.
\end{itemize}
The objective of reinforcement learning is to find an optimal policy that enables the agent to maximize the expected cumulative 
reward through interaction with the environment. In this work, we use a deterministic policy, and the algorithm is
\textbf{Twin Delayed Deep Deterministic Policy Gradient (TD3)}. The policy is represented by a neural network, where the input is 
the state $(x,y)$ and the output is the action $(u,v)$.
\end{itemize}

\bigskip

In traditional reinforcement learning, when the state or action space is too large (such as with high-dimensional image inputs), 
the learning process becomes very difficult. To slove this issue, deep learning is introduced as a function approximator to 
estimate the policy function and value function, call \textbf{Deep Reinforcement Learning (DRL)}. This enables deep 
reinforcement learning to handle high-dimensional inputs and output, significantly expanding the range of applications for 
reinforcement learning.

\section{Markov Decision Process (MDP)}

The reinforcement learning problem is usually a \textbf{Markov Decision Process (MDP)}, it is used to describe how an agent 
interacts with an environment by selecting actions based on the current state and receiving rewards in order to achieve long-term 
goals (see Figure 1). The Markov Decision Process provides a formalized approach to defining the problem of learning and 
optimizing behavioral policies, it can be represented as a 4-tuple 
$(S, A, P, R)$, where:
\begin{itemize}
\item $\mathcal{S}$: State space, which represents the set of all possible states.
\item $\mathcal{A = A}(s)$: Action space, which represents the set of all possible actions the agent can take in a given state.
\item ${P = P}(s'|s,a)$: State transition probability, which represents the probability of moving to the next state given 
that $s'$ the agent takes action $a$ in state $s$.
\item $\mathcal{R = R}(s,a)$: Reward, which represents the set (or distribution) of immediate rewards the agent receives after 
taking action $a$ in state $s$.
\end{itemize}

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=0.8\textwidth]{RL_1.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 1: The interaction between the agent and the environment in a Markov Decision Process.}
\end{figure}

\bigskip

A Markov Decision Process is characterized by the Markov property: the future state transition depends only on the current state 
and action, and not on the history of past states or actions. That means:
\begin{equation}
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t).
\end{equation}

\bigskip

Markov Decision Process is the theoretical basis for most reinforcement learning problems. Whether it is Q-learning, DQN, 
Actor-Critic, or PPO algorithms, their goal is to learn the best policy in a given Markov Decision Process so that the agent can 
maximize long-term rewards. Although in real applications we often do not know the state transition probabilities $P$ exactly, 
the main idea of reinforcement learning is to estimate this information through interaction and experience.

\section{Actor-Critic}

Classic reinforcement learning methods can generally be divided into \textbf{value-based} methods and \textbf{policy-based} 
methods. The core idea of value-based methods is to evaluate the value of all possible actions in a given state and then choose 
the one with the highest expected return (i.e., the expected cumulative reward). A typical example of this approach is DQN. 
It can be thought of as "scoring" all the options before making a decision. On the other hand, policy-based methods directly 
learn a policy, without calculating the value of each action. Instead, they aim to learn a decision-making behavior. 
REINFORCE is a well-known example of this type. In simple terms, value-based methods involve evaluation before decision-making, 
while policy-based methods make decisions directly. They can both use deep neural networks to represent the value function and 
the policy function, respectively.

\bigskip

Can we construct two neural networks that can learn both the value function and the policy function? The answer is "yes". 
This is exactly what the famous Actor-Critic method does.

\bigskip

Actor-Critic is an important class of algorithmic frameworks in reinforcement learning. It combines the strengths of both 
value-based and policy-based methods, making it one of the most successful methods in modern deep reinforcement learning. 
The core idea is to separate the policy function (the actor) and the value function (the critic), allowing them to work together 
cooperatively.
\begin{itemize}
\item \textbf{Actor}: The policy function $\pi_\phi(a|s)$ ($\phi$ represents the parameters of the neural network) responsible for selecting 
actions.
\item \textbf{Critic}: The value function $V_\theta(s,a)$ or $Q_\theta(s,a)$ ($\theta$ represents the parameters of the neural network) that 
evaluates the quality of actions.
\end{itemize}
The workflow is as follows:
\begin{enumerate}
    \item The Actor selects an action based on the state and the current policy, resulting in a state-action pair.
    \item This state-action pair interacts with the environment to receive a reward.
    \item The Critic evaluates the value (i.e., the quality) of the current state-action pair based on the received reward.
    \item The feedback from the Critic is then used to improve the Actor's policy.
\end{enumerate}

\section{Twin Delayed Deep Deterministic Policy Gradient (TD3)}

In this work, we use the TD3 algorithm because the action space is continuous, so the classic algorithm DQN can't be used. Other common algorithms for continuous action 
spaces include TD3, PPO, and SAC. After testing them, TD3 gave the best results. TD3 (Twin Delayed Deep Deterministic Policy Gradient) is an improved version of DDPG
(Deep Deterministic Policy Gradient), so before understanding TD3, it's important to first know DDPG.

\bigskip

DDPG is an algorithm specifically designed for continuous action spaces and cannot be used for discrete action spaces. It was proposed to address the limitation of DQN, 
which doesn't work with continuous actions. Although the name includes "Policy Gradient", DDPG is actually an actor-critic method that combines DQN with deep policy 
gradient techniques. The word "Deterministic" indicates that it uses a deterministic policy (For details, see Figure 2).

\bigskip

DDPG has four main important techniques: 
\begin{itemize}
\item \textbf{Experience Replay Buffer}: Experience Replay Buffer is a large set used to store data. From the actor network, we can get the current action $a_t$ based 
on the current state $s_t$. After applying state-action pair $(s_t,a_t)$ to the environment, we receive the current reward $r_t$ and the next state $s_{t+1}$. In this 
way, we obtain one data $(s_t,a_t,r_t,s_{t+1})$. At every time step t, we do this and put the data into the Experience Replay Buffer. When training, we randomly 
sample a batch of data from the Experience Replay Buffer to break the correlations.
\item \textbf{Target Network}: To enhance the stability of the learning process, DDPG introduces two additional networks: the target actor network and the target critic 
network — for a total of four neural networks. Their main purpose is to prevent the learning targets from fluctuating drastically due to frequent parameter updates, 
which could lead to unstable or divergent training. The actor and target actor networks both represent the policy function, while the critic and target critic networks 
both represent the value function.
\item \textbf{Soft Update}: As mentioned earlier, the purpose of the target networks is to prevent the learning targets from fluctuating drastically with parameter 
updates. Therefore, the target networks cannot be updated in the same way as the main networks. So how are they updated?
DDPG adopts a soft update approach, which means that at each step, the parameters of the main networks are gradually transferred to the target networks using a small 
factor $\tau$ (e.g., 0.001). The update formula is as follows:
\begin{equation}
\theta^{Q'} \leftarrow \tau\theta^{Q} + (1-\tau)\theta^{Q'}
\end{equation}
\begin{equation}
\theta^{\mu'} \leftarrow \tau\theta^{\mu} + (1-\tau)\theta^{\mu'},
\end{equation}
here, $\theta^{Q}$ represents the parameters of the critic network, and $\theta^{Q'}$ represents the parameters of the target critic network. $\theta^{\mu}$ represents 
the parameters of the actor network, and $\theta^{\mu'}$ represents the parameters of the target actor network.
\item \textbf{Exploration}: This refers to the agent choosing actions that are not currently optimal in order to gain more information about the value of those actions. 
The goal is to gather information, with the aim of making better decisions in the future. \\
Since the policy(actor network) is deterministic, it does not explore by itself. Therefore, DDPG adds noise (e.g., Gaussian noise) to the actions output by the actor to 
enable the agent to explore the environment. Therefore, the action $a_t$ stored in the experience replay buffer mentioned above is actually:
\begin{equation}
a_t = \mu(s_t|\theta^{\mu}) + N_t,
\end{equation}
with the $N_t$ is the exploration noise.
\end{itemize}

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{DDPG_1.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 2: DDPG algorithm pseudocode.}
\end{figure}

\bigskip

It’s important to note that in reinforcement learning, we often encounter the trade-off between exploration and exploitation (This refers to the agent selecting the 
action it currently believes has the highest value based on existing knowledge. The goal is to exploit the known best options in order to obtain stable and good returns). 
The agent faces a fundamental challenge: should it exploit the currently known actions that yield higher returns, or should it explore other actions that may have 
potentially higher returns but have not yet been thoroughly evaluated?

\bigskip

This trade-off is crucial because it directly impacts the agent's learning efficiency and final performance. If the agent focuses too much on exploitation, it may 
converge prematurely to a sub-optimal policy and get stuck in a local optimum. Conversely, if the agent focuses too much on exploration, the learning process can become 
inefficient. The agent will spend excessive resources trying out poor actions, resulting in slow convergence or even instability. An effective reinforcement learning 
algorithm must strike a delicate balance between exploration and exploitation in order to achieve both learning efficiency and policy optimality.

\bigskip

Although DDPG is powerful, during training the Q-values tend to be overestimated, which can lead to unstable or divergent policy performance. Therefore, to improve DDPG, 
TD3 was introduced.

\bigskip

The problem of Q-value overestimation is not unique to DDPG. It is a common issue among Q-Learning-based algorithms, including DQN. The main reason lies in the use of 
neural networks as function approximators to estimate Q-values. Neural networks are not perfect—they always introduce some estimation error. In Q-Learning's update rule, there 
is a "maximization" operation, which selects the action with the highest estimated Q-value. However, this does not necessarily mean that the selected action has the 
highest true Q-value—it simply has the highest estimated Q-value. It's called \textbf{maximization bias}.

\bigskip

A single overestimation is not a big problem. But the real problem is that the algorithm uses bootstrapping, causing the bias to propagate and accumulate over time 
(see Figure 3).

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=0.85\textwidth]{DDPG_2.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 3: Q-value overestimation propagates and accumulates over time.}
\end{figure}


\bigskip

\newpage

TD3 is an improved version of DDPG. They are largely the same (see Figure 4), with the main improvements being:
\begin{itemize}
\item \textbf{Twin Q-Networks}: To address the problem of Q-value overestimation, TD3 introduces double Q-network. This approach simultaneously trains two independent 
critic networks and uses the minimum of their predictions when calculating the target value. The paper [] theoretically proves that this operation introduces an 
underestimation bias, which can offset the overestimation bias in Q-Learning. Furthermore, their comparative experiments show that using two networks already captures 
the majority of the gains, while increasing to three or more networks leads to diminishing returns and does not provide significant additional benefits. Therefore, TD3 
adopts the double Q-network design to balance performance and computational efficiency.
\item \textbf{Target Policy Smoothing}: To improve policy stability, a small clipped Gaussian noise is added to the target action to prevent the target Q-value 
computation from being overly deterministic. To reduce the Q-network's dependence on the estimated Q-value of a single action during the training process.This also 
helps improve the generalization ability during training, making the learned policy smoother and more stable.
\item \textbf{Delayed Update}: In order to improve the accuracy of value estimation, TD3 reduces the update frequency of the actor network compared to the critic 
networks. Specifically, the actor network and target networks are updated once for every multiple updates of the critic network (e.g., every two critic updates).
This ensures that the critic’s value function is sufficiently accurate before the policy is updated, preventing the policy from being updated based on inaccurate 
Q-value.
\end{itemize}

\newpage

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{TD3_1.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 4: TD3 algorithm pseudocode.}
\end{figure}

\bigskip

\chapter{Training Results}

In this chapter, we present the training results for optimization problem (1.1), solved using the TD3 algorithm, under the 
following flow velocity conditions:

\bigskip

\begin{itemize}
\item Cellular flow
\end{itemize}
\begin{equation}
\bm{V}(x,y)=A \cdot \langle -\sin{(2\pi x)}\cos{(2\pi y)},\cos{(2\pi x)} \sin{(2\pi y)} \rangle.
\end{equation}

\bigskip

\begin{itemize}
\item Rayleigh-Benard advection
\end{itemize}
\begin{equation}
\mathbf{V}(x,y,t) = A \cdot \langle \cos(y) + \sin(y) \cos(\omega t), \cos(x) + \sin(x) \cos(\omega t) \rangle.
\end{equation}

\bigskip

\begin{itemize}
\item the time periodically shifted cellular flow
\end{itemize}
\begin{equation}
\mathbf{V}(\mathbf{x},t) = A \cdot \langle -\sin\bigl(2\pi x + B\sin(2\pi \omega t)\bigr)\cos(2\pi y),\cos\bigl(2\pi x + B\sin(2\pi \omega t)\bigr)\sin(2\pi y) \rangle.
\end{equation}

\bigskip

where \(A>0\) is the intensity, \(B\) is the amplitude. Note that only the flow (3.1) is time-independent, while flows (3.2) and 
(3.3) vary with time. Recall that our goal is to find the optimal trajectories that travel as far as possible in \(x\)-direction.

\bigskip

In [], the authors numerically estimated the turbulent speeds in the cellular flow, Rayleigh-Bénard advection, and the 
time-periodically shifted cellular flow by either constructing specific controls or finding periodic orbits in the \(x\)-direction. 
In the present paper, instead of analyzing the flow fields in detail, we directly apply machine learning methods to obtain the 
results.

\newpage

\section{Cellular Flow}
We consider the optimization problem (1.1) for cellular flow in $2$-dimensional space. In this case, the flow velocity V is time-independent.
\begin{equation}
V(\bm{x},t) = V(\bm{x}),
\end{equation}
then we can simplify Problem (1.1) by assuming
\begin{equation}
||\bm{\alpha}(\cdot)|| = 1.
\end{equation}

\bigskip

For a given admissible contral $\bm{\alpha}$, assume the corresponding trajectory $\bm{x}$ is smooth with
\begin{equation}
\dot{\bm{x}}(s) = \bm{V}(\bm{x}(s)) + \bm{\alpha}(s) \neq 0, \quad s \in [0,T].
\end{equation} 
For any $s \in [0,T]$, there exists $p(s) \geq  0$ such that
\begin{equation}
||\bm{\alpha} + p(\bm{V} + \bm{\alpha})|| = 1
\end{equation} 
\begin{equation}
||\bm{\alpha} + p(\bm{V} + \bm{\alpha})||^2 = 1
\end{equation}
\begin{equation}
||\bm{\alpha}||^2 + 2p[\bm{\alpha} \cdot (\bm{V} + \bm{\alpha})] + p^2||\bm{V} + \bm{\alpha}||^2 = 1
\end{equation}
\begin{equation}
\underbrace{||\bm{V} + \bm{\alpha}||^2}_{a}p^2 + \underbrace{(2\bm{V} \cdot \bm{\alpha} + 2||\bm{\alpha}||^2)}_{b}p + \underbrace{(||\bm{\alpha}||^2 - 1)}_{c} = 0.
\end{equation}
\begin{align}
    \Delta &= b^2 - 4ac = (2\bm{V} \cdot \bm{\alpha} + 2||\bm{\alpha}||^2)^2 - 4||\bm{V} + \bm{\alpha}||^2(||\bm{\alpha}||^2 - 1) \\
    &= 4[(\bm{V} \cdot \bm{\alpha})^2 + 2||\bm{\alpha}||^2(\bm{V} \cdot \bm{\alpha}) + ||\bm{\alpha}||^4] \notag \\
    &\quad - 4[||\bm{V}||^2||\bm{\alpha}||^2 - ||\bm{V}||^2 + 2||\bm{\alpha}||^2(\bm{V} \cdot \bm{\alpha}) - 2(\bm{V} \cdot \bm{\alpha}) + ||\bm{\alpha}||^4 - 
    ||\bm{\alpha}||^2] \\
    &= 4[(\bm{V} \cdot \bm{\alpha})^2 - ||\bm{V}||^2||\bm{\alpha}||^2 +  ||\bm{V}||^2 + ||\bm{\alpha}||^2 + 2(\bm{V} \cdot \bm{\alpha})] \\
    &= 4[(\bm{V} \cdot \bm{\alpha})^2 - ||\bm{V}||^2||\bm{\alpha}||^2 + ||\bm{V} + \bm{\alpha}||^2].
\end{align}
It is easy to prove that $\Delta \geq 0$ if $||\bm{\alpha}|| \leq 1$.

\bigskip

Now, choose 
\begin{equation}
p = \frac{-b + \sqrt{b^2 - 4ac}}{2a} = \frac{-(\bm{V} \cdot \bm{\alpha}) - ||\bm{\alpha}||^2 + \sqrt{-||\bm{V}||^2||\bm{\alpha}||^2 + 
||\bm{V} + \bm{\alpha}||^2 + (\bm{V} \cdot \bm{\alpha})^2}}{||\bm{V} + \bm{\alpha}||^2}.
\end{equation}
We need to prove that $p \geq 0$.

\bigskip

First, consider
\begin{align}
    &\quad [-||\bm{V}||^2||\bm{\alpha}||^2 + ||\bm{V} + \bm{\alpha}||^2 + (\bm{V} \cdot \bm{\alpha})^2] - [((\bm{V} \cdot \bm{\alpha})) + ||\bm{\alpha}||^2]^2 \\
    &\quad = ||\bm{V} + \bm{\alpha}||^2 - ||\bm{V}||^2||\bm{\alpha}||^2 - 2||\bm{\alpha}||^2(\bm{V} \cdot \bm{\alpha}) - ||\bm{\alpha}||^4 \\
    &\quad = ||\bm{V}||^2 + 2(\bm{V} \cdot \bm{\alpha}) + ||\bm{\alpha}||^2 - ||\bm{V}||^2||\bm{\alpha}||^2 - 2||\bm{\alpha}||^2(\bm{V} \cdot \bm{\alpha}) - 
    ||\bm{\alpha}||^4 \\
    &\quad = ||\bm{V}||^2(1 - ||\bm{\alpha}||^2) + ||\bm{\alpha}||^2(1 - ||\bm{\alpha}||^2) + 2(\bm{V} \cdot \bm{\alpha})(1 - ||\bm{\alpha}||^2) \\ 
    &\quad = [||\bm{V}||^2 + ||\bm{\alpha}||^2 + 2(\bm{V} \cdot \bm{\alpha})](1 - ||\bm{\alpha}||^2)
\end{align}
Now, Consider the numerator of (3.15)
\begin{align}
    &\quad \sqrt{-||\bm{V}||^2||\bm{\alpha}||^2 + ||\bm{V} + \bm{\alpha}||^2 + (\bm{V} \cdot \bm{\alpha})^2} - [(\bm{V} \cdot \bm{\alpha}) + ||\bm{\alpha}||^2] \\
    &\quad = \frac{[-||\bm{V}||^2||\bm{\alpha}||^2 + ||\bm{V} + \bm{\alpha}||^2 + (\bm{V} \cdot \bm{\alpha})^2] - [(\bm{V} \cdot \bm{\alpha}) + ||\bm{\alpha}||^2]^2} 
    {\sqrt{-||\bm{V}||^2||\bm{\alpha}||^2 + ||\bm{V} + \bm{\alpha}||^2 + (\bm{V} \cdot \bm{\alpha})^2} + [(\bm{V} \cdot \bm{\alpha}) + ||\bm{\alpha}||^2]} \\
    &\quad = \frac{[||\bm{V}||^2 + ||\bm{\alpha}||^2 + 2(\bm{V} \cdot \bm{\alpha})](1 - ||\bm{\alpha}||^2)} 
    {\sqrt{-||\bm{V}||^2||\bm{\alpha}||^2 + ||\bm{V} + \bm{\alpha}||^2 + (\bm{V} \cdot \bm{\alpha})^2} + [(\bm{V} \cdot \bm{\alpha}) + ||\bm{\alpha}||^2]} \\
    &\quad = \frac{||\bm{V} + \bm{\alpha}||^2(1 - ||\bm{\alpha}||^2)}
    {\sqrt{-||\bm{V}||^2||\bm{\alpha}||^2 + ||\bm{V} + \bm{\alpha}||^2 + (\bm{V} \cdot \bm{\alpha})^2} + [(\bm{V} \cdot \bm{\alpha}) + ||\bm{\alpha}||^2]} \geq 0.
\end{align}
Therefore, $p \geq 0$.

\bigskip

Denote $\bm{\alpha}^* = \bm{\alpha} + p(\bm{V} + \bm{\alpha})$, then $\bm{\alpha}^*$ is also an admissible control with $||\bm{\alpha}^*|| = 1$ and the corresponding 
trajectory $\bm{x}^*$ is
\begin{align}
\dot{\bm{x}}^*(s) &= \bm{V}(\bm{x}(s)) + \bm{\alpha}^*(s) \\
&= \bm{V}(\bm{x}(s)) + \bm{\alpha}(s) + p(s)(\bm{V}(s) + \bm{\alpha}(s)) \\
&= \dot{\bm{x}}(s) + p(s)\dot{\bm{x}}(s) = (1 + p(s))\dot{\bm{x}}(s).
\end{align}
This implies that $\bm{x}^*$ and $\bm{x}$ follow the same trajectory, with $\bm{x}^*$ moving at a higher speed. 
\begin{equation}
    ||\dot{\bm{x}}^*|| = (1 + p)||\dot{\bm{x}}||, \quad p \geq 0.
\end{equation}

\bigskip

Therefore, the constraint in optimization Problem (1.1) only needs to be considered for admissible controls with norm equal to 1. This greatly simplifies our problem, 
as we only need to adjust the direction and no longer have to worry about the norm.

\bigskip

It is important to note that in problem (1.1), the objective is to maximize the x-coordinate. Therefore, we can to show the optimal control direction $\bm{\alpha}^*$ must 
satisfy
\begin{equation}
    \bm{V}(\bm{x}(s)) \cdot \bm{\alpha}^*(s) \geq 0, \quad \forall s \in [0,T],
\end{equation}
when $\bm{A}$ is sufficiently large

\bigskip

Prrof by contradiction, let the optimal control be $\bm{\alpha}^*$, suppose there exists $s_0 \in [0,T] $ such that
\begin{equation}
    \bm{V}(\bm{x}(s_0)) \cdot \bm{\alpha}^*(s_0) < 0.
\end{equation}
Since V is continuous, there exists an interval $I = [s_0,s_0 + \delta ]$ such that
\begin{equation}
    \bm{V}(\bm{x}(s)) \cdot \bm{\alpha}^*(s) < 0, \quad \forall s \in I.
\end{equation}
Now, define a new control
\begin{equation}
    \widetilde{\bm{\alpha}}(s) =
    \begin{cases}
        \bm{\alpha}^*(s) - 2\frac{\bm{V}(\bm{x}^*(s)) \cdot \bm{\alpha}^*(s)}{||\bm{V}(\bm{x}^*(s))||}\bm{V}(\bm{x}^*(s)), \quad \forall s \in I \wedge  
        \bm{V}(\bm{x}^*(s)) \cdot e_1 > 0\\
        \frac{\bm{V}(\bm{x}^*(s))}{||\bm{V}(\bm{x}^*(s))||}, \quad \forall s \in I \wedge \bm{V}(\bm{x}^*(s)) \cdot e_1 \leq 0\\
        \bm{\alpha}^*(s), \quad \forall s \notin  I,
    \end{cases}
\end{equation}
where the trajectories corresponding to controls $\widetilde{\bm{\alpha}}(s)$ and $\bm{\alpha}^*(s)$ are $\widetilde{\bm{x}}(s)$ and $\bm{x}^*(s)$, respectively. \\
Then
\begin{itemize}
    \item $||\widetilde{\bm{\alpha}}(s)|| = 1, \quad \forall s \in I $
    \item $\bm{V}(\bm{x}(s)) \cdot \widetilde{\bm{\alpha}}(s) = -\bm{V}(\bm{x}^*(s)) \cdot \bm{\alpha}^*(s) > 0, \quad \forall s \in I \wedge  
        \bm{V}(\bm{x}^*(s)) \cdot e_1 > 0$
    \item $\bm{V}(\bm{x}(s)) \cdot \widetilde{\bm{\alpha}}(s) = ||\bm{V}(\bm{x}^*(s))|| > 0, \quad \forall s \in I \wedge \bm{V}(\bm{x}^*(s)) \cdot e_1 \leq 0$.
\end{itemize}
Now, we will show that 
\begin{equation}
    \widetilde{\bm{x}}(T) \cdot e_1 > \bm{x}^*(T) \cdot e_1.
\end{equation}
Consider
\begin{equation}
    \bm{x}^*(T) \cdot e_1 = [\bm{x}^*(0) + \int_{0}^{T} \bm{V}(\bm{x}^*(s)) + \bm{\alpha}^*(s) \,ds ] \cdot e_1
\end{equation}
\begin{equation}
    \widetilde{\bm{x}}(T) \cdot e_1 = [\widetilde{\bm{x}}(0) + \int_{0}^{T} \bm{V}(\widetilde{\bm{x}}(s)) + \widetilde{\bm{\alpha}}(s) \,ds ] \cdot e_1.
\end{equation}
\begin{align}
    \widetilde{\bm{x}}(T) \cdot e_1 - \bm{x}^*(T) \cdot e_1 &= [\widetilde{\bm{x}}(T) - \bm{x}^*(T)] \cdot e_1 \\
    &= [\bm{x}(0) + \int_{0}^{T} \bm{V}(\widetilde{\bm{x}}(s)) + \widetilde{\bm{\alpha}}(s) \,ds - \bm{x}(0) - \int_{0}^{T} \bm{V}(\bm{x}^*(s)) + \bm{\alpha}^*(s) \,ds ] 
    \cdot e_1 \\
    &= [\int_{s_0}^{s_0 + \delta} \bm{V}(\widetilde{\bm{x}}(s)) + \widetilde{\bm{\alpha}}(s) \,ds - \int_{s_0}^{s_0 + \delta} \bm{V}(\bm{x}^*(s)) + \bm{\alpha}^*(s) \,ds] 
    \cdot e_1 \notag \\
    &\quad + [\int_{s_0 + \delta}^{T} \bm{V}(\widetilde{\bm{x}}(s)) + \widetilde{\bm{\alpha}}(s) \,ds - \int_{s_0 + \delta}^{T} \bm{V}(\bm{x}^*(s)) + \bm{\alpha}^*(s) \,ds]
    \cdot e_1 \\
    &= C + D
\end{align}
Within the interval I, since $\delta$ is small and $\bm{V}$ is continuous, the difference between the trajectories $\widetilde{\bm{x}}(s)$ and $\bm{x}^*(s)$ can be 
ignored. Therefore,
\begin{equation}
    C \approx [\int_{s_0}^{s_0 + \delta} \widetilde{\bm{\alpha}}(s) - \bm{\alpha}^*(s) \,ds] \cdot e_1 
\end{equation}
\begin{equation}
    D = [\int_{s_0 + \delta}^{T} \bm{V}(\widetilde{\bm{x}}(s)) - \bm{V}(\bm{x}^*(s)) \,ds] \cdot e_1.
\end{equation}
If $\bm{V}(\bm{x}^*(s)) \cdot e_1 > 0$,
\begin{equation}
    C \approx [\int_{s_0}^{s_0 + \delta} - 2 \frac{\bm{V}(\bm{x}(s)) \cdot \bm{\alpha}^*(s)}{||\bm{V}(\bm{x}(s))||^2} \bm{V}(\bm{x}(s)) \,ds] \cdot e_1 > 0.
\end{equation}
We can let
\begin{equation}
    C = c_0 \delta + o(\delta),
\end{equation}
with $c_0 > 0$ is independent of $\delta$. \\
Since $\bm{V}$ is Lipschitz,
\begin{equation}
    ||\bm{V}(\widetilde{\bm{x}}(s)) - \bm{V}(\bm{x}^*(s))|| \leq L||\widetilde{\bm{x}}(s) - \bm{x}^*(s)||.
\end{equation}
For $s \in [s_0 + \delta,T]$,
\begin{equation}
    ||\widetilde{\bm{x}}^\prime(s) - \bm{x}^{*\prime} (s)|| = ||\bm{V}(\widetilde{\bm{x}}(s)) - \bm{V}(\bm{x}^*(s))|| \leq L ||\widetilde{\bm{x}}(s) - \bm{x}^*(s)||.
\end{equation}
Solve the ode,
\begin{equation}
    ||\widetilde{\bm{x}}(s) - \bm{x}^*(s)|| \leq ||\widetilde{\bm{x}}(s_0 + \delta) - \bm{x}^*(s_0 + \delta)|| e^{L(s-(s_0 + \delta))},
\end{equation}
where
\begin{equation}
    ||\widetilde{\bm{x}}(s_0 + \delta) - \bm{x}^*(s_0 + \delta)|| \leq \int_{s_0}^{s_0 + \delta} ||\widetilde{\bm{\alpha}}(s) - \bm{\alpha}^*(s)|| \,ds.
\end{equation}
\begin{align}
    |D| &\leq [\int_{s_0 + \delta}^{T} ||\bm{V}(\widetilde{\bm{x}}(s)) - \bm{V}(\bm{x}^*(s))|| \,ds] \cdot e_1 \\
    &\leq 2\delta L [\int_{s_0 + \delta}^{T} e^{L(s-(s_0 + \delta))} \,ds] \cdot e_1 \\ 
    &= 2\delta(e^{L(T-(s_0 + \delta))} - 1) =: \delta K(\delta).
\end{align}
Thus, we can choose a proper $\delta$ such that
\begin{equation}
    (c_0 - K(\delta)) > 0 \Rightarrow C + D = \delta(c_0 - K(\delta)) + o(\delta) > 0.
\end{equation}
If $\bm{V}(\bm{x}^*(s)) \cdot e_1 \leq 0$,
\begin{equation}
    C \approx [\int_{s_0}^{s_0 + \delta} \frac{\bm{V}(\bm{x}^*(s))}{||\bm{V}(\bm{x}^*(s))||} - \bm{\alpha}^*(s) \,ds] \cdot e_1
\end{equation}
\begin{equation}
    ||C|| \leq 2\delta = O(\delta).
\end{equation}
Assuming that the magnitude of the average flow velocity along $e_1$ is approximately $\gamma_AA$ ($\gamma_A > 0$), and the width of the cell along $e_1$ is w \\
When $\bm{\alpha}$ is in the same direction as $\bm{V}$, the time it takes to cross the cell is
\begin{equation}
    t_{same} \approx \frac{w}{\gamma_AA + O(1)}.
\end{equation}
When $\bm{\alpha}$ is in the opposite direction as $\bm{V}$, the time it takes to cross the cell is
\begin{equation}
    t_{opp} \approx \frac{w}{\gamma_AA - O(1)}.
\end{equation}
The time difference is
\begin{equation}
    \Delta t = t_{opp} - t_{same} = \frac{2wO(1)}{(\gamma_AA)^2 - O(1)^2} = O(\frac{1}{A^2}).
\end{equation}
Because $A$ is sufficiently large, even if the control acts in the opposite direction to resist, the particle will still be carried away by the strong flow, which makes 
it necessary for the particle to reach the downstream region. \\
the gain in the downstream region is
\begin{equation}
    Gain_{downstream} \approx \gamma_A A \Delta t = O(\frac{1}{A}).
\end{equation}
\begin{equation}
    D \geq Gain_{downstream} - K\delta \Rightarrow C + D \geq Gain_{downstream} - (2+K)\delta.
\end{equation}
Given $A$, there always exists a proper $\delta$ such that
\begin{equation}
    C + D \geq Gain_{downstream} - (2+K)\delta > 0.
\end{equation}

Accordingly, during training, we can start adjusting the action from the tangential direction. In this work, we adopt the \textbf{Pre-filled Experience Replay Buffer} 
approach. The core idea of this method is to pre-generate high-quality datas using the environment's physical model and store them in the experience replay buffer 
before the agent performs any policy updates. It can effectively solve the problems of low sample efficiency and slow convergence caused by random exploration at the 
beginning of training (see Figure 5). Specifically, we perform the following steps before training.
\begin{itemize}
\item \textbf{State space sampling}: We perform uniform grid-based sampling within the state space that the agent may access. In the case of the two-dimensional, 
the state space is discretized into an $N \times N$ grid of points to ensure complete coverage of the initial exploration region.
\item \textbf{Action generation}: For each sampled state $s_t = (x,y)$, instead of using a random policy or the actor network under training to select an action, we directly use 
the tangential direction as the action (e.g., $v_{flow}(x,y) = 4 \cdot \langle -\sin{(2\pi x)}\cos{(2\pi y)},\cos{(2\pi x)} \sin{(2\pi y)} \rangle, \quad a_t = \frac
{v_{flow}}{||v_{flow}||}$).
\item \textbf{Pre-filled data}: For each state-action pair $(s_t,a_t)$, we input it into the environment model to obtain the next state $s_{t+1}$ and the immediate 
reward $r_t$, forming a tuple $(s_t,a_t,r_t,s_{t+1})$. Large number of such experiences are pre-filled into the experience replay buffer, so that it already contains 
rich samples at the beginning of training.
\end{itemize}
Although the Pre-filled Experience Replay Buffer approach can significantly accelerate convergence in the early stages of training, its performance does not increase 
monotonically with the amount of pre-filled data. Over-prefilling may lead to several negative effects, mainly because it breaks the exploration–exploitation balance in 
reinforcement learning.
\begin{figure}[ht]
    \centering
    \begin{subfigure}{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{pre_3.png}
        \captionsetup{labelformat=empty}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{pre_4.png}
        \captionsetup{labelformat=empty}
    \end{subfigure}
    \captionsetup{labelformat=empty}
    \caption{Figure 5: Given $\bm{x}_0 = (0.25, 0.25)$, and $A=16$. The left figure shows the result without using the Pre-filled 
    Experience Replay Buffer approach, where convergence begins after 1000 episodes. In contrast, the right figure uses the Pre-filled Experience Replay 
    Buffer approach, and we can observe a significantly faster convergence.}
\end{figure}

\bigskip

In Figure 6, 
given $\bm{x}_0 = (0.25, 0.25)$, $T = 1$ and $\Delta t = 0.0025$, we present the optimal trajectory 
$\bm{x}^*(t)$ of the reinforcement learning. In the beginning the action is likely perpendicular 
to the trajectory so that the particle may escape from the center of the cell. Afterwards the particle travels in $x$-direction 
along the edges of the cells.

\bigskip

In Figure 7, we compare the trajectories with respect to different intensities. If the flow intensity increases , 
then the trajectories are closer to edge of cells. We also see that all trajectories have the same escaping time from the center 
of the cells.

\bigskip

In Figure 8, we present the turbulent flow speed given. We see that our results agree with the results from 
numerical simulation of G-equation.

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{cell_1.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 6: The optimal trajectory given $\bm{x}_0 = (0.25, 0.25)$, $T = 1$, $\Delta t = 0.0025$, and $A=16$.}
\end{figure}

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{cell_4_1.png} % 圖片名稱
    \captionsetup{labelformat=empty}
\end{figure}
\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{cell_4_2.png} % 圖片名稱
    \captionsetup{labelformat=empty}
\end{figure}
\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{cell_4_3.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 7: The trajectories for different intensities: $A = 4$, $A = 8$ and $A = 16$}
\end{figure}
\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{cell_5.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 8: The turbulent speed corresponding to different $A$ values}
\end{figure} 

\newpage
\clearpage
\section{Rayleigh-Benard advection}
We consider the optimization problem (1.1) for Rayleigh-Benard advection in $2$-dimensional space. In Figure 9,
given $\bm{x}_0 = (0, 0)$, $T = 5$ and $\Delta t = 0.0125$, we present the optimal trajectory 
$\bm{x}^*(t)$

\bigskip

Figure 10 presents the turbulent flow speed. We observe that our training results are completely different from the numerical 
results. According to our results, variations in omega do not significantly affect the final x-value of the optimal trajectory 
(there might be some effect, but it is not significant; the fluctuations seen in the figure are mainly due to errors in 
reinforcement learning).

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{RB_1.png} % 圖片名稱
    \captionsetup{labelformat=empty}
\end{figure}
\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{RB_2.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 9: The optimal trajectory given $\bm{x}_0 = (0, 0)$, $T = 5$, $\Delta t = 0.0125$, $A = 4$, and $\omega = 3$.}
\end{figure}

\newpage

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=0.8\textwidth]{RB_4.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 10: The turbulent speed corresponding to different $\omega$ values and $A = 4$. Red indicates training results, 
    and blue indicates numerical results.}
\end{figure}

\section{the time periodically shifted cellular flow}
We consider the optimization problem (1.1) for the time periodically shifted cellular flow in $2$-dimensional space. In Figure 11,
given $\bm{x}_0 = (0, 0)$, $T = 5$ and $\Delta t = 0.0125$, we present the optimal trajectory 
$\bm{x}^*(t)$

\bigskip

Figure 12 presents the turbulent flow speed. We observe that our training results are completely different from the numerical 
results. According to our results, variations in omega do not significantly affect the final x-value of the optimal trajectory 
(there might be some effect, but it is not significant; the fluctuations seen in the figure are mainly due to errors in 
reinforcement learning).

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{RB_shifted_1.png} 
    \captionsetup{labelformat=empty}
\end{figure}
\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{RB_shifted_2.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 11: The optimal trajectory given $\bm{x}_0 = (0, 0)$, $T = 5$, $\Delta t = 0.0125$, $A = 4$, $B = 1$, and $\omega = 3$.}
\end{figure}

\newpage

\begin{figure}[h] % 圖片環境
    \centering
    \includegraphics[width=1.0\textwidth]{RB_shifted_4.png} % 圖片名稱
    \captionsetup{labelformat=empty}
    \caption{Figure 12: The turbulent speed corresponding to different $\omega$ values and $A = 4$, $B = 1$. Red indicates training results, 
    and blue indicates numerical results.}
\end{figure}


\chapter{Conclusion}

In this work, we formulated a trajectory optimization problem governed by a flow field and a bounded control, and applied reinforcement learning to approximate its 
optimal solution. By adopting the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, we successfully trained an agent capable of discovering sub-optimal 
control strategies under various flow configurations, including the cellular flow, Rayleigh–Bénard advection, and the time-periodically shifted cellular flow.

\bigskip

For cellular flows, the reinforcement learning results are close to the theoretical and numerical results of the G-equation. However, the number of loops the escaping 
cells make is greater than that obtained using traditional methods (such as the stochastic gradient descent method []). For time-dependent flows, traditional numerical 
methods fail to produce effective results, which is why reinforcement learning was considered for training. Fortunately, reinforcement learning can yield effective 
results. However, the trained results differ completely from the theoretical predictions of the G-equation. Further investigation will be conducted to determine whether 
there are any flaws in the G-equation theory.

\bigskip

This study demonstrates that reinforcement learning provides a promising, data-driven approach for solving high-dimensional optimal-control problems and the 
time-dependent flowsthat are otherwise difficult to handle using classical numerical methods. Future work will focus on improving the reward design, algorithm and
neural-network architecture to enhance stability and accuracy, as well as extending the approach to three-dimensional or Other flows.

\end{document}
