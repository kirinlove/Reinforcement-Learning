# Finding Optimal Trajectories in Front Propagation Problem by Deep Reinforcement Learning 


### Reinforcement Learning Policy Performance in Unsteady Cellular Flow
<img width="350" height="711" alt="image" src="https://github.com/user-attachments/assets/93ab1a8b-c296-408e-828c-67f03d0dd498" />
<img width="500" height="711" alt="image" src="https://github.com/user-attachments/assets/14eb639e-7ff7-4b67-97be-e0450e3e6cdc" />

# Environment Setup
&#8226; Install [**Anaconda 24.11.3**](https://www.anaconda.com/products/distribution) for your operating system.  

&#8226; Anaconda Prompt:
```bash
conda install git
git clone https://github.com/kirinlove/Reinforcement-Learning.git
cd Reinforcement-Learning
conda create -n thesis-rl python=3.10 -y
conda activate thesis-rl
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy matplotlib scipy gym
```

# Training
&#8226; Cellular Flow
Anaconda Prompt:
```bash
conda activate thesis-rl
python Reinforcement-Learning/Cellular_flow/Cellular_flow_2.py
```

&#8226; Rayleigh-Benard Advection
Anaconda Prompt:
```bash
conda activate thesis-rl
python Reinforcement-Learning/Rayleigh-Benard_advection/Rayleigh-Benard_advection_4.py
```

&#8226; Unsteady Cellular Flow
Anaconda Prompt:
```bash
conda activate thesis-rl
python Reinforcement-Learning/Unsteady_Cellular_flow/Unsteady_Cellular_flow_3.py
```

# Testing
Anaconda Prompt:
```bash
conda activate thesis-rl
python Reinforcement-Learning/test.py
```
