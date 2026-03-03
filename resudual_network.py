import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, dim):
        super(Residual, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(dim, 512)
        self.layer2 = nn.Linear(512, dim)

        self.norm = nn.LayerNorm(512)
        self.activate = nn.ReLU()
        
    def forward(self, y):
        x = self.activate(self.norm((self.layer1(y))))
        x = self.layer2(x)
        return self.activate(x+y)
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.output_layer = nn.Linear(256, action_dim)

        self.res = Residual(256)
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
        x = self.res(x)
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
        self.layer4 = nn.Linear(256, 1)
        
        # Q2網路
        self.layer8 = nn.Linear(state_dim + action_dim, 256)
        self.layer9 = nn.Linear(256, 256)
        self.layer11 = nn.Linear(256, 1)

        self.res = Residual(256)
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
        q1 = self.res(q1)
        q1 = self.activate(self.norm((self.layer2(q1))))
        q1 = self.res(q1)
        q1 = self.layer4(q1)
        
        # Q2前向傳播
        q2 = self.activate(self.norm((self.layer8(sa))))
        q2 = self.res(q2)
        q2 = self.activate(self.norm((self.layer9(q2))))
        q2 = self.res(q2)
        q2 = self.layer11(q2)
        return q1, q2
