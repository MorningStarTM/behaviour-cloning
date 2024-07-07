import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


class MLPBehaviorCloning(nn.Module):
    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, state_size, action_size, hidden_size, n_layer, dropout=0.1, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size

        layers = [nn.Linear(state_size, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)


    def forward(self, states):
        actions = self.model(states)
        return actions


    def forward_loss(self, states, actions):
        action_preds = self.forward(states)

        loss = torch.mean((action_preds - actions) ** 2)
        return loss
