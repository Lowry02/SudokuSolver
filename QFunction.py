import torch
import math
import torch.nn as nn
from typing import Callable

class QConv(nn.Module):
  def __init__(self, head_dim:int, one_hot_dim:int, input_dim:int) -> None:
    super().__init__()
    self.head_dim = head_dim  # number of classifier's components
    self.one_hot_dim = one_hot_dim
    self.input_dim = input_dim  # the input is considered flattened
    self.matrix_side = int(math.sqrt(input_dim))
    needs_padding = int(self.matrix_side - 6 == 0)  # the matrix side dimension decreases of 6 during the convolution
    output_side = (self.matrix_side + 2 * needs_padding) - 6
    
    self.register_buffer("one_hot_encoding", torch.eye(one_hot_dim))
    self.conv = nn.Sequential(
      nn.Conv2d(one_hot_dim, 32, 3, 1, needs_padding),
      nn.ReLU(),
      nn.Conv2d(32, 16, 3, 1, 0),
      nn.ReLU(),
      nn.Conv2d(16, 8, 3, 1, 0),
      nn.ReLU()
    )
    
    self.flattened_shape = (-1, 8*output_side*output_side)
    self.head = nn.Sequential(
      nn.Linear(self.flattened_shape[1], 258),
      nn.ReLU(),
      nn.Linear(258, head_dim)
    )

  def forward(self, x:torch.Tensor, legal_actions:torch.Tensor):
    # x shape: (batch_size, 81)
    x = x.long()
    x = self.one_hot_encoding[x]  # type: ignore
    x = x.view(x.size(0), self.one_hot_dim, self.matrix_side, self.matrix_side)
    x = self.conv(x)
    x = x.reshape(self.flattened_shape)
    x = self.head(x)
    x[legal_actions == 0] = -1000 # setting illegal actions to -1000
    return x
    
class QMoE(nn.Module):
  def __init__(self,
    head_dim: int,
    one_hot_dim: int,
    input_dim: int,
    n_experts: int,
    gate_function: Callable,
  ) -> None:
    super().__init__()
    
    self.n_experts = n_experts
    self.gate_function = gate_function  # gate_function is a function
    self.head_dim = head_dim
    self.one_hot_dim = one_hot_dim
    self.input_dim = input_dim
    self.n_experts = n_experts
    self.experts = nn.ModuleList([QConv(head_dim, one_hot_dim, input_dim) for _ in range(n_experts)])
    
  def forward(self, x:torch.Tensor, legal_actions:torch.Tensor) -> torch.Tensor:
    expert1, expert2, expert1_idx, expert2_idx = self.gate_function(x)
    out = torch.zeros((x.shape[0], self.head_dim))
    out[expert1_idx] = self.experts[expert1](x[expert1_idx], legal_actions[expert1_idx])
    if expert2 != -1:
      out[expert2_idx] = self.experts[expert2](x[expert2_idx], legal_actions[expert2_idx])
    return out