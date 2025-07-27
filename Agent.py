import torch
import random
from QFunction import QConv, QMoE
from collections import deque, namedtuple

class Agent():
  def __init__(self, n_actions:int, n_values:int, input_dim:int, train_mode:bool=False, device:str='cpu') -> None:
    """_summary_

    Args:
        n_actions (int, optional): Number of actions the agent can perform. Defaults to 729.
        n_values (int, optional): Feasible number of values in the sudoku grid. Defaults to 10.
        train_mode (bool, optional): Mainly used to disable epsilon-greedy policy and enable the deterministic one. Defaults to False.
    """
    self.device = device
    self.buffer = deque(maxlen=10000)
    
    self.behaviour_net = QConv(head_dim=n_actions, one_hot_dim=n_values, input_dim=input_dim).to(device)
    self.target_net = QConv(head_dim=n_actions, one_hot_dim=n_values, input_dim=input_dim).to(device)
    self.target_net.load_state_dict(self.behaviour_net.state_dict())
    self.train_performed = 0
    self.network_delay = 1000 # after network_delay train process, copy the behaviour net into the target net

    self.loss_fn = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.behaviour_net.parameters(), lr=1e-3)
    
    self.train_mode = train_mode
    self.n_actions = 729
    self.original_epsilon = 0.99
    self.epsilon = self.original_epsilon
    self.gamma = 1.
    self.epsilon_decay = 0.97
    self.epsilon_clip_value = 0.05
    self.steps_taken = 0
    self.epsilon_update = 500 # apply epsilon decay after epsilon_update steps
    self.batch_size = 64 # batch size to train the model
    
    self.Experience = namedtuple('Experience', ['state', 'legal_actions', 'action', 'reward', 'next_state', 'next_legal_actions', 'done']) # type used to save epxerience inside buffer
    
  def enable_train_mode(self):
    self.train_mode = True
    
  def disable_train_mode(self):
    self.train_mode = False
    
  def get_q_values(self, state:torch.Tensor, legal_actions:torch.Tensor) -> torch.Tensor:
    self.behaviour_net.eval()
    with torch.no_grad():
      return self.behaviour_net(state.to(self.device), legal_actions.to(self.device)).to('cpu')
    
  def take_action(self, state:torch.Tensor, legal_actions:torch.Tensor) -> int:
    # implementing epsilon_greedy
    # updating epsilon if needed
    
    self.steps_taken += 1
    needs_update = (self.steps_taken % self.epsilon_update) == 0
    self.epsilon = torch.clip(torch.Tensor([needs_update * self.epsilon * self.epsilon_decay + (1 - needs_update) * self.epsilon]), self.epsilon_clip_value, 1.).item()
    self.steps_taken = (1 - needs_update) * self.steps_taken
    
    legal_actions = legal_actions.unsqueeze(0)
    q_values = self.get_q_values(state, legal_actions).squeeze(0)
    valid_q_values_idx = (q_values != -1000).nonzero(as_tuple=True)[0]
    perform_random_action = self.train_mode and (torch.rand((1, )) < self.epsilon).item()
    if perform_random_action:
      return int(valid_q_values_idx[torch.randint(0, len(valid_q_values_idx), (1,))].item())
    else: 
      return int(valid_q_values_idx[torch.argmax(q_values[valid_q_values_idx])])
    
  def save_experience(self, state, legal_actions, action, reward, next_state, next_legal_actions, done):
    self.buffer.append(self.Experience(state, legal_actions, action, reward, next_state, next_legal_actions, done))
    
  def learn(self) -> float | None:
    if len(self.buffer) < self.batch_size:
      return None
    
    batch = random.sample(self.buffer, self.batch_size)
    states = torch.stack([experience.state for experience in batch])
    legal_actions = torch.stack([experience.legal_actions for experience in batch])
    actions = torch.tensor([experience.action for experience in batch])
    rewards = torch.tensor([experience.reward for experience in batch])
    next_states = torch.stack([experience.next_state for experience in batch])
    next_legal_actions = torch.stack([experience.next_legal_actions for experience in batch])
    dones = torch.tensor([experience.done for experience in batch]).to(torch.float)
    
    current_q = self.get_q_values(states, legal_actions)
    self.target_net.eval()
    with torch.no_grad():
      next_q = self.target_net.forward(next_states.to(self.device), next_legal_actions.to(self.device)).to('cpu')
    target_q = current_q.clone()
    
    target_q[torch.arange(len(batch)), actions] = (1 - dones)*(rewards + self.gamma * torch.max(next_q, dim=1).values) + dones * rewards
    return self.train(states, target_q, legal_actions)
    
  def train(self, states:torch.Tensor, q_values:torch.Tensor, legal_actions:torch.Tensor):
    self.train_performed += 1
    self.behaviour_net.train()
    self.optimizer.zero_grad()
    predictions = self.behaviour_net.forward(states.to(self.device), legal_actions.to(self.device))
    loss = self.loss_fn(predictions, q_values.to(self.device))
    loss.backward()
    self.optimizer.step()
    
    if self.train_performed % self.network_delay == 0:
      # aligning behaviour and target net
      self.target_net.load_state_dict(self.behaviour_net.state_dict())
      self.train_performed = 0
    
    return loss.item()
    
  def reset_epsilon(self):
    self.epsilon = self.original_epsilon
    self.steps_taken = 0
    

class MoEAgent(Agent):
  def __init__(self, n_actions:int, n_values:int, input_dim:int, train_mode:bool = False, device:str = 'cpu') -> None:
    super().__init__(n_actions, n_values, input_dim, train_mode, device)
    self.n_experts = 12
    self.empty_cells_group = 3 # divide the difficulty of the sudoku through the agent based on the number of empty cells
    self.behaviour_net = QMoE(
      head_dim=n_actions,
      one_hot_dim=n_values,
      input_dim=input_dim,
      n_experts=self.n_experts,
      gate_function=self.gate_function
    )
    self.target_net = QMoE(
      head_dim=n_actions,
      one_hot_dim=n_values,
      input_dim=input_dim,
      n_experts=self.n_experts,
      gate_function=self.gate_function
    )
    self.target_net.load_state_dict(self.behaviour_net.state_dict())
    self.optimizer = torch.optim.Adam(self.behaviour_net.parameters(), lr=1e-3)
    
  def gate_function(self, x:torch.Tensor):
    empty_cells = (x == 0).sum(dim=1)
    expert_involved = empty_cells // self.empty_cells_group
    expert_ids = expert_involved.unique() # there can be max 2 experts
    if expert_ids.shape[0] == 1:
      return expert_ids[0], -1, expert_involved == expert_ids[0], expert_involved == -1
        
    return expert_ids[0], expert_ids[1], expert_involved == expert_ids[0], expert_involved == expert_ids[1]
  
  def clone_expert(self, expert_idx:int):
    # clone the weights of one expert into another one
    self.behaviour_net.experts[expert_idx + 1].load_state_dict(self.behaviour_net.experts[expert_idx].state_dict())
    self.optimizer = torch.optim.Adam(self.behaviour_net.parameters(), lr=1e-3)
    
  def align_target_behaviour_nets(self):
    self.target_net.load_state_dict(self.behaviour_net.state_dict())
    
  def empty_buffer(self):
    self.buffer.clear()