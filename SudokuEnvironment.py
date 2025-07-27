import random
import torch
from typing import Callable
class SudokuEnvironment:
  """
  A general Sudoku environment for N x N grids with M x K sub-boxes.
  Supports grid creation, validation, solving, random puzzle generation, and RL-style step/reward.
  """

  def __init__(self, with_legal_actions:bool=True, grid=None, size=9, box_height=None, box_width=None):
    # WRITTEN BY CHATGPT
    """
    Initialize the Sudoku environment.

    Args:
      grid (list of lists or torch.Tensor, optional): Initial grid values.
      size (int): Grid dimension (e.g. 9 for 9x9, 6 for 6x6).
      box_height (int, optional): Number of rows per sub-box. Defaults to int(sqrt(size)).
      box_width (int, optional): Number of cols per sub-box. Defaults to size//box_height.
    """
    self.size = size
    self.with_legal_actions = with_legal_actions
    self.get_reward:Callable = self._sparse_reward
    # Determine sub-box dimensions
    if box_height is None or box_width is None:
      # For perfect squares, default to sqrt-size boxes
      root = int(size**0.5)
      if root * root == size:
        self.box_height = self.box_width = root
      else:
        assert box_height and box_width, (
          "For non-square sub-boxes, both box_height and box_width must be provided."
        )
        self.box_height, self.box_width = box_height, box_width
    else:
      self.box_height, self.box_width = box_height, box_width

    num_cells = self.size * self.size
    # Load or initialize grid
    if isinstance(grid, torch.Tensor):
      assert grid.numel() == num_cells, f"Tensor must have {num_cells} elements."
      self.grid = grid.clone().to(dtype=torch.int)
    elif grid is not None:
      flat = [v for row in grid for v in row]
      assert len(flat) == num_cells, f"Grid must be {size}x{size}."
      self.grid = torch.tensor(flat, dtype=torch.int)
    else:
      self.grid = torch.zeros(num_cells, dtype=torch.int)

    self.starting_grid = self.grid.clone()

  def load_from_tensor(self, tensor):
    # WRITTEN BY CHATGPT
    assert isinstance(tensor, torch.Tensor), "Input must be a torch.Tensor."
    assert tensor.numel() == self.size * self.size, \
      f"Tensor must have {self.size*self.size} elements."
    self.grid = tensor.clone().to(dtype=torch.int)

  def reset(self):
    """Reset to the starting grid."""
    self.grid = self.starting_grid.clone()

  # ----- Puzzle generation -----
  def create_new(self):
    # WRITTEN BY CHATGPT
    """
    Generate a completely filled, valid Sudoku.
    """
    self.grid.zero_()
    self._fill_grid()
    self.starting_grid = self.grid.clone()
    return self.grid

  def _fill_grid(self):
    # WRITTEN BY CHATGPT
    for idx in range(self.size * self.size):
      if self.grid[idx] == 0:
        row, col = divmod(idx, self.size)
        nums = list(range(1, self.size + 1))
        random.shuffle(nums)
        for num in nums:
          if self._is_valid_cell(self.grid, row, col, num):
            self.grid[idx] = num
            if self._fill_grid():
              return True
            self.grid[idx] = 0
        return False
    return True

  def add_remove_cells(self, num_cells, remove=True):
    # WRITTEN BY CHATGPT
    """
    Remove or add clues to create a puzzle with a unique solution.
    """
    assert num_cells >= 0 and isinstance(num_cells, int)
    total = self.size * self.size
    if remove:
      indices = [i for i in range(total) if self.grid[i] != 0]
      random.shuffle(indices)
      removed = 0
      for idx in indices:
        backup = self.grid[idx].item()
        self.grid[idx] = 0
        sols = []
        self._count_solutions(self.grid.clone(), sols)
        if len(sols) == 1:
          removed += 1
        else:
          self.grid[idx] = backup
        if removed >= num_cells:
          break
    else:
      indices = [i for i in range(total) if self.grid[i] == 0]
      random.shuffle(indices)
      added = 0
      for idx in indices:
        row, col = divmod(idx, self.size)
        nums = list(range(1, self.size + 1))
        random.shuffle(nums)
        for num in nums:
          if self._is_valid_cell(self.grid, row, col, num):
            self.grid[idx] = num
            added += 1
            break
        if added >= num_cells:
          break
    return self.grid

  # ----- Solving and validation -----
  def _count_solutions(self, grid, solutions, limit=2):
    # WRITTEN BY CHATGPT
    for idx in range(self.size * self.size):
      if grid[idx] == 0:
        row, col = divmod(idx, self.size)
        for num in range(1, self.size + 1):
          if self._is_valid_cell(grid, row, col, num):
            grid[idx] = num
            self._count_solutions(grid, solutions, limit)
            grid[idx] = 0
        return
    solutions.append(grid.clone())
    if len(solutions) >= limit:
      return

  def solve(self):
    # WRITTEN BY CHATGPT
    """Backtracking solver."""
    return self._solve_grid()

  def _solve_grid(self):
    # WRITTEN BY CHATGPT
    for idx in range(self.size * self.size):
      if self.grid[idx] == 0:
        row, col = divmod(idx, self.size)
        for num in range(1, self.size + 1):
          if self._is_valid_cell(self.grid, row, col, num):
            self.grid[idx] = num
            if self._solve_grid():
              return True
            self.grid[idx] = 0
        return False
    return True

  def check_insertion(self, row, col, num):
    # WRITTEN BY CHATGPT
    """True if inserting num at (row,col) is valid."""
    assert 0 <= row < self.size and 0 <= col < self.size
    assert 1 <= num <= self.size
    return self._is_valid_cell(self.grid, row, col, num)

  def _is_valid_cell(self, grid, row, col, num):
    # WRITTEN BY CHATGPT
    """Check row, column, and sub-box constraints."""
    if grid[row*self.size + col] != 0:
      return False
    for i in range(self.size):
      if grid[row*self.size + i] == num or grid[i*self.size + col] == num:
        return False
    br = row - row % self.box_height
    bc = col - col % self.box_width
    for dr in range(self.box_height):
      for dc in range(self.box_width):
        idx = (br+dr)*self.size + (bc+dc)
        if grid[idx] == num:
          return False
    return True

  # ----- Action & reward API -----
  def get_legal_actions(self):
    # WRITTEN BY CHATGPT
    """Returns a flat tensor [size*size*size] of legal 0/1 entries."""
    if self.with_legal_actions: 
      total = self.size * self.size * self.size
      legal = torch.zeros(total, dtype=torch.int)
      for a in range(total):
        cell = a // self.size
        r, c = divmod(cell, self.size)
        num = (a % self.size) + 1
        if self._is_valid_cell(self.grid, r, c, num):
          legal[a] = 1
      return legal
    return torch.ones(self.size * self.size * self.size)

  def _sparse_reward(self):
    leg = self.get_legal_actions().view(-1, self.size)
    at_least_one_action_per_cell = (leg.sum(dim=1) > 0).sum().item()
    n_empty_cells = int((self.grid == 0).sum().item())
    lose = (at_least_one_action_per_cell != n_empty_cells)
    win = (at_least_one_action_per_cell == 0 and not lose)
    return win * 100 + lose * -100, (win or lose), win
  
  def _dense_reward(self):
    leg = self.get_legal_actions().view(-1, self.size)
    at_least_one_action_per_cell = (leg.sum(dim=1) > 0).sum().item()
    n_empty_cells = int((self.grid == 0).sum().item())
    lose = (at_least_one_action_per_cell != n_empty_cells)
    win = (at_least_one_action_per_cell == 0 and not lose)
    completed = (win or lose)
    return win * 100 + lose * -100 + (not completed), (win or lose), win

  def step(self, action):
    cell = action // self.size
    row, col = divmod(cell, self.size)
    num = (action % self.size) + 1
    if self.check_insertion(row, col, num) or self.with_legal_actions:
      self.grid[cell] = num
      reward, done, win = self.get_reward()
    else:
      reward, done, win = -100, True, False
    return self.grid, reward, done, win, self.get_legal_actions()
  
  def set_sparse_reward(self):
    self.get_reward = self._sparse_reward
    
  def set_dense_reward(self):
    self.get_reward = self._dense_reward

  def __str__(self):
    # WRITTEN BY CHATGPT
    mat = self.grid.view(self.size, self.size)
    lines = []
    for row in mat:
      lines.append(" ".join(str(int(v)) if v!=0 else '.' for v in row))
    return "\n".join(lines)
