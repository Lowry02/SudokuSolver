(WRITTEN BY CHATGPT)
# Solving Sudoku with Reinforcement Learning

This project explores the use of Reinforcement Learning (RL) to solve Sudoku puzzles. It frames Sudoku as a sequential decision-making problem and applies Deep Q-Learning with a Replay Buffer to approximate optimal policies. The approach includes custom reward shaping, curriculum learning, and a Mixture of Experts architecture to enable efficient learning from smaller boards (6x6) to full Sudoku puzzles (9x9).

## Problem Statement

Sudoku is a constraint-based puzzle where the objective is to fill a grid such that each number appears exactly once in every row, column, and subgrid. Instead of solving this via brute force or constraint programming, I approach it through reinforcement learning, where an agent fills in the puzzle step-by-step, learning through trial and feedback.

## Reinforcement Learning Setup

Core Algorithm: Deep Q-Learning with experience replay.

Replay Buffer: Stores and samples past transitions to stabilize training.

Reward Shaping: Both sparse and dense rewards are designed to guide the learning process.

Curriculum Learning: The agent starts with fully completed grids and learns progressively as more cells are removed, increasing task difficulty.

## Deep Q-Learning with Mixture of Experts

To enhance the learning and generalization capability of the Q-network, I used a Mixture of Experts (MoE) in which each expert is specialized on a specific board difficulty.

## Experiments

Initial Testing: Conducted on 6x6 Sudoku boards to validate the RL pipeline and network behavior.

Final Evaluation: Scaled the system to standard 9x9 puzzles, evaluating generalization and solving capability.

