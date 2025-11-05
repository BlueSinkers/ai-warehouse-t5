import numpy as np
import random

def q_learning_path(grid, start_pos, goal_pos, alpha=0.1, gamma=0.9, epsilon=0.2, episodes=5000, max_steps=50, allow_diagonals=False):
    # reward constants (can adjust if needed)
    s = -20      # start
    e = -1     # empty
    w = -100   # wall
    g = 10     # goal

    n_rows = len(grid)
    n_cols = len(grid[0])

    # movement options
    base_actions = ["up", "down", "left", "right"]
    diag_actions = ["up-left", "up-right", "down-left", "down-right"]
    actions = base_actions + diag_actions if allow_diagonals else base_actions

    # initialize qtable
    Q = np.zeros((n_rows, n_cols, len(actions)))

    # check if pos is valid
    def is_valid(pos):
        row, col = pos
        return 0 <= row < n_rows and 0 <= col < n_cols and grid[row][col] != w

    # find next pos based on action
    def next_position(pos, action):
        row, col = pos
        if action == "up":
            row -= 1
        elif action == "down":
            row += 1
        elif action == "left":
            col -= 1
        elif action == "right":
            col += 1
        elif action == "up-left":
            row -= 1; col -= 1
        elif action == "up-right":
            row -= 1; col += 1
        elif action == "down-left":
            row += 1; col -= 1
        elif action == "down-right":
            row += 1; col += 1
        return (row, col) if is_valid((row, col)) else pos  # stay if invalid

    # choose action
    def choose_action(pos):
        if random.uniform(0,1) < epsilon:
            return random.choice(range(len(actions)))  # explore
        row, col = pos
        return np.argmax(Q[row, col, :])  # exploit

    # q-learning loop
    for episode in range(episodes):
        pos = start_pos
        for step in range(max_steps):
            row, col = pos
            action_idx = choose_action(pos)
            action = actions[action_idx]

            new_pos = next_position(pos, action)
            new_row, new_col = new_pos

            reward = grid[new_row][new_col]

            # qlearning update
            Q[row, col, action_idx] = Q[row, col, action_idx] + alpha * (
                reward + gamma * np.max(Q[new_row, new_col, :]) - Q[row, col, action_idx]
            )

            pos = new_pos
            if pos == goal_pos:
                break

    # extract optimal path
    pos = start_pos
    path = [pos]
    for _ in range(max_steps):
        row, col = pos
        action_idx = np.argmax(Q[row, col, :])
        pos = next_position(pos, actions[action_idx])
        path.append(pos)
        if pos == goal_pos:
            break

    return path

