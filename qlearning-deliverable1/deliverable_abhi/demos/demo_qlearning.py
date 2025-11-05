import numpy as np
import random


# shorter single-letter reward constants
s = 0      # start
e = -1     # empty
w = -100   # wall
g = 10     # goal

grid = [
    [e, e, e, e, e],
    [e, w, e, w, e],
    [e, e, e, e, e],
    [e, w, e, e, e],
    [e, e, e, w, e]
]

n_rows = len(grid)
n_cols = len(grid[0])

# movement options: allow enabling diagonal moves by setting allow_diagonals True
base_actions = ["up", "down", "left", "right"]
diag_actions = ["up-left", "up-right", "down-left", "down-right"]
allow_diagonals = False  # set to True to enable diagonal movement
actions = base_actions + diag_actions if allow_diagonals else base_actions

# defining hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration probability
episodes = 5000
max_steps = 50

#initialize qtable
Q = np.zeros((n_rows, n_cols, len(actions)))

#checks if the position is valid    
def is_valid(pos):
    row, col = pos
    return 0 <= row < n_rows and 0 <= col < n_cols and grid[row][col] != w

#finds the next 
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
        row -= 1
        col -= 1
    elif action == "up-right":
        row -= 1
        col += 1
    elif action == "down-left":
        row += 1
        col -= 1
    elif action == "down-right":
        row += 1
        col += 1
    return (row, col) if is_valid((row, col)) else pos  # stay in place if invalid

def choose_action(pos):
    if random.uniform(0,1) < epsilon:
        return random.choice(range(len(actions)))  # explore
    row, col = pos
    return np.argmax(Q[row, col, :])  # exploit

# q-learning algorithm picks starting and ending
start_pos = (0,0)
goal_pos = (2,4)

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
for _ in range(50):
    row, col = pos
    action_idx = np.argmax(Q[row, col, :])
    pos = next_position(pos, actions[action_idx])
    path.append(pos)
    if pos == goal_pos:
        break

# print on path
print("Optimal path (row, col):")
print(path)

# visualization
print("\nGrid with path (P = path, G = goal, W = wall, 0 = empty):")
visual_grid = [["0" for _ in range(n_cols)] for _ in range(n_rows)]
for r, c in path:
    visual_grid[r][c] = "P"
visual_grid[goal_pos[0]][goal_pos[1]] = "G"
for r in range(n_rows):
    for c in range(n_cols):
        if grid[r][c] == w:
            visual_grid[r][c] = "W"
for row in visual_grid:
    print(" ".join(row))
