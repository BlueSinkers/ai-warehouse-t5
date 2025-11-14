
import random

# 5x5 grid from GridWorld
class GridWorld:


    #Legend: 'S' = Start, 'G' = Goal, 'X' = Wall, 'P' = Pit/Hole, ' ' = Empty cell
    
    # Everything starting with self is part of the grid
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols

        self.grid = [
            ['S', ' ', ' ', ' ', ' '],
            [' ', 'X', ' ', 'P', ' '],
            [' ', ' ', ' ', 'X', ' '],
            [' ', 'P', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', 'G'],
        ]

        # Find the start and goal positions on the grid
        self.start_pos = self._find_cell('S')
        self.goal_pos = self._find_cell('G')

        # Reward values
        self.step_reward = -1.0    # penalty for each move
        self.goal_reward = 10.0    # reward for reaching the goal
        self.pit_reward = -10.0    # penalty for falling into a pit
        self.wall_penalty = -1.0   # penalty for trying to move into a wall

        # Actions: 0 = up, 1 = right, 2 = down, 3 = left
        # state = where the agent is right now
        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.n_states = self.rows * self.cols

        # Current state is stored as a single integer index
        # the current place where the player starts is saved as one single number.
        self.state = self.pos_to_state(self.start_pos)

    # a helper to find where S or G is in the grid.
    def _find_cell(self, cell_type):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == cell_type:
                    return (r, c)
        raise ValueError(f"Cell type '{cell_type}' not found in grid.")

    # convert (row, col) → single integer
    def pos_to_state(self, pos):
        r, c = pos
        return r * self.cols + c

    # convert integer state back → (row, col)
    def state_to_pos(self, state):
        r = state // self.cols
        c = state % self.cols
        return (r, c)

    # Note to self: use (row, col) when dealing with the grid but use state integers when indexing the Q-table

    # Sets the current state to the start square. Called at the start of every episode. That’s how the agent “respawns” at the beginning.
    def reset(self):
        self.state = self.pos_to_state(self.start_pos)
        return self.state

    # taking an action and getting reward
    def step(self, action):

        # gets current position
        r, c = self.state_to_pos(self.state)

        # decide where to move
        if action == 0:      # up
            new_r, new_c = r - 1, c
        elif action == 1:    # right
            new_r, new_c = r, c + 1
        elif action == 2:    # down
            new_r, new_c = r + 1, c
        elif action == 3:    # left
            new_r, new_c = r, c - 1
        else:
            raise ValueError("Invalid action (must be 0, 1, 2, or 3).")

        # check if move is outside the grid, if it is, stay in place and give a penalty
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            reward = self.wall_penalty
            next_state = self.state
            done = False
            return next_state, reward, done, {}

        cell = self.grid[new_r][new_c]

        # check if move is into a wall, if it is, stay in place and give a penalty
        if cell == 'X':
            reward = self.wall_penalty
            next_state = self.state
            done = False
            return next_state, reward, done, {}

        # move to the new cell
        self.state = self.pos_to_state((new_r, new_c))
        next_state = self.state

        # output for the new cell
        # state + action → next_state, reward, done
        if cell == 'G':
            reward = self.goal_reward
            done = True
        elif cell == 'P':
            reward = self.pit_reward
            done = True
        else:
            reward = self.step_reward
            done = False

        return next_state, reward, done, {}

    # converts the list of state indices in path_states to (row, col) positions and draws * on those cells. Visualizes the path taken by the agent.
    def render(self, path_states=None):

        path_positions = set()
        if path_states is not None:
            for s in path_states:
                path_positions.add(self.state_to_pos(s))

        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                pos = (r, c)
                cell = self.grid[r][c]

                # If this cell is in the path and is not S, G, X, or P,
                # we show it as '*'
                if pos in path_positions and cell == ' ':
                    row_cells.append('*')
                else:
                    row_cells.append(cell)
            print(" ".join(row_cells))
        print()  # blank line after the grid


# loops through the list and finds the index of the largest value, this chooses the action with the highest Q-value in that state
def argmax(values):

    max_index = 0
    max_value = values[0]
    for i, v in enumerate(values):
        if v > max_value:
            max_value = v
            max_index = i
    return max_index


# Q-Learning Algorithm

def q_learning_train(
    env,
    num_episodes=500,
    max_steps_per_episode=100,
    alpha=0.1,      # learning rate
    gamma=0.99,     # discount factor
    epsilon=1.0,            # exploration probability
    epsilon_min=0.01,
    epsilon_decay=0.995
):

    # initialize Q-table with zeros
    Q = [[0.0 for _ in range(env.n_actions)] for _ in range(env.n_states)]
    episode_rewards = []

    # each episode = one full run from start until goal/pit or step limit
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0


        for step in range(max_steps_per_episode):
            # Epsilon-greedy action selection:
            # with probability epsilon, choose a random action (explore)
            # or choose the best known action (exploit)
            if random.random() < epsilon:
                action = random.choice(env.action_space)
            else:
                action = argmax(Q[state])

            # Take the action in the environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Bellman Equation:
            # Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s,a)]
            old_value = Q[state][action]
            next_max = max(Q[next_state])  # best action value in the next state

            new_value = old_value + alpha * (
                reward + gamma * next_max - old_value
            )
            Q[state][action] = new_value

            state = next_state
        # the agent “moves” in its head to the new state. If goal or pit, done is True and we break out of the inner loop
            if done:
                break

        # Save the total reward for this episode
        episode_rewards.append(total_reward)

        # Decay epsilon so we explore less over time. So early on: lots of exploration. Later: mostly exploitation

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return Q, episode_rewards



def get_greedy_path(env, Q, max_steps=50):

    state = env.reset()
    path = [state]

    for _ in range(max_steps):
        # chooses the best action according to Q (no exploration)
        action = argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    return path

# putting it all together
def main():
    # create the gridworld environment
    env = GridWorld()

    # train Q-learning on this environment
    Q, episode_rewards = q_learning_train(env)

    # show the learned greedy path
    path = get_greedy_path(env, Q)
    print("Greedy path (state indices):")
    print(path)
    print()

    print("Grid with learned path shown as '*':")
    env.render(path_states=path)

if __name__ == "__main__":
    main()
