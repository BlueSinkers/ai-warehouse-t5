import numpy as np
import time
import sys 

class GridEnvironment:
    def __init__(self, n_rows=5, n_cols=5, start_pos=[0,0], package_pos=[2,4], goal_pos=[4,0], max_steps=20, move_diaganols=True, obstacles=[], restricted_area=[]):
        self.start_pos = start_pos
        self.package_pos = package_pos
        self.goal_pos = goal_pos
        self.move_diaganols = move_diaganols
        self.obstacles = obstacles
        self.restricted_area = restricted_area

        boundary_checks = [start_pos, package_pos, goal_pos] + obstacles + restricted_area
        for pos in boundary_checks:
            if not (0 <= pos[0] < n_rows and 0 <= pos[1] < n_cols):
                raise ValueError(f"Position {pos} is out of bounds for grid size {n_rows}x{n_cols}")

        self.agent_pos = list(self.start_pos)
        self.has_package = False

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_package_status = 2 
        self.n_states = self.n_rows * self.n_cols * self.n_package_status
        if move_diaganols:
            self.n_actions = 8
        else:
            self.n_actions = 4

        self.move_penalty = -1 # Penalty for each move
        self.diaganol_move_penalty = -1.4 # Penalty for diagonal move (sqrt(2) approximately since thats the distance) 
        self.wall_penalty = -5 # Penalty for hitting a wall (trying to move out of bounds or going into an obstacle)
        self.restricted_area_penalty = -100 # Penalty for entering restricted area
        self.pickup_reward = 15 # Reward for picking up the package
        self.goal_reward = 30 # Reward for reaching goal with package

        self.max_steps = max_steps
        self.step_count = 0

    def get_state_index(self):
        if(self.has_package):
            item_status = 1
        else:
            item_status = 0
        state_index = (self.agent_pos[0] * self.n_cols * self.n_package_status) + (self.agent_pos[1] * self.n_package_status) + item_status
        return int(state_index)
        

    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.has_package = False
        self.step_count = 0
        return self.get_state_index(), {}
    
    def step(self, action):
        self.step_count += 1
        reward = self.move_penalty
        done = False

        if action == 0:  
            new_pos = [self.agent_pos[0] - 1, self.agent_pos[1]]
            reward = self.move_penalty
        elif action == 1:  
            new_pos = [self.agent_pos[0] + 1, self.agent_pos[1]]
            reward = self.move_penalty
        elif action == 2:  
            new_pos = [self.agent_pos[0], self.agent_pos[1] - 1]
            reward = self.move_penalty
        elif action == 3:  
            new_pos = [self.agent_pos[0], self.agent_pos[1] + 1]
            reward = self.move_penalty
        elif action == 4:
            new_pos = [self.agent_pos[0] - 1, self.agent_pos[1] - 1]
            reward = self.diaganol_move_penalty
        elif action == 5:
            new_pos = [self.agent_pos[0] - 1, self.agent_pos[1] + 1]
            reward = self.diaganol_move_penalty
        elif action == 6:
            new_pos = [self.agent_pos[0] + 1, self.agent_pos[1] - 1]
            reward = self.diaganol_move_penalty
        elif action == 7:
            new_pos = [self.agent_pos[0] + 1, self.agent_pos[1] + 1]
            reward = self.diaganol_move_penalty
        else:
            raise ValueError("Invalid action")


        # Add wall penalty if out of bounds
        if 0 <= new_pos[0] < self.n_rows and 0 <= new_pos[1] < self.n_cols:
            if(new_pos in self.obstacles):
                reward += self.wall_penalty
            else:
                if(new_pos in self.restricted_area):
                    reward += self.restricted_area_penalty
                else:
                    self.agent_pos = new_pos
        else:
            reward += self.wall_penalty

        # If agent found package
        if self.agent_pos == self.package_pos and not self.has_package:
            self.has_package = True
            reward += self.pickup_reward

        # If agent reached goal with package
        if self.agent_pos == self.goal_pos and self.has_package:
            reward += self.goal_reward
            done = True

        # If max steps reached
        if self.step_count >= self.max_steps:
            done = True

        return self.get_state_index(), reward, done, {}
    
    # Render the environment for visualization
    def render(self):
        for r in range(self.n_rows):
            row_str = "| "
            for c in range(self.n_cols):
                char = "."
                
                if [r, c] in self.obstacles:
                    char = "o" # Obstacle

                if [r, c] in self.restricted_area:
                    char = "X" # Restricted area
                # Check for item
                if [r, c] == self.package_pos and not self.has_package:
                    char = "P" # Package
                # Check for goal
                elif [r, c] == self.goal_pos:
                    char = "G" # Goal
                    
                # Check for agent
                if [r, c] == self.agent_pos:
                    char = "A" # Agent
                    if self.has_package:
                        char = "A+" # Agent with item
                        
                row_str += f"{char} "
            row_str += "|"
            print(row_str)
        print(f"Agent {self.agent_pos} | Item: {'Yes' if self.has_package else 'No'} | Steps: {self.step_count}/{self.max_steps}")

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, total_epochs=20000):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.total_epochs = total_epochs

        self.q_table = np.zeros((env.n_states, env.n_actions))

    # Exploring vs. exploitation:
    # High epsilon = more exploration (picks completely random action)
    # Low epsilon = more exploitation (picks best known action)
    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    # Building the Q-table
    # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
    def train_agent(self):
        for epoch in range(self.total_epochs):
            state, info = self.env.reset()
            done = False
            total_reward = 0

            # Lowers curr_epsilon each epoch to reduce exploration over time (exploit more when smarter)
            curr_epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * epoch)

            while not done:
                action = self.choose_action(state, curr_epsilon)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward

                # Target Q is the value we want to move towards. Max is for looking at the next state and finding Q-value of best possible action. The (1 - done) ensures no future reward is considered if game is over
                # Gamme makes future rewards less important. Higher gamma = more importance on future rewards. Lower gamma = more importance on immediate rewards
                target_q = reward + self.gamma * np.max(self.q_table[next_state, :]) * (1 - done)

                # Update Q-value. Alpha is the learning rate (how much new info overrides old info). Higher alpha = more new info. Lower alpha = more old info
                self.q_table[state, action] += self.alpha * (target_q - self.q_table[state, action])
                state = next_state

    def run_agent(self):
        state, info = self.env.reset()
        done = False
        step = 0

        while not done:
            step += 1
            action = self.choose_action(state, epsilon=0)
            new_state, reward, done, info = self.env.step(action)
            self.env.render()

            state = new_state
            time.sleep(0.5)

class SarsaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, total_epochs=20000):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.total_epochs = total_epochs

        self.q_table = np.zeros((env.n_states, env.n_actions))
    
    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
        
    def train_agent(self):
        for epoch in range(self.total_epochs):
            state, info = self.env.reset()
            done = False
            total_reward = 0

            curr_epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * epoch)
            action = self.choose_action(state, curr_epsilon)

            while not done:
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                next_action = self.choose_action(next_state, curr_epsilon)

                old_q = self.q_table[state, action]
                next_q = self.q_table[next_state, next_action]
                target_q = reward + self.gamma * next_q * (1 - done)
                self.q_table[state, action] = old_q + self.alpha * (target_q - old_q)

                state = next_state
                action = next_action

    def run_agent(self):
        state, info = self.env.reset()
        done = False
        step = 0

        while not done:
            step += 1
            action = self.choose_action(state, epsilon=0)
            new_state, reward, done, info = self.env.step(action)
            self.env.render()

            state = new_state
            time.sleep(0.5)


if __name__ == "__main__":

    rows = 5
    cols = 10
    start_pos = [3, 0]
    package_pos = [2, 9]
    goal_pos = [4, 9]
    max_steps = 40
    restricted_area = [[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8]]

    env = GridEnvironment(start_pos=start_pos, package_pos=package_pos, goal_pos=goal_pos, n_rows=rows, n_cols=cols, move_diaganols=True, max_steps=max_steps, restricted_area=restricted_area)

    qlearn_agent = QLearningAgent(env)
    sarsa_agent = SarsaAgent(env)

    print("Q-Learning Agent Training")
    qlearn_agent.train_agent()
    print("Q-Learning Training completed.")

    print("Running Q-Learning agent...")
    qlearn_agent.run_agent()

    time.sleep(3)

    print("SARSA Agent Training")
    sarsa_agent.train_agent()
    print("SARSA Training completed.")

    print("Running SARSA agent...")
    sarsa_agent.run_agent()
