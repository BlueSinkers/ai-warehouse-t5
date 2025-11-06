import numpy as np
import time
import sys 

class GridEnvironment:
    def __init__(self, n_rows=5, n_cols=5, start_pos=[0,0], package_pos=[2,4], goal_pos=[4,0], max_steps=20, move_diaganols=True):
        self.start_pos = start_pos
        self.package_pos = package_pos
        self.goal_pos = goal_pos
        self.move_diaganols = move_diaganols

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
        self.wall_penalty = -5 # Penalty for hitting a wall (trying to move out of bounds)
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
        print(f"Agent {self.agent_pos} | Item: {'Yes' if self.has_package else 'No'}")

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 1 - 0.995
        self.min_epsilon = 0.01
        self.total_epochs = 20000

        self.q_table = np.zeros((env.n_states, env.n_actions))

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])


    def train_agent(self):
        rewards_per_epoch = []
        for epoch in range(self.total_epochs):
            state, info = self.env.reset()
            done = False
            total_reward = 0

            curr_epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * epoch)
            while not done:
                action = self.choose_action(state, curr_epsilon)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                target_q = reward + self.gamma * np.max(self.q_table[next_state, :]) * (1 - done)
                self.q_table[state, action] += self.alpha * (target_q - self.q_table[state, action])
                state = next_state

            rewards_per_epoch.append(total_reward)

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

    rows = 10
    cols = 15
    start_pos = [6, 0]
    package_pos = [1, 12]
    goal_pos = [9, 3]
    max_steps = 30

    env = GridEnvironment(start_pos=start_pos, package_pos=package_pos, goal_pos=goal_pos, n_rows=rows, n_cols=cols, move_diaganols=True, max_steps=max_steps)
    agent = QLearningAgent(env)

    print("Training agent...")
    agent.train_agent()
    print("Training completed.")

    print("Running trained agent...")
    agent.run_agent()

    print(agent.q_table)




