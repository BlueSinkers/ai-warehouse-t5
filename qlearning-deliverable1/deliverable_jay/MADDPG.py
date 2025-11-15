import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# Import the new environment
from environment import MultiRobotEnv
from collections import deque
import random

# --- 1. Define the Actor Network (Decentralized) ---
# Each agent gets its own independent actor.
# For each robot, the Actor decides: “Given my observation, which continuous action should I take?”
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        # Simple feed-forward network
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim) 

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        # Use tanh to scale action between -1 and 1
        # This is for a continuous action space
        action = torch.tanh(self.fc3(x)) 
        return action

# --- 2. Define the Critic Network (Centralized) ---
# Each agent gets its own critic.
# It sees ALL observations and ALL *continuous* actions and outputs a Q-value.
class Critic(nn.Module):
    def __init__(self, n_agents, obs_dim_full, action_dim_full):
        super(Critic, self).__init__()
        
        # The critic's input is the concatenation of:
        # [all_observations, all_actions]
        input_dim = obs_dim_full + action_dim_full
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1) # Outputs a single Q-value

    def forward(self, full_obs_batch, full_action_batch):
        # Concatenate all observations and actions
        # full_action_batch is expected to be continuous actions
        x = torch.cat([full_obs_batch, full_action_batch], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# --- 3. Define a Replay Buffer ---
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        # Store the transition
        transition = (obs, actions, rewards, next_obs, dones)
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample a batch of transitions
        batch = random.sample(self.buffer, self.batch_size)
        
        # Unzip the batch
        obs_b, actions_b, rewards_b, next_obs_b, dones_b = zip(*batch)
        
        # Convert to numpy arrays for easier processing
        obs_b = np.array(obs_b)
        actions_b = np.array(actions_b)
        rewards_b = np.array(rewards_b)
        next_obs_b = np.array(next_obs_b)
        dones_b = np.array(dones_b)
        
        return obs_b, actions_b, rewards_b, next_obs_b, dones_b

    def __len__(self):
        return len(self.buffer)

# --- 4. Define the MADDPG Agent Controller ---
class MADDPG:
    def __init__(self, n_agents, obs_dims, action_dims, lr=0.001, gamma=0.99, tau=0.01,
                 buffer_size=1000000, batch_size=64):
        self.n_agents = n_agents
        self.obs_dims = obs_dims       # List of obs_dim for each agent
        self.action_dims = action_dims # List of action_dim for each agent
        self.gamma = gamma
        self.tau = tau # For soft target updates
        self.batch_size = batch_size

        # --- Centralized vs. Decentralized Dimensions ---
        # Total dimensions for the centralized critic
        self.obs_dim_full = sum(obs_dims)
        self.action_dim_full = sum(action_dims)

        # --- Create Networks ---
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []

        for i in range(n_agents):
            # Create actor and its target
            actor = Actor(obs_dims[i], action_dims[i])
            target_actor = Actor(obs_dims[i], action_dims[i])
            target_actor.load_state_dict(actor.state_dict())
            
            # Create critic and its target
            critic = Critic(n_agents, self.obs_dim_full, self.action_dim_full)
            target_critic = Critic(n_agents, self.obs_dim_full, self.action_dim_full)
            target_critic.load_state_dict(critic.state_dict())
            
            # Store networks and optimizers
            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.critics.append(critic)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr))
        
        # Create Replay Buffer
        self.buffer = ReplayBuffer(buffer_size, batch_size)

    def get_actions(self, obs_list, noise_scale=0.1):
        """
        --- DECENTRALIZED EXECUTION ---
        Gets actions for all agents based only on their
        individual observations.
        Includes Gaussian noise for exploration.
        """
        actions = []
        for i in range(self.n_agents):
            obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32)
            with torch.no_grad():
                # Get action from actor
                action = self.actors[i](obs_tensor).numpy()
                
            # Add noise for exploration
            noise = np.random.normal(0, noise_scale, size=self.action_dims[i])
            action = action + noise
            
            # Clip action to be between -1 and 1
            action = np.clip(action, -1.0, 1.0)
            
            actions.append(action)
        return actions
        
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """Stores a transition in the replay buffer."""
        self.buffer.store_transition(obs, actions, rewards, next_obs, dones)

    def _soft_update(self, local_model, target_model):
        """Helper function for soft target network updates."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    def update(self):
        """
        --- CENTRALIZED TRAINING ---
        Samples a batch from the buffer and updates all networks.
        """
        if len(self.buffer) < self.batch_size:
            # Not enough samples to train
            return
            
        # Sample a batch
        obs_b, actions_b, rewards_b, next_obs_b, dones_b = self.buffer.sample_batch()
        
        # Convert to tensors
        full_obs_batch = torch.tensor(obs_b.reshape(self.batch_size, -1), dtype=torch.float32)
        full_next_obs_batch = torch.tensor(next_obs_b.reshape(self.batch_size, -1), dtype=torch.float32)
        full_actions_batch = torch.tensor(actions_b.reshape(self.batch_size, -1), dtype=torch.float32)
        rewards_batch = torch.tensor(rewards_b, dtype=torch.float32)
        dones_batch = torch.tensor(dones_b, dtype=torch.float32)

        # --- 1. Update all Critic Networks ---
        
        # Get next actions from *target* actors (NO Gumbel-Softmax)
        with torch.no_grad():
            full_next_actions_batch = []
            for i in range(self.n_agents):
                agent_next_obs = torch.tensor(next_obs_b[:, i, :], dtype=torch.float32)
                next_action = self.target_actors[i](agent_next_obs)
                full_next_actions_batch.append(next_action)
            full_next_actions_batch_tensor = torch.cat(full_next_actions_batch, dim=1)

        for i in range(self.n_agents):
            # Get Q-target value from the target critic
            with torch.no_grad():
                target_q_next = self.target_critics[i](full_next_obs_batch, full_next_actions_batch_tensor)
                
                reward_i = rewards_batch[:, i].unsqueeze(1)
                done_i = dones_batch[:, i].unsqueeze(1) # Use agent's own done
                q_target = reward_i + self.gamma * target_q_next * (1 - done_i)

            # --- Centralized Critic Input ---
            # Critic uses the *actual actions* from the buffer
            q_current = self.critics[i](full_obs_batch, full_actions_batch)

            critic_loss = F.mse_loss(q_current, q_target)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # --- 2. Update all Actor Networks ---
        # This part is the most complex.
        # For agent i's loss, we need to use its *own* policy's action
        # but treat other agents' actions as constants (detached).
        
        # Get actions from *current* actors (with gradients)
        current_actions_list = []
        for i in range(self.n_agents):
            agent_obs = torch.tensor(obs_b[:, i, :], dtype=torch.float32)
            current_actions_list.append(self.actors[i](agent_obs))
        
        # Get actions from *current* actors (NO gradients)
        current_actions_list_detached = [a.detach() for a in current_actions_list]
        
        for i in range(self.n_agents):
            # Create the full action tensor for *this agent's loss*
            full_actions_for_loss_i = []
            for j in range(self.n_agents):
                if i == j:
                    # Use the one *with* grads
                    full_actions_for_loss_i.append(current_actions_list[j])
                else:
                    # Use the one *without* grads
                    full_actions_for_loss_i.append(current_actions_list_detached[j])
            
            full_actions_tensor_i = torch.cat(full_actions_for_loss_i, dim=1)
            
            # Calculate actor loss for agent i
            actor_loss = -self.critics[i](full_obs_batch, full_actions_tensor_i).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # --- 3. Soft-update all Target Networks ---
        for i in range(self.n_agents):
            self._soft_update(self.actors[i], self.target_actors[i])
            self._soft_update(self.critics[i], self.target_critics[i])

# --- 5. Example Usage ---
if __name__ == "__main__":
    
    # --- 1. Setup Environment ---
    N_ROBOTS = 2
    GRID_SIZE = (10, 10)
    BAY_POS = [0, 0]
    TARGET_POS = [[5, 5], [8, 8]]
    DEPOTS = [[9, 9]]
    LIMITS = [1, 2] 
    
    print("Initializing Multi-Robot Environment (Continuous)...")
    env = MultiRobotEnv(
        n_robots=N_ROBOTS,
        grid_size=GRID_SIZE,
        bay_pos=BAY_POS,
        target_pos=TARGET_POS,
        depot_pos=DEPOTS,
        robot_limits=LIMITS,
        max_speed=1.0,      # Robots can move 1 unit per step
        pickup_radius=2   # Must be within 0.5 units to pick/drop
    )

    # --- 2. Setup MADDPG ---
    obs_dims = [env.obs_dim] * N_ROBOTS
    action_dims = [env.action_dim] * N_ROBOTS # Now [2, 2]
    
    print(f"Setting up MADDPG for {N_ROBOTS} agents.")
    print(f"Agent Obs Dim: {env.obs_dim}, Agent Action Dim: {env.action_dim}")
    print(f"Centralized Critic Obs Dim: {sum(obs_dims)}")
    print(f"Centralized Critic Action Dim (Continuous): {sum(action_dims)}")
    print("-" * 30)

    maddpg = MADDPG(N_ROBOTS, obs_dims, action_dims, batch_size=32)

    # --- 3. Simple Training Loop (Conceptual) ---
    N_EPISODES = 10
    MAX_STEPS_PER_EP = 100
    NOISE_START = 0.5   # Starting noise scale
    NOISE_END = 0.05
    NOISE_DECAY_STEPS = N_EPISODES * MAX_STEPS_PER_EP * 0.5
    noise_scale = NOISE_START
    
    print("\n--- Starting Conceptual Training Loop ---")
    
    for ep in range(N_EPISODES):
        obs_list = env.reset()
        done_list = [False] * N_ROBOTS
        ep_reward_total = 0

        for step in range(MAX_STEPS_PER_EP):
            # 1. Get decentralized actions (with noise)
            actions = maddpg.get_actions(obs_list, noise_scale=noise_scale)
            
            # 2. Step the environment
            next_obs_list, rewards, dones, info = env.step(actions)
            
            # 3. Store transition in buffer
            maddpg.store_transition(obs_list, actions, rewards, next_obs_list, dones)
            
            # 4. Run centralized update
            maddpg.update()
            
            # 5. Update for next step
            obs_list = next_obs_list
            ep_reward_total += sum(rewards)
            noise_scale = max(NOISE_END, noise_scale - (NOISE_START - NOISE_END) / NOISE_DECAY_STEPS)
            
            if all(dones):
                break # Episode finished
        
        print(f"Episode {ep+1}/{N_EPISODES}, Total Reward: {ep_reward_total:.2f}, Noise: {noise_scale:.2f}")

    print("...Conceptual training complete.")