"""
Multi-Agent Pickup and Delivery with Continuous MAPPO
A system where 3 agents learn to pickup items and deliver them to a depot
using continuous observations and discrete actions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon
from collections import deque

# CONFIGURATION

class Config:
    """Configuration parameters for the environment and training"""
    # Environment parameters
    grid_size = 4.0  # Continuous grid from [0,0] to [4,4]
    num_agents = 3
    max_capacity = 2  # Max items each agent can carry
    move_step = 0.3  # How far agents move per step
    num_items = 5  # Number of items in the environment
    
    # Fixed locations
    depot_location = np.array([3.5, 3.5])  # Top-right corner
    item_spawn_points = [
        np.array([0.5, 0.5]),
        np.array([0.5, 3.5]),
        np.array([2.0, 2.0]),
        np.array([3.5, 0.5]),
        np.array([1.5, 1.5])
    ]
    
    # Agent starting positions
    agent_start_positions = [
        np.array([0.5, 2.0]),
        np.array([2.0, 0.5]),
        np.array([3.5, 2.0])
    ]
    
    # Training parameters
    lr_actor = 3e-4
    lr_critic = 1e-3
    gamma = 0.99  # Discount factor
    gae_lambda = 0.95  # GAE parameter
    clip_epsilon = 0.2  # PPO clip parameter
    ppo_epochs = 4  # Number of PPO update epochs per batch
    batch_size = 64
    max_episodes = 2000
    max_steps_per_episode = 200
    
    # Action space: 0=up, 1=down, 2=left, 3=right, 4=pickup, 5=dropoff
    num_actions = 6
    
    # Observation space size per agent
    obs_size = (
        2 +  # Agent's own position
        1 +  # Agent's inventory count
        2 +  # Depot position
        num_items * 3 +  # Each item: x, y, available (1 or 0)
        (num_agents - 1) * 3  # Other agents: x, y, inventory
    )


# ============================================================================
# ENVIRONMENT
# ============================================================================

class MultiAgentPickupEnvironment:
    """
    Multi-agent environment where agents pickup items and deliver to depot.
    Continuous observation space, discrete action space.
    """
    
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Initialize agent positions (continuous)
        self.agent_positions = [pos.copy() for pos in self.config.agent_start_positions]
        
        # Initialize agent inventories (number of items held)
        self.agent_inventories = [0] * self.config.num_agents
        
        # Initialize items (all available at spawn points)
        self.item_positions = [pos.copy() for pos in self.config.item_spawn_points]
        self.item_available = [True] * self.config.num_items
        
        # Statistics
        self.total_deliveries = 0
        self.step_count = 0
        
        return self._get_observations()
    
    def _get_observations(self):
        """Get observations for all agents"""
        observations = []
        
        for i in range(self.config.num_agents):
            obs = []
            
            # Agent's own position
            obs.extend(self.agent_positions[i])
            
            # Agent's inventory count (normalized)
            obs.append(self.agent_inventories[i] / self.config.max_capacity)
            
            # Depot position
            obs.extend(self.config.depot_location)
            
            # Item positions and availability
            for j in range(self.config.num_items):
                obs.extend(self.item_positions[j])
                obs.append(1.0 if self.item_available[j] else 0.0)
            
            # Other agents' positions and inventories
            for j in range(self.config.num_agents):
                if i != j:
                    obs.extend(self.agent_positions[j])
                    obs.append(self.agent_inventories[j] / self.config.max_capacity)
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return observations
    
    def step(self, actions):
        """
        Execute actions for all agents
        actions: list of action indices for each agent
        """
        rewards = [0.0] * self.config.num_agents
        
        # Process movement actions first
        for i, action in enumerate(actions):
            if action == 0:  # Up
                self.agent_positions[i][1] = min(self.config.grid_size, 
                                                  self.agent_positions[i][1] + self.config.move_step)
            elif action == 1:  # Down
                self.agent_positions[i][1] = max(0, self.agent_positions[i][1] - self.config.move_step)
            elif action == 2:  # Left
                self.agent_positions[i][0] = max(0, self.agent_positions[i][0] - self.config.move_step)
            elif action == 3:  # Right
                self.agent_positions[i][0] = min(self.config.grid_size, 
                                                  self.agent_positions[i][0] + self.config.move_step)
        
        # Process pickup and dropoff actions
        for i, action in enumerate(actions):
            if action == 4:  # Pickup
                rewards[i] += self._try_pickup(i)
            elif action == 5:  # Dropoff
                rewards[i] += self._try_dropoff(i)
        
        # Small negative reward for each step (encourages efficiency)
        for i in range(self.config.num_agents):
            rewards[i] -= 0.01
        
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.config.max_steps_per_episode or \
               (not any(self.item_available) and sum(self.agent_inventories) == 0)
        
        observations = self._get_observations()
        
        return observations, rewards, done, {}
    
    def _try_pickup(self, agent_idx):
        """Attempt to pickup an item"""
        # Check if agent has capacity
        if self.agent_inventories[agent_idx] >= self.config.max_capacity:
            return -0.1  # Small penalty for invalid action
        
        # Check if there's an available item nearby
        agent_pos = self.agent_positions[agent_idx]
        pickup_distance = 0.5  # Distance threshold for pickup
        
        for item_idx in range(self.config.num_items):
            if self.item_available[item_idx]:
                item_pos = self.item_positions[item_idx]
                distance = np.linalg.norm(agent_pos - item_pos)
                
                if distance < pickup_distance:
                    # Successful pickup
                    self.item_available[item_idx] = False
                    self.agent_inventories[agent_idx] += 1
                    return 1.0  # Reward for pickup
        
        return -0.1  # Penalty for failed pickup
    
    def _try_dropoff(self, agent_idx):
        """Attempt to dropoff items at depot"""
        # Check if agent has items
        if self.agent_inventories[agent_idx] == 0:
            return -0.1  # Small penalty for invalid action
        
        # Check if agent is near depot
        agent_pos = self.agent_positions[agent_idx]
        depot_pos = self.config.depot_location
        dropoff_distance = 0.5  # Distance threshold for dropoff
        
        distance = np.linalg.norm(agent_pos - depot_pos)
        
        if distance < dropoff_distance:
            # Successful dropoff
            items_delivered = self.agent_inventories[agent_idx]
            self.agent_inventories[agent_idx] = 0
            self.total_deliveries += items_delivered
            return 5.0 * items_delivered  # Large reward for delivery
        
        return -0.1  # Penalty for failed dropoff


# ============================================================================
# NEURAL NETWORKS

class Actor(nn.Module):
    """
    Actor network: maps observation to action probabilities
    Used by each agent to select actions
    """
    
    def __init__(self, obs_size, num_actions, hidden_size=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions)
        )
    
    def forward(self, obs):
        """Forward pass: obs -> action logits"""
        return self.network(obs)
    
    def get_action(self, obs):
        """Sample action from policy"""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class Critic(nn.Module):
    """
    Critic network: maps global state to value estimate
    Centralized - sees observations from all agents
    """
    
    def __init__(self, global_obs_size, hidden_size=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(global_obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, global_obs):
        """Forward pass: global observation -> state value"""
        return self.network(global_obs)


# ============================================================================
# MAPPO ALGORITHM

class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization
    Centralized training with decentralized execution
    """
    
    def __init__(self, config):
        self.config = config
        
        # Create shared actor for all agents (parameter sharing)
        self.actor = Actor(config.obs_size, config.num_actions)
        
        # Create centralized critic
        global_obs_size = config.obs_size * config.num_agents
        self.critic = Critic(global_obs_size)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        
        # Storage for trajectories
        self.reset_buffers()
    
    def reset_buffers(self):
        """Clear trajectory buffers"""
        self.observations = [[] for _ in range(self.config.num_agents)]
        self.actions = [[] for _ in range(self.config.num_agents)]
        self.log_probs = [[] for _ in range(self.config.num_agents)]
        self.rewards = [[] for _ in range(self.config.num_agents)]
        self.values = []
        self.dones = []
    
    def select_actions(self, observations):
        """Select actions for all agents"""
        actions = []
        log_probs = []
        
        # Get actions from each agent's actor
        for obs in observations:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob = self.actor.get_action(obs_tensor)
            actions.append(action)
            log_probs.append(log_prob)
        
        # Get value estimate from centralized critic
        global_obs = torch.FloatTensor(np.concatenate(observations)).unsqueeze(0)
        value = self.critic(global_obs)
        
        return actions, log_probs, value.item()
    
    def store_transition(self, observations, actions, log_probs, rewards, value, done):
        """Store transition in buffers"""
        for i in range(self.config.num_agents):
            self.observations[i].append(observations[i])
            self.actions[i].append(actions[i])
            self.log_probs[i].append(log_probs[i])
            self.rewards[i].append(rewards[i])
        
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        # Average rewards across agents for shared learning signal
        avg_rewards = [np.mean([self.rewards[i][t] for i in range(self.config.num_agents)]) 
                       for t in range(len(self.rewards[0]))]
        
        advantages = []
        gae = 0
        
        # Compute GAE backwards through time
        for t in reversed(range(len(avg_rewards))):
            if t == len(avg_rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = avg_rewards[t] + self.config.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_observations):
        """Update actor and critic networks using PPO"""
        # Get value for next state
        global_next_obs = torch.FloatTensor(np.concatenate(next_observations)).unsqueeze(0)
        next_value = self.critic(global_next_obs).item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.FloatTensor(returns)
        
        # Prepare data for update
        all_obs = []
        all_actions = []
        all_log_probs = []
        
        for i in range(self.config.num_agents):
            all_obs.extend(self.observations[i])
            all_actions.extend(self.actions[i])
            all_log_probs.extend(self.log_probs[i])
        
        obs_tensor = torch.FloatTensor(np.array(all_obs))
        actions_tensor = torch.LongTensor(all_actions)
        old_log_probs = torch.stack(all_log_probs)
        
        # Prepare global observations for critic
        num_steps = len(self.values)
        global_obs_list = []
        for t in range(num_steps):
            step_obs = [self.observations[i][t] for i in range(self.config.num_agents)]
            global_obs_list.append(np.concatenate(step_obs))
        global_obs_tensor = torch.FloatTensor(np.array(global_obs_list))
        
        # PPO update for multiple epochs
        for _ in range(self.config.ppo_epochs):
            # Actor update
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            # Expand advantages to match all agent-step pairs
            expanded_advantages = advantages.repeat_interleave(self.config.num_agents)
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * expanded_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                               1 + self.config.clip_epsilon) * expanded_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Critic update
            values_pred = self.critic(global_obs_tensor).squeeze()
            critic_loss = nn.MSELoss()(values_pred, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Clear buffers
        self.reset_buffers()
        
        return actor_loss.item(), critic_loss.item()


# ============================================================================
# TRAINING

def train_mappo(config, render_every=100):
    """Main training loop"""
    env = MultiAgentPickupEnvironment(config)
    mappo = MAPPO(config)
    
    # Training statistics
    episode_rewards = []
    episode_deliveries = []
    recent_rewards = deque(maxlen=100)
    
    print("Starting MAPPO Training...")
    print(f"Episodes: {config.max_episodes}, Max Steps: {config.max_steps_per_episode}")
    print("-" * 60)
    
    for episode in range(config.max_episodes):
        observations = env.reset()
        episode_reward = 0
        
        for step in range(config.max_steps_per_episode):
            # Select actions
            actions, log_probs, value = mappo.select_actions(observations)
            
            # Take step in environment
            next_observations, rewards, done, _ = env.step(actions)
            
            # Store transition
            mappo.store_transition(observations, actions, log_probs, rewards, value, done)
            
            episode_reward += sum(rewards)
            observations = next_observations
            
            if done:
                break
        
        # Update networks
        actor_loss, critic_loss = mappo.update(observations)
        
        # Track statistics
        episode_rewards.append(episode_reward)
        episode_deliveries.append(env.total_deliveries)
        recent_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_deliveries = np.mean(list(recent_rewards)[-50:])
            print(f"Episode {episode + 1}/{config.max_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Deliveries: {env.total_deliveries} | "
                  f"Actor Loss: {actor_loss:.4f} | "
                  f"Critic Loss: {critic_loss:.4f}")
        
        # Render visualization periodically
        if (episode + 1) % render_every == 0:
            print(f"\nRendering episode {episode + 1}...")
            visualize_episode(env, mappo, config)
    
    print("\nTraining complete!")
    return mappo, episode_rewards, episode_deliveries


# ============================================================================
# VISUALIZATION

def visualize_episode(env, mappo, config, save_path=None):
    """Visualize a single episode"""
    observations = env.reset()
    
    # Storage for animation
    agent_trajectories = [[] for _ in range(config.num_agents)]
    inventory_history = [[] for _ in range(config.num_agents)]
    item_history = []
    
    # Run episode
    for step in range(config.max_steps_per_episode):
        # Store current state
        for i in range(config.num_agents):
            agent_trajectories[i].append(env.agent_positions[i].copy())
            inventory_history[i].append(env.agent_inventories[i])
        item_history.append([available for available in env.item_available])
        
        # Get actions (no exploration noise for visualization)
        actions, _, _ = mappo.select_actions(observations)
        
        # Take step
        observations, rewards, done, _ = env.step(actions)
        
        if done:
            break
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def init():
        ax.clear()
        ax.set_xlim(-0.5, config.grid_size + 0.5)
        ax.set_ylim(-0.5, config.grid_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Multi-Agent Pickup and Delivery')
        return []
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(-0.5, config.grid_size + 0.5)
        ax.set_ylim(-0.5, config.grid_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Step {frame} - Total Deliveries: {env.total_deliveries}')
        
        # Draw depot
        depot = plt.Circle(config.depot_location, 0.2, color='gold', 
                          label='Depot', zorder=1)
        ax.add_patch(depot)
        ax.plot(config.depot_location[0], config.depot_location[1], 
               'k*', markersize=20, zorder=2)
        
        # Draw items
        for i, spawn_point in enumerate(config.item_spawn_points):
            if item_history[frame][i]:  # Item is available
                item = plt.Rectangle((spawn_point[0] - 0.1, spawn_point[1] - 0.1), 
                                    0.2, 0.2, color='green', 
                                    label='Item' if i == 0 else '', zorder=1)
                ax.add_patch(item)
        
        # Draw agents
        colors = ['blue', 'red', 'purple']
        for i in range(config.num_agents):
            pos = agent_trajectories[i][frame]
            agent = plt.Circle(pos, 0.15, color=colors[i], 
                             label=f'Agent {i+1}', alpha=0.7, zorder=3)
            ax.add_patch(agent)
            
            # Add agent ID and inventory
            ax.text(pos[0], pos[1], f'{i+1}\n[{inventory_history[i][frame]}]', 
                   ha='center', va='center', color='white', 
                   fontweight='bold', fontsize=9, zorder=4)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                  frames=len(agent_trajectories[0]), 
                                  interval=200, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
        print(f"Animation saved to {save_path}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION

if __name__ == "__main__":
    # Create configuration
    config = Config()
    
    # Train the agents
    #mappo, rewards, deliveries = train_mappo(config, render_every=500)

    #for no visualizations
    mappo, rewards, deliveries = train_mappo(config, render_every=10000)  # No visualizations during training
    
    # Plot training progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(deliveries)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Deliveries')
    ax2.set_title('Successful Deliveries')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final visualization
    print("\nGenerating final visualization...")
    env = MultiAgentPickupEnvironment(config)
    visualize_episode(env, mappo, config)
