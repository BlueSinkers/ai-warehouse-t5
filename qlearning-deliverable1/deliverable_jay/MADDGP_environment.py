import numpy as np

# This environment is a 2D grid

class MultiRobotEnv:
    # This function is just defining and storing the basic environment settings
    def __init__(self, n_robots, grid_size, bay_pos, target_pos, depot_pos, robot_limits, max_steps=100, max_speed=1.0, pickup_radius=2):
        self.n_robots = n_robots
        self.grid_rows, self.grid_cols = grid_size
        self.bay_pos = np.array(bay_pos, dtype=np.float32)
        self.target_pos = [np.array(pos, dtype=np.float32) for pos in target_pos]
        self.depot_pos = [np.array(pos, dtype=np.float32) for pos in depot_pos]
        self.robot_limits = robot_limits 
        
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.pickup_radius = pickup_radius
        
        # --- Define Action Space ---
        # Continuous: [delta_y, delta_x]
        # Actions will be scaled by max_speed
        self.action_dim = 2 
        
        # --- Define Observation Space ---
        # 2 for its own coordinates (x, y), 1 for how much its holding, 1 for the limit, the rest are the coordinates for the other robots, targets, and depos
        self.obs_dim = 2 + 1 + 1 + 2 + (2 * (n_robots - 1)) + (2 * len(target_pos)) + (2 * len(depot_pos))
        
        # --- Penalties and Rewards ---
        self.move_penalty = -0.01
        self.collision_penalty = -5
        self.pickup_reward = 15
        self.dropoff_reward = 20
        
        # Initialize internal state
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        # All robots start at the bay
        self.robot_positions = np.array([self.bay_pos for _ in range(self.n_robots)], dtype=np.float32)
        # All robots start empty-handed
        self.robot_loads = np.zeros(self.n_robots, dtype=int)
        # Reset targets (use tuple(pos) as key)
        self.active_targets = {tuple(pos): 1 for pos in self.target_pos} # 1 = available
        self.step_count = 0
        
        return self._get_obs_list()

    # Building a long vector of observations for the actor network
    def _get_obs_list(self):
        """Gets the observation for each agent."""
        obs_list = []
        for i in range(self.n_robots):
            obs = []
            robot_pos = self.robot_positions[i]
            
            # 1-4: Own info
            obs.extend(robot_pos)
            obs.append(self.robot_loads[i])
            obs.append(self.robot_limits[i])
            obs.extend(self.bay_pos)
            
            # 5: Relative to other robots
            for j in range(self.n_robots):
                if i == j: continue
                obs.extend(self.robot_positions[j] - robot_pos)
                
            # 6: Relative to targets
            for pos_arr in self.target_pos:
                pos_tuple = tuple(pos_arr)
                if self.active_targets[pos_tuple] == 1:
                    obs.extend(pos_arr - robot_pos)
                else:
                    obs.extend([0, 0]) # Target is gone
                    
            # 7: Relative to depots
            for pos in self.depot_pos:
                obs.extend(np.array(pos) - robot_pos)
                
            obs_list.append(np.array(obs, dtype=np.float32))
        return obs_list

    def step(self, action_list):
        """
        Takes a list of actions (one for each robot) and executes a step.
        action_list = [[dy, dx], [dy, dx], ...]
        Actions are assumed to be in range [-1, 1]
        """
        self.step_count += 1
        rewards = np.full(self.n_robots, self.move_penalty)
        old_positions = self.robot_positions.copy() 
        new_positions = self.robot_positions.copy() 

        # 1. Calculate new positions
        for i, action in enumerate(action_list):
            # action is [dy, dx], scaled by max_speed
            delta = np.array(action, dtype=np.float32) * self.max_speed
            new_positions[i] += delta
        
        # 2. Check for collisions (walls and other robots)
        for i in range(self.n_robots):
            # Wall collision (clamp position)
            new_positions[i, 0] = np.clip(new_positions[i, 0], 0, self.grid_rows - 1)
            new_positions[i, 1] = np.clip(new_positions[i, 1], 0, self.grid_cols - 1)
            
            # Robot-robot collision (simple model: just apply penalty)
            for j in range(i + 1, self.n_robots):
                dist = np.linalg.norm(new_positions[i] - new_positions[j])
                if dist < self.pickup_radius * 2: # Simple collision radius
                    rewards[i] += self.collision_penalty
                    rewards[j] += self.collision_penalty

        # 3. Update positions
        self.robot_positions = new_positions

        # Reward Shaping to make sure agents move toward targets
        for i in range(self.n_robots):
            pos_before = old_positions[i]
            pos_after = self.robot_positions[i]

            # Robot carrying something -> move toward nearest depot
            if self.robot_loads[i] > 0:
                depot_dists_before = [np.linalg.norm(pos_before - d) for d in self.depot_pos]
                depot_dists_after = [np.linalg.norm(pos_after - d) for d in self.depot_pos]
                reward_delta = (min(depot_dists_before) - min(depot_dists_after))
                rewards[i] += 0.5 * reward_delta

            # Robot empty -> move toward nearest active target
            else:
                active_targets = [pos for pos in self.target_pos if self.active_targets[tuple(pos)] == 1]
                if active_targets:
                    target_dists_before = [np.linalg.norm(pos_before - t) for t in active_targets]
                    target_dists_after = [np.linalg.norm(pos_after - t) for t in active_targets]
                    reward_delta = (min(target_dists_before) - min(target_dists_after))
                    rewards[i] += 0.5 * reward_delta
        
        # 4. Check for pickups and dropoffs
        # Pickup if robot is close enough to a target, robot has free capacity, then update the target being taken and increment robot load
        # Droff Off if robot is close to a depot, then drop all items and recieve dropoff reward
        for i in range(self.n_robots):
            # Check for pickup at an active target
            for pos_arr in self.target_pos:
                pos_tuple = tuple(pos_arr)
                if self.active_targets[pos_tuple] == 1 and self.robot_loads[i] < self.robot_limits[i]:
                    dist = np.linalg.norm(self.robot_positions[i] - pos_arr)
                    if dist <= self.pickup_radius:
                        self.robot_loads[i] += 1
                        self.active_targets[pos_tuple] = 0 # Target is taken
                        rewards[i] += self.pickup_reward
                        break # Can only pick up one item per step
            
            # Check for dropoff at a depot
            if self.robot_loads[i] > 0:
                for pos_arr in self.depot_pos:
                    dist = np.linalg.norm(self.robot_positions[i] - pos_arr)
                    if dist <= self.pickup_radius:
                        rewards[i] += self.dropoff_reward * self.robot_loads[i]
                        self.robot_loads[i] = 0 # Empties load
                        break # Can only drop at one depot
        
        # 5. Check for termination
        # End if all targets are collected & all robots are in depot OR if max steps are reached
        done = False
        all_targets_delivered = all(v == 0 for v in self.active_targets.values())
        
        # Check if all robots are back at the bay
        all_at_bay = True
        for i in range(self.n_robots):
            if np.linalg.norm(self.robot_positions[i] - self.bay_pos) > self.pickup_radius:
                all_at_bay = False
                break
        
        if (all_targets_delivered and all_at_bay) or self.step_count >= self.max_steps:
            done = True
            
        dones = [done] * self.n_robots
        obs_list = self._get_obs_list()
        
        return obs_list, rewards, dones, {}
