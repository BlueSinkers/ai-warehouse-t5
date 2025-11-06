# Jay's Deliverables for QLearning

File: qlearn_sarsa.py
This file contains 3 classes (GridEnvironment, QLearningAgent, SarsaAgent)
I used the libraries NumPy, Time, Sys

This project requires an environment (number of rows, number of columns, starting position of the agent, position of the package, position of the goal, locations of walls, locations of restricted area, 
max number of steps the agent can take, whether or not the agent can move diagonally). These are all customizable but there are default values for all of them if any of them are not specified. After 
the environment is created, it creates 2 agents (QLearningAgent and SarsaAgent). When the main function is ran, it runs the QLearningAgent and shows the path that it takes and how many moves it makes. 
After the QLearningAgent runs, the SarsaAgent runs and does the same. In the current main function, there is an example of when the QLearningAgent runs faster than the SarsaAgent.

## QLearning vs. SARSA:
- Qlearning is "Off-Policy" because it finds the optimal path through an optimistic approach since it is more willing to go to risky areas if it's closer to the goal.
- SARSA is "On-Policy" because it will take a safer approach to getting to the goal.

This can be seen as in the example that is in the main function. While there is a restricted area near the goal, you can see that the QLearningAgent walks right next to the restricted area to get to the goal faster. 
This is different than the SARSA agent that goes further away from the restricted area and ends up taking more moves to get to the goal. 

This can be seen as in both of their formula's used for their Q-value. 

QLearning: Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a') - Q(s,a))
                                                          
SARSA: Q(s, a) = Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))

where alpha = learning rate and gamma = discount rate.
The difference in these formulas is that QLearning is getting the max a' as it is optimistic but SARSA will pick a' by using the choose_action (epsilon-greedy) function at s'.

## Hyperparameters for Running
- These are the same for both QLearning and SARSA
- Alpha / Learning Rate: How much you want the new information to override the old information or how fast the agent learns [High Alpha -> More importance to new info]
- Gamma / Discount Rate: How much importance you want on future rewards compared to immediate rewards. [Higher gamma = more importance on future rewards]
- Epsilon / Exploration Rate: The probability the agent will explore vs. explot. [Higher epsilon -> More exploration as it makes the agent more likely to pick random actions]
- Epsilon Decay / Exploration Decay Rate: How fast the epsilon value decreases after each run.  [High Epsilon Decay -> Epsilon drops quickly and agent stops exploring quickly and begins to exploit (picking best known action)]
- Min Epsilon / Minimum Exploration Rate: The lowest epsilon will go [Choose a small, positive amount to make sure there is always some king of exploration happening]
- Total_epochs: The number of runs the agent does for training [Higher total_epochs -> More runtime but better Q-table]


