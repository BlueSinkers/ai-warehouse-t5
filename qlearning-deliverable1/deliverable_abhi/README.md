# Abhi's Deliverable for the QLEARNING PROJECT

This is not a great project, but it accomplishes what I was asking for -- it uses PyGame to show some visualization of tasks being completed, and uses the QLearning algorithm. **Pygame is needed to run the project**.

**Note that like I said, I used GPT throughout. I did NOT use Cursor or Github Copilot however, and did most of the integration/combining myself.**

# Basic structure:
- **DEMOS**: This folder contains demos of the transforms so I could see how they work and perfected the algorithm (this part was about 40% me, 60% GPT)
    - *demo_qlearning.py* -> Demo of QLearning
    - *demo_sarsa.py* -> Demo of SARSA
- **FUNCTIONS**: This folder contains the exact functions being used in the calls for the path.
    - *qlearning_function.py* -> Function for QLearning (contains the adjustments for the hyperparameters)
    - *sarsa_function.py* -> Function for SARSA
- **GRAPHICS**: This folder contains the scripts used for handling the PyGame rendering etc.
    - *convert.py* -> converts the rendering from characters representing obstacles/objects to numbers corresponding to reward/policy
    - *graphics.py* -> main engine handling PyGame for rendering of the path 
- **MAIN**: This file runs everything altogether, and allows for adjustment of the input in terms of the grid and obstacles/objects.

# Improvements
The diagonal toggle isn't working fully, and I need to run some more testcases to get this fully working.

