from functions.qlearning_function import q_learning_path
from graphics.graphics import visualize_path_pygame
from graphics.convert import convert_grid


#reward values
s = 0      # start
e = -1     # empty
w = -100   # wall
g = 10     # goal

#start and goal positions
start_pos = (0, 0)
goal_pos = (2, 4)

#grid parameters

def main():
    #adjust all the hyperparameters in the qlearning function if needed 
    
    
    
    numeric_grid = convert_grid(grid)
    path = q_learning_path(grid, start_pos, goal_pos)
    visualize_path_pygame(grid, path)