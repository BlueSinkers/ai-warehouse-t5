import pygame
import time

def visualize_path_pygame(grid, path, cell_size=50, delay=300, padding=50):
    """
    grid: 2D list, strings representing cell types
        'e' = empty
        'w' = wall
        's' = spill
        'g' = goal
    path: list of (row, col) tuples representing robot's path
    cell_size: size of each square in pixels
    delay: milliseconds between robot moves
    padding: space around the grid in pixels
    """
    pygame.init()

    n_rows = len(grid)
    n_cols = len(grid[0])
    legend_width = 200
    width = n_cols * cell_size + 2*padding + legend_width
    height = n_rows * cell_size + 2*padding

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Robot Path Visualization")

    font = pygame.font.SysFont(None, 24)

    # define colors
    colors = {
        'e': (255, 255, 255),   # empty = white
        'w': (0, 0, 0),         # wall = black
        's': (0, 0, 255),       # spill = blue
        'g': (0, 255, 0),       # goal = green
    }
    path_color = (255, 150, 150)  # light red for visited path
    robot_color = (255, 0, 0)     # bright red for current robot position

    visited = set()

    def draw_grid(robot_pos=None):
        screen.fill((50, 50, 50))  # background
        # draw main grid
        for r in range(n_rows):
            for c in range(n_cols):
                if (r,c) in visited:
                    color = path_color
                else:
                    color = colors.get(grid[r][c], (200, 200, 200))
                pygame.draw.rect(screen, color, 
                                 (padding + c*cell_size, padding + r*cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, (150, 150, 150), 
                                 (padding + c*cell_size, padding + r*cell_size, cell_size, cell_size), 1)
        # draw robot on top
        if robot_pos:
            r, c = robot_pos
            pygame.draw.rect(screen, robot_color, 
                             (padding + c*cell_size, padding + r*cell_size, cell_size, cell_size))

        # draw legend panel
        legend_x = padding + n_cols*cell_size + 20
        start_y = padding
        legend_items = [
            ("empty", colors['e']),
            ("wall", colors['w']),
            ("spill", colors['s']),
            ("goal", colors['g']),
            ("visited path", path_color),
            ("robot", robot_color)
        ]
        for i, (label, color) in enumerate(legend_items):
            rect_y = start_y + i*40
            pygame.draw.rect(screen, color, (legend_x, rect_y, 30, 30))
            # draw label text in white or black depending on color brightness
            text_color = (0,0,0) if sum(color)/3 > 128 else (255,255,255)
            text_surf = font.render(label, True, text_color)
            screen.blit(text_surf, (legend_x + 40, rect_y + 5))


    # show static grid first for 3 seconds
    draw_grid()
    pygame.display.update()
    start_time = time.time()
    while time.time() - start_time < 3:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    # animate robot along path
    for pos in path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        visited.add(pos)
        draw_grid(robot_pos=pos)
        pygame.display.update()
        pygame.time.delay(delay)

    # keep window open until closed
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


# Example usage
grid = [
    ['e', 'e', 'e', 'e', 'e'],
    ['e', 'w', 'e', 'w', 'e'],
    ['e', 'e', 'e', 'e', 'g'],
    ['e', 'w', 'e', 'e', 'e'],
    ['e', 'e', 'e', 'w', 'e']
]

path = [(0,0), (1,0), (2,0), (2,1), (2,2), (2,3), (2,4)]

visualize_path_pygame(grid, path)
