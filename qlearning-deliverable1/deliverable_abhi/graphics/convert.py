def convert_grid(char_grid):
    #Convert a character grid to numeric reward grid.
    mapping = {
        'e': e,
        'w': w,
        'g': g,
        's': s
    }
    numeric_grid = [[mapping[cell] for cell in row] for row in char_grid]
    return numeric_grid