import queue
import random
import heapq  # For priority queue used in A* algorithm

# Function to create a grid based on user-defined size and randomly place the treasure
def create_grid(size):
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    grid[0][0] = 'S'  # Player start point
    treasure_x, treasure_y = random.randint(0, size-1), random.randint(0, size-1)
    while (treasure_x, treasure_y) == (0, 0):
        treasure_x, treasure_y = random.randint(0, size-1), random.randint(0, size-1)
    grid[treasure_x][treasure_y] = 'T'  # Treasure
    return grid, (treasure_x, treasure_y)

# Function to add random obstacles
def add_obstacles(grid, num_obstacles):
    size = len(grid)
    for _ in range(num_obstacles):
        obstacle_x, obstacle_y = random.randint(0, size-1), random.randint(0, size-1)
        while grid[obstacle_x][obstacle_y] in ['S', 'T']:
            obstacle_x, obstacle_y = random.randint(0, size-1), random.randint(0, size-1)
        grid[obstacle_x][obstacle_y] = 'X'  # Obstacle
    return grid

# Function to check if a position is valid
def is_valid_position(grid, x, y):
    size = len(grid)
    return 0 <= x < size and 0 <= y < size and grid[x][y] != 'X'

# A* algorithm for monster pathfinding
def a_star(grid, start, goal):
    size = len(grid)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}

    while open_list:
        current = heapq.heappop(open_list)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for direction in directions:
            next_x, next_y = current[0] + direction[0], current[1] + direction[1]
            next_state = (next_x, next_y)

            if is_valid_position(grid, next_x, next_y):
                tentative_g_score = g_score[current] + 1

                if next_state not in g_score or tentative_g_score < g_score[next_state]:
                    came_from[next_state] = current
                    g_score[next_state] = tentative_g_score
                    f_score[next_state] = tentative_g_score + manhattan_distance(next_state, goal)
                    heapq.heappush(open_list, (f_score[next_state], next_state))

    return []

# Helper function to reconstruct path in A* algorithm
def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# Manhattan distance heuristic for A*
def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

# Player movement function
def move_player(grid, player_pos):
    print("Your move! Enter direction (W/A/S/D):")
    move = input().lower()
    directions = {'w': (-1, 0), 's': (1, 0), 'a': (0, -1), 'd': (0, 1)}

    if move in directions:
        new_x = player_pos[0] + directions[move][0]
        new_y = player_pos[1] + directions[move][1]
        if is_valid_position(grid, new_x, new_y):
            return new_x, new_y
        else:
            print("Invalid move, obstacle ahead!")
            return player_pos
    else:
        print("Invalid input! Use W/A/S/D.")
        return player_pos

# Function to print the grid with player, monster, and treasure
def print_grid(grid, player_pos, monster_pos):
    size = len(grid)
    display_grid = [row.copy() for row in grid]

    # Mark player and monster
    display_grid[player_pos[0]][player_pos[1]] = 'P'  # Player
    display_grid[monster_pos[0]][monster_pos[1]] = 'M'  # Monster

    # Print the grid
    for row in display_grid:
        print(' | '.join(row))
        print('-' * (size * 4 - 1))

# Main function to run the game
def monster_chase():
    # Setup game
    size = int(input("Enter grid size: "))
    num_obstacles = int(input("Enter number of obstacles: "))

    grid, treasure_pos = create_grid(size)
    grid = add_obstacles(grid, num_obstacles)

    player_pos = (0, 0)
    monster_pos = (size - 1, size - 1)  # Monster starts at bottom-right corner

    while player_pos != treasure_pos:
        print_grid(grid, player_pos, monster_pos)

        # Move player
        player_pos = move_player(grid, player_pos)

        # Monster chases player using A*
        path_to_player = a_star(grid, monster_pos, player_pos)
        if path_to_player:
            monster_pos = path_to_player[1]  # Move monster one step along the path

        # Check if monster caught the player
        if player_pos == monster_pos:
            print("The monster caught you! Game over.")
            return

    print("Congratulations! You reached the treasure!")
    print_grid(grid, player_pos, monster_pos)

# Run the game
monster_chase()
