import pygame
import random

pygame.init()

# Colors
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BROWN = (150, 75, 0)

# Grid and screen settings
WIDTH, HEIGHT = 1000, 1000
TILE_SIZE = 50
GRID_WIDTH = WIDTH // TILE_SIZE
GRID_HEIGHT = HEIGHT // TILE_SIZE
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Font for displaying health and money
font = pygame.font.SysFont("Arial", 12)

def gen_resources(num):
    """Generate random positions for resources."""
    return set([(random.randrange(0, GRID_HEIGHT), random.randrange(0, GRID_WIDTH)) for _ in range(num)])

def draw_grid(positions, foods, money_tiles, health, money):
    """Draw cells, foods, and money tiles on the grid."""
    for position in positions:
        col, row = position
        top_left = (col * TILE_SIZE, row * TILE_SIZE)
        pygame.draw.rect(screen, WHITE, (*top_left, TILE_SIZE, TILE_SIZE))

        # Display health and money inside the cell
        text_surface = font.render(f"H:{health[position]}|M:{money[position]}", True, BLACK)
        text_rect = text_surface.get_rect(center=(top_left[0] + TILE_SIZE // 2, top_left[1] + TILE_SIZE // 2))
        screen.blit(text_surface, text_rect)

    # Draw food
    for food in foods:
        col, row = food
        top_left = (col * TILE_SIZE, row * TILE_SIZE)
        pygame.draw.rect(screen, BROWN, (*top_left, TILE_SIZE, TILE_SIZE))

    # Draw money tiles
    for money_tile in money_tiles:
        col, row = money_tile
        top_left = (col * TILE_SIZE, row * TILE_SIZE)
        pygame.draw.rect(screen, GREEN, (*top_left, TILE_SIZE, TILE_SIZE))

    # Draw grid lines
    for row in range(GRID_HEIGHT):
        pygame.draw.line(screen, WHITE, (0, row * TILE_SIZE), (WIDTH, row * TILE_SIZE))
    for col in range(GRID_WIDTH):
        pygame.draw.line(screen, WHITE, (col * TILE_SIZE, 0), (col * TILE_SIZE, HEIGHT))

def get_state(position, health, money, foods, money_tiles):
    return (position, health, money, tuple(sorted(foods)), tuple(sorted(money_tiles)))

def get_possible_actions():
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

def transition(state, action, grid_width, grid_height):
    (position, health, money, foods, money_tiles) = state
    x, y = position
    dx, dy = action
    new_position = ((x + dx) % grid_width, (y + dy) % grid_height)
    
    # Update health and money
    new_health = health - 1  # Movement penalty
    new_money = money
    new_foods = set(foods)
    new_money_tiles = set(money_tiles)
    
    if new_position in new_foods and new_money >= 1:
        new_health += 5  # Food restores health
        new_money -= 1
        new_foods.remove(new_position)

    if new_position in new_money_tiles:
        new_money += 1
        new_money_tiles.remove(new_position)

    if new_health <= 0:
        return None  # Invalid state (cell dies)

    return (new_position, new_health, new_money, tuple(sorted(new_foods)), tuple(sorted(new_money_tiles)))

def reward(state, action, grid_width, grid_height):
    new_state = transition(state, action, grid_width, grid_height)
    if new_state is None:
        return -100  # Huge penalty for dying

    (new_position, new_health, new_money, foods, money_tiles) = new_state
    reward = -1  # Default movement penalty
    if new_position in foods:
        reward += 10  # Incentive for eating food
    if new_position in money_tiles:
        reward += 5  # Incentive for collecting money
    return reward

def value_iteration(states, actions, grid_width, grid_height, gamma=0.9, epsilon=1e-4):
    V = {state: 0 for state in states}
    policy = {state: None for state in states}

    while True:
        delta = 0
        for state in states:
            max_value = float('-inf')
            best_action = None
            for action in actions:
                if transition(state, action, grid_width, grid_height) is None:
                    continue
                value = reward(state, action, grid_width, grid_height) + gamma * V.get(
                    transition(state, action, grid_width, grid_height), 0
                )
                if value > max_value:
                    max_value = value
                    best_action = action
            delta = max(delta, abs(max_value - V[state]))
            V[state] = max_value
            policy[state] = best_action

        if delta < epsilon:
            break

    return policy

def adjust_grid(positions, foods, money_tiles, health, money):
    new_positions = set()
    new_health = {}
    new_money = {}

    states = [get_state(pos, health[pos], money[pos], foods, money_tiles) for pos in positions]
    actions = get_possible_actions()
    policy = value_iteration(states, actions, GRID_WIDTH, GRID_HEIGHT)

    for position in positions:
        state = get_state(position, health[position], money[position], foods, money_tiles)
        action = policy.get(state, (0, 0))  # Default to no movement if no action
        new_state = transition(state, action, GRID_WIDTH, GRID_HEIGHT)

        if new_state:
            new_positions.add(new_state[0])
            new_health[new_state[0]] = new_state[1]
            new_money[new_state[0]] = new_state[2]

    return new_positions, foods, money_tiles, new_health, new_money

def main():
    running = True
    playing = False
    update_freq = 120

    positions = set()
    foods = gen_resources(50)
    money_tiles = gen_resources(20)
    health = {pos: 10 for pos in positions}
    money = {pos: 5 for pos in positions}

    while running:
        clock.tick(FPS)

        if playing:
            positions, foods, money_tiles, health, money = adjust_grid(
                positions, foods, money_tiles, health, money
            )
            if len(foods) < 30:
                foods |= gen_resources(5)
            if len(money_tiles) < 10:
                money_tiles |= gen_resources(5)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col, row = x // TILE_SIZE, y // TILE_SIZE
                pos = (col, row)
                if pos in positions:
                    positions.remove(pos)
                    del health[pos]
                    del money[pos]
                else:
                    positions.add(pos)
                    health[pos] = 10
                    money[pos] = 5
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing

        screen.fill(GREY)
        draw_grid(positions, foods, money_tiles, health, money)
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
