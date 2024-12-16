import pygame
import random
import numpy as np
from typing import Dict, Tuple, List, Any

class MDPCellularSimulation:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height

        # State Space
        self.states = {}  # {(x,y): {'health': int, 'money': int}}
        
        # Action Space
        self.actions = [
            (0, 1),   # Move Down
            (0, -1),  # Move Up
            (1, 0),   # Move Right
            (-1, 0),  # Move Left
            (0, 0)    # Stay
        ]

        # Resource Locations
        self.foods = set()
        self.money_tiles = set()

        # MDP Parameters
        self.GAMMA = 0.9  # Discount factor
        self.LEARNING_RATE = 0.1
        self.EPSILON = 0.1  # Exploration rate

        # Q-Table for learning optimal policy
        self.q_table = {}

    def generate_resources(self, count: int) -> set:
        """Generate random resource positions."""
        resources = set()
        while len(resources) < count:
            pos = (random.randint(0, self.width - 1), 
                   random.randint(0, self.height - 1))
            resources.add(pos)
        return resources

    def get_state_key(self, position: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Create a hashable state representation."""
        state = self.states.get(position, {'health': 0, 'money': 0})
        return (
            position[0], 
            position[1], 
            state['health'], 
            state['money']
        )

    def is_valid_state(self, position: Tuple[int, int]) -> bool:
        """Check if a state is within grid bounds."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def compute_reward(self, 
                    state: Tuple[int, int], 
                    action: Tuple[int, int], 
                    next_state: Tuple[int, int]) -> float:
        """
        Compute reward based on state transition.
        """
        reward = 0

        # Movement cost
        if action != (0, 0):  # Penalize movement slightly
            reward -= 0.2

        # Health preservation bonus
        current_health = self.states[state]['health']
        reward += current_health / 10

        # Food collection
        if next_state in self.foods:
            reward += 10  # Higher reward for food

        # Money collection
        if next_state in self.money_tiles:
            reward += 5  # Reward for collecting money

        return reward


    def q_learning_update(self, 
                           state: Tuple[int, int], 
                           action: Tuple[int, int], 
                           reward: float, 
                           next_state: Tuple[int, int]):
        """Q-learning update rule."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.actions}

        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())
        
        new_q = current_q + self.LEARNING_RATE * (
            reward + self.GAMMA * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q

    def choose_action(self, state: Tuple[int, int]) -> Tuple[int, int]:
        """
        Epsilon-greedy action selection.
        Balances exploration and exploitation.
        """
        state_key = self.get_state_key(state)

        # Initialize Q-values if not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}

        # Exploration
        if random.random() < self.EPSILON:
            return random.choice(self.actions)

        # Exploitation
        max_q = max(self.q_table[state_key].values())
        best_actions = [a for a, q in self.q_table[state_key].items() if q == max_q]
        return random.choice(best_actions)  # Break ties randomly

    def step(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
        """
        Perform a step in the environment.
        Returns next state and reward.
        """
        x, y = state
        dx, dy = action

        # Compute next state
        next_state = (x + dx, y + dy)

        # Validate next state
        if not self.is_valid_state(next_state):
            next_state = state

        # Initialize the next state if not already present
        if next_state not in self.states:
            self.states[next_state] = {'health': 10, 'money': 0}  # Default values

        # Resource consumption should modify the current state's attributes
        if next_state in self.foods:
            self.states[state]['health'] += 5  # Increase health of the current cell
            self.foods.remove(next_state)      # Remove the food tile

        if next_state in self.money_tiles:
            self.states[state]['money'] += 1  # Increase money of the current cell
            self.money_tiles.remove(next_state)  # Remove the money tile

        # Compute reward
        reward = self.compute_reward(state, action, next_state)

        return next_state, reward

    def train(self, episodes: int = 500):
        self.foods = self.generate_resources(200)
        self.money_tiles = self.generate_resources(100)

        for episode in range(episodes):
            # Decay EPSILON to focus more on exploitation as training progresses
            self.EPSILON = max(0.01, self.EPSILON * 0.99)

            state = random.choice(list(self.states.keys()))
            total_reward = 0
            self.replenish_resources(food_threshold=10, money_threshold=5)

            for _ in range(100):  # Limit steps per episode
                action = self.choose_action(state)
                next_state, reward = self.step(state, action)
                self.q_learning_update(state, action, reward, next_state)
                total_reward += reward
                state = next_state

            print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")

    def replenish_resources(self, food_threshold: int = 10, money_threshold: int = 5):
        """
        Replenish resources if their count drops below a specified threshold.
        """
        if len(self.foods) < food_threshold:
            new_foods = self.generate_resources(food_threshold - len(self.foods))
            self.foods.update(new_foods)

        if len(self.money_tiles) < money_threshold:
            new_money_tiles = self.generate_resources(money_threshold - len(self.money_tiles))
            self.money_tiles.update(new_money_tiles)

    def reset_environment(self):
        """Reset the simulation environment."""
        self.states.clear()
        self.foods = self.generate_resources(50)
        self.money_tiles = self.generate_resources(20)

        # Re-add a few initial cells
        for _ in range(3):  # Limit to 3 initial cells
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            self.states[pos] = {
                'health': 10,
                'money': 5
            }


# (Assume the previous MDPCellularSimulation class is defined before this)

def draw_grid(screen, sim, tile_size):
    """Visualize the grid, resources, and cells."""
    # Clear background
    screen.fill((128, 128, 128))

    # Draw food tiles
    for food in sim.foods:
        pygame.draw.rect(screen, (150, 75, 0), (food[0] * tile_size, food[1] * tile_size, tile_size, tile_size))
    
    # Draw money tiles
    for money in sim.money_tiles:
        pygame.draw.rect(screen, (0, 255, 0), (money[0] * tile_size, money[1] * tile_size, tile_size, tile_size))

    # Draw cells and their stats
    for pos, state in sim.states.items():
        x, y = pos
        pygame.draw.rect(screen, (255, 255, 255), (x * tile_size, y * tile_size, tile_size, tile_size))
        
        # Display health and money
        font = pygame.font.SysFont("Arial", 12)
        text = font.render(f"H:{state['health']} M:{state['money']}", True, (0, 0, 0))
        screen.blit(text, (x * tile_size + 5, y * tile_size + 5))

def main():
    pygame.init()

    # Screen and grid settings
    WIDTH, HEIGHT = 1000, 1000
    TILE_SIZE = 50
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MDP Cellular Simulation")
    clock = pygame.time.Clock()

    # Simulation environment
    sim = MDPCellularSimulation(width=WIDTH // TILE_SIZE, height=HEIGHT // TILE_SIZE)
    
    # Initialize some initial states before training
# Initialize some initial states before training
    for _ in range(3):  # Limit to 3 initial cells
        pos = (random.randint(0, sim.width - 1), random.randint(0, sim.height - 1))
        sim.states[pos] = {
            'health': 10,
            'money': 5
        }

    
    # Initial training
    sim.train(episodes=500)

    running = True
    playing = False
    update_counter = 0
    UPDATE_FREQUENCY = 60

    # Rest of the code remains the same...

    # Rest of the code remains the same...

    while running:
        clock.tick(60)

        if playing:
            update_counter += 1
            if update_counter >= UPDATE_FREQUENCY:
                update_counter = 0
                
                # Simulate one step for each cell
                new_states = {}
                for pos, state in list(sim.states.items()):
                    # Choose action based on learned policy
                    action = sim.choose_action(pos)
                    
                    # Take step
                    next_pos, reward = sim.step(pos, action)
                    
                    # Update state if cell survives
                    if state['health'] > 0:
                        new_states[next_pos] = {
                            'health': max(0, state['health'] - 1),
                            'money': state['money']
                        }

                # Update simulation states
                sim.states = new_states

                sim.replenish_resources(food_threshold=10, money_threshold=5)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                col, row = x // TILE_SIZE, y // TILE_SIZE
                cell_pos = (col, row)

                # Toggle cell existence
                if cell_pos in sim.states:
                    del sim.states[cell_pos]
                else:
                    sim.states[cell_pos] = {
                        'health': 10,  # Initial health
                        'money': 5     # Initial money
                    }

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                if event.key == pygame.K_c:
                    sim.reset_environment()
                    # Re-add some initial cells
                    for _ in range(5):
                        pos = (random.randint(0, sim.width - 1), random.randint(0, sim.height - 1))
                        sim.states[pos] = {
                            'health': 10, 
                            'money': 5
                        }

        # Draw the grid
        draw_grid(screen, sim, TILE_SIZE)
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()