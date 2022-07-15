import numpy as np
import pygame
import random
from collections import namedtuple
from enum import Enum

pygame.init()
font = pygame.font.SysFont('lato', 30)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

# reset
# reward
# play_step(action) -> Direction
# move
# frame iteration
# is_collision

class SnekGeymAI:
    def __init__(self, w = 800, h = 600):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, n_games=0, high_score=0):
        # init game stats
        self.n_games = n_games + 1
        self.high_score = high_score
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE

        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        # hits boundary
        if pt is None:
            pt = self.head

        if pt.x > self.w - BLOCK_SIZE \
            or pt.x < 0 or pt.y > self.h - BLOCK_SIZE \
            or pt.y < 0:
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False

    def _update_ui(self, gens = 1):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        text1 = font.render("Generation: " + str(gens), True, WHITE)
        text2 = font.render("High Score: " + str(self.high_score), True, WHITE)
        self.display.blit(text, [10, 5])
        self.display.blit(text1, [10, 25])
        self.display.blit(text2, [660, 5])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # straight
            new_dir = clock_wise[idx] # no change, go straight
        elif np.array_equal(action, [0, 1, 0]): # right
            next_idx = (idx + 1) % 4 
            new_dir = clock_wise[next_idx] # right r -> d -> l -> u
        else: # [0, 0, 1], left
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left l -> d -> r -> u

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def play_step(self, action):
        # 1. correct user inputs
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move user
        self._move(action) # update head
        self.snake.insert(0, self.head) 

        # 3. check collision
        game_over=False
        reward = 0
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over=True
            reward = -10
            return game_over, self.score, reward

        # 4. place new food
        if self.food == self.head:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui
        self._update_ui(self.n_games)
        self.clock.tick(SPEED)

        # 6. return game over, score
        return game_over, self.score, reward


