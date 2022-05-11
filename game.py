import pygame
from pygame.locals import *
from pygame.color import Color
import numpy as np
from enum import Enum
from settings import *


class Dir(Enum):
    L = 0
    R = 1
    U = 2
    D = 3


class DefaultImediateReward:
    collision_with_wall  = -10
    collision_With_itself = -10
    scored=10
    loop=-10
    moving_away_=0
    default_value_for_moving_Closer=0
    empty_Cell=0
    veryfar_from_food=0
    mid_dist_from_food=0
    far_from_food=0
    close_to_food=0

class Snake:
   
    TITLE = "Snake Game using Reinforcement Learning"

    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT, n_food=None):
        
        self.size = SIZE
        self.speed = DEFAULT_SPEED
        self.w = w
        self.h = h
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(Snake.TITLE)
        self.clock = pygame.time.Clock()

        self.n_food = DEFAULT_N_FOOD if n_food == None else n_food
        self.reset()
        self.frame = 0

    def reset(self):
        
        self.dir = Dir.R

        x, y = self.head = (self.w/2, self.h/2)
        self.body = [self.head, (x-self.size, y),(x-(2*self.size), y)]

        self.score = 0
        self.le = 3
        self.food = []
        self.food.clear()

        self.gen_food()
        self.frame = 0

    def gen_food(self):
        
        x = random.randint(0, (self.w-self.size )//self.size )*self.size
        y = random.randint(0, (self.h-self.size )//self.size )*self.size

        
        self.food.append((x, y))

        
        if self.food[-1] in self.body:
            self.gen_food()
            
        if self.n_food == 1 and len(self.food) > 1:
            del self.food[1:]

    def get_state(self):
      
        x, y = self.head
        fx, fy = zip(*self.food)
        
        fx, fy = np.array(fx), np.array(fy)

        
        point_l = (x - self.size, y)
        point_r = (x + self.size, y)
        point_u = (x, y - self.size)
        point_d = (x, y + self.size)

        
        dir_l = self.dir == Dir.L
        dir_r = self.dir == Dir.R
        dir_u = self.dir == Dir.U
        dir_d = self.dir == Dir.D

        state = [
            
            (dir_r and self.collision(point_r)) or
            (dir_l and self.collision(point_l)) or
            (dir_u and self.collision(point_u)) or
            (dir_d and self.collision(point_d)),

    
            (dir_u and self.collision(point_r)) or
            (dir_d and self.collision(point_l)) or
            (dir_l and self.collision(point_u)) or
            (dir_r and self.collision(point_d)),

            
            (dir_d and self.collision(point_r)) or
            (dir_u and self.collision(point_l)) or
            (dir_r and self.collision(point_u)) or
            (dir_l and self.collision(point_d)),

            
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            
            any(fx < x),  
            any(fx > x),  
            any(fy < y),  
            any(fy > y)   
            ]
                
        return np.array(state, dtype=int)

    def play_step(self, action, kwargs={None:None}):
        
        self.frame += 1

        
        if len(self.food) < self.n_food:
            self.gen_food()

       
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit()
                quit()
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    pygame.quit()
                elif ev.key == K_q:
                    pygame.quit()
                quit()

      
        self.move(action)

     
        self.body.insert(0, self.head)

       
        reward = kwargs.get('far_range_from_food', DefaultImediateReward.veryfar_from_food)
        terminal = False

        
        if self.collision():
           
            terminal = True
            reward = kwargs.get('wall_Collision', DefaultImediateReward.collision_with_wall)
            return reward, terminal, self.score

        
        if self.frame > kwargs.get('kill_frame', DEFAULT_KILL_FRAME)*len(self.body):
          
            terminal = True
            reward = kwargs.get('loop_with_itself',  DefaultImediateReward.loop)
            return reward, terminal, self.score

       
        for fx, fy in self.food:
            if self.head == (fx, fy):
                self.score += 1
                self.le += 1
                reward = kwargs.get('rewards_with_food',  DefaultImediateReward.scored)
                del self.food[self.food.index((fx, fy))]
                self.gen_food()

      
        if len(self.body) > self.le:
            self.body.pop()

        self.update()
      
        self.clock.tick(self.speed)

        distance = np.array(self.distance())//self.size 

       
        if any(CLOSE_RANGE[0] <= distance) and  any(distance < CLOSE_RANGE[1]):
            reward = kwargs.get('close_range', DefaultImediateReward.close_to_food)
            return reward, terminal, self.score
       
        elif any(FAR_RANGE[0] <= distance) and any(distance <FAR_RANGE[1]):
            reward = kwargs.get('far_range', DefaultImediateReward.far_from_food) 
            return reward, terminal, self.score

        
        if kwargs.get('is_dir', False):
            for fx, fy in self.food:
                x, y = self.head
                if self.dir == Dir.R or self.dir == Dir.L:
                    if  x > fx and self.dir == Dir.R:
                        reward = kwargs.get("moving_away", DefaultImediateReward.moving_away_) 
                    elif x < fx and self.dir == Dir.L:
                        reward = kwargs.get("moving_away", DefaultImediateReward.moving_away_) 
                    elif x > fx and self.dir == Dir.L:
                        reward = kwargs.get("moving_closer", DefaultImediateReward.default_value_for_moving_Closer) 
                    if x < fx and self.dir == Dir.R:
                        reward = kwargs.get("moving_closer", DefaultImediateReward.default_value_for_moving_Closer)
                if self.dir == Dir.U or self.dir == Dir.D:
                    if y > fy and self.dir == Dir.D:
                        reward = kwargs.get("moving_away", DefaultImediateReward.moving_away_)
                    elif y < fy and self.dir == Dir.U:
                        reward = kwargs.get("moving_away", DefaultImediateReward.moving_away_) 
                    elif y > fy and self.dir == Dir.U:
                        reward = kwargs.get("moving_closer", DefaultImediateReward.default_value_for_moving_Closer) 
                    if y < fy and self.dir == Dir.D:
                        reward = kwargs.get("moving_closer", DefaultImediateReward.default_value_for_moving_Closer) 
                return reward, terminal, self.score

        return reward, terminal, self.score

    
    def distance(self):
      
        distance_val = []
        p, q = self.head
        for fx, fy in self.food:
            distance_val.append(((fx - p)**2 + (fy - q)**2)**0.5)
        return distance_val
    def draw(self):
      
        for l, k in self.body:
            pygame.draw.rect(self.screen, Color('red'), (l, k, self.size, self.size))
        for m, n in self.food:
            pygame.draw.rect(self.screen, Color('yellow'), (m, n, self.size, self.size))

    def collision(self, pt=None):
        
        if not pt:
            x, y = self.head
        else:
            x, y = pt
        if x > self.w - self.size or x < 0 or y > self.h - self.size or y < 0:
            return True
        if (x, y) in self.body[1:]:
            return True

        return False
    def update(self):
        
        self.screen.fill(Color('Black'))
        self.draw()
        pygame.display.update()

    def move(self, action):
        
        clock_wise = [Dir.R, Dir.D, Dir.L, Dir.U]
        idx = clock_wise.index(self.dir)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] 
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] 
        else: 
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] 

        self.dir = new_dir

        
        x, y = self.head

       
        if self.dir == Dir.R:
            x += self.size
        elif self.dir == Dir.L:
            x -= self.size
        elif self.dir == Dir.U:
            y -= self.size
        elif self.dir == Dir.D:
            y += self.size

        self.head = (x, y)

            
    def control(self):
     
        keys = pygame.key.get_pressed()

        if keys[K_LEFT]:
            if self.dir != Dir.L:
                action = np.array([0, 0, 1])
            else:
                action = np.array([1, 0, 0])

        elif keys[K_RIGHT]:
            if self.dir != Dir.R:
                action = np.array([0, 1, 0])
            else:
                action = np.array([1, 0, 0])
        elif keys[K_UP]:
            if self.dir != Dir.R:
                action = np.array([0, 1, 0])
            else:
                action = np.array([0, 0, 1])
        elif keys[K_DOWN]:
            if self.dir != Dir.L:
                action = np.array([0, 1, 0])
            else:
                action = np.array([0, 0, 1])

        else:
            action = np.array([1, 0, 0])

        return action