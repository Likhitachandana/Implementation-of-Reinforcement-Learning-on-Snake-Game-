from game import Snake
from agent import Agent
from settings import *
from enum import Enum
import multiprocessing as mp
from multiprocessing import Pool, Process
import os
import json
import random

class Windows(Enum):
    W14 = (20, 20, 1, 1)


class Game:

    def __init__(self, lv=1):
        self.lv = lv
        self.awake()


    def awake(self):
        processes = []

        file = open('par_lev.json', 'r')
        json_pars = json.load(file)
        file.close()
        for window in Windows:


            pars = json_pars.get(window.name, [{}])

            if window.name == "W" + str(self.lv):
                a, b, c, d = window.value
                a, b = (set_size(a), set_size(b))
                index = 0
                for i in range(c):
                    for j in range(d):
                        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100+(a+b)*i,100+(a+b)*j)
                        if index < len(pars) and len(pars) > 0:
                            p  = Process(target=self.train, args=(a, b, pars[index]))
                        elif len(pars) >= index:
                            p  = Process(target=self.train, args=(a, b, {}))
                        else:
                            p  = Process(target=self.train, args=(a, b, pars[0]))
                        p.start()
                        processes.append(p)
                        index += 1
                break
        for p in processes:
            # join every processors
            p.join()

    def save_to_file(self, path, game_num, score, record):
  
        file = open(path, "a+")
        file.write("%s %s %s\n" % (game_num, score, record))
        file.close()

    def train(self, n, m, pars):
    
        record = 0
        game = Snake(n, m, pars.get('n_food', None))
        agent = Agent(game, pars)

        while True:
            state_old = game.get_state()

            final_move = agent.get_action(state_old)

            reward, done, score = game.play_step(final_move, pars)
            state_new = game.get_state()

            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            agent.remember(state_old, final_move, reward, state_new, done)

            if pars.get('num_games', DEFAULT_END_GAME_POINT) != -1:
                if agent.n_games > pars.get('num_games', DEFAULT_END_GAME_POINT):
                    quit()
                    break

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                decrease_probability = pars.get('decrease_food_chance', DECREASE_FOOD_CHANCE)
                if (game.n_food > 1) and (random.random() < decrease_probability):
                    game.n_food -= 1
                
                print('Game number', agent.n_games, 'Score obtained', score, 'Highest_Record:', record)
                self.save_to_file(f"graphs/{pars.get('graph', 'test')}.txt", agent.n_games, score, record)    
    

if __name__ == "__main__":
    Game(14)