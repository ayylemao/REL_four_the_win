'''
Player classes (random or network) to play connect 4.
'''

import torch as th
import numpy as np
import random

class RandomPlayer:
    '''
    The random player plays each choice with equal probability and cannot learn.
    '''
    def __init__(self, game) -> None:
        self.game = game
    
    def make_move(self, player):
        nonzero_indices = np.nonzero(self.game.admissable_moves)[0]
        self.game.process_move(np.random.choice(nonzero_indices), player)



class NNPlayer:
    '''
    The network player plays each move by using its model's output and by only allowing admissible moves. 
    '''
    def __init__(self, game, model, rnd_move_chance=None) -> None:
        self.game = game
        self.model = model
        self.rnd_move_chance = rnd_move_chance
        
    def make_move(self, player : int):
        rnd_move = False 
        # network is always player 1, so if we are letting it play as player -1 we need to adjust for it
        if player == 1:
            flip = 1.0
        elif player == -1:
            flip = -1.0
        else:
            raise RuntimeError('Player must be -1 or 1!')
        
        if self.rnd_move_chance is not None:
            if random.uniform(a=0.0, b=1.0) <= self.rnd_move_chance:
                rnd_move = True
            else:
                rnd_move = False
            
        if rnd_move:
            nonzero_indices = np.nonzero(self.game.admissable_moves)[0]
            self.game.process_move(np.random.choice(nonzero_indices), player)
        else: 
            # get probabilities on moves from model
            game_state = th.tensor(self.game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
            move_probabilities = self.model(game_state*flip)

            # get admissible choices
            admissable_tens = th.tensor(self.game.admissable_moves)
            nonzero_indices = th.nonzero(admissable_tens)
            masked_tensor = move_probabilities[0, nonzero_indices]
            max_index = nonzero_indices[th.argmax(masked_tensor)]
            move = max_index.item()
            
            # make choice in the game
            self.game.process_move(move, player)
        