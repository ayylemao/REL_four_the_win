import torch as th
import numpy as np
from game import Game


class RandomPlayer:
    def __init__(self, game) -> None:
        self.game = game
    
    def move(self, player):
        nonzero_indices = np.nonzero(self.game.admissable_moves)[0] 
        self.game.move(np.random.choice(nonzero_indices), player) 

