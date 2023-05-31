import torch as th
import numpy as np

class RandomPlayer:
    def __init__(self, game) -> None:
        self.game = game
    
    def move(self, player):
        nonzero_indices = np.nonzero(self.game.admissable_moves)[0] 
        self.game.move(np.random.choice(nonzero_indices), player) 



class NNPlayer:
    def __init__(self, game, model) -> None:
        self.game = game
        self.model = model
        
    def move(self, player : int):
        if player == 1:
            flip = 1.0
        elif player == -1:
            flip = -1.0
        else:
            raise RuntimeError('Player must be -1 or 1!')
        state_input = th.tensor(self.game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
        model_out = self.model(state_input*flip)

        admissable_tens = th.tensor(self.game.admissable_moves)
        nonzero_indices = th.nonzero(admissable_tens)
        masked_tensor = model_out[0, nonzero_indices]
        max_index = nonzero_indices[th.argmax(masked_tensor)]
        choice = max_index.item()
        self.game.move(choice, player)