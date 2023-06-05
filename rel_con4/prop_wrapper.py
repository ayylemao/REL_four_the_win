'''
Weight updater class used to train models.
'''

import torch as th
import numpy as np


class Propagation:
    '''
    Weight updater class used to train models.
    '''
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 win_bonus : float,
                 loss_penalty : float,
                 neutral_bonus : float,
                 non_starter_bonus : float
                 ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.neutral_bonus = neutral_bonus
        self.non_starter_bonus = non_starter_bonus 


    def rel_win_prop(self,
                     game,
                     player : int,
                     starter : bool,
                     ):
        '''Update weights of winner.'''
        loss_sum = 0.0
        if starter:
            init_turn = 0
            eff_win_bonus = self.win_bonus

        elif not starter:
            init_turn = 1
            eff_win_bonus = self.win_bonus + self.non_starter_bonus

        for turn in range(init_turn, game.turn_counter, 2):
            try:
                current_move = game.move_history[turn]
                y = np.zeros(7)
                y[current_move[0]] = eff_win_bonus
                y = th.tensor(y).unsqueeze(0).float()
                current_state = th.tensor(game.history[turn]*player).unsqueeze(0).unsqueeze(0).float()
                y_pred = self.model(current_state)
                loss = self.criterion(y_pred, y)
                loss_sum += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            except IndexError:
                pass
            return loss_sum / (game.turn_counter / 2)
    
    def rel_loss_prop(self,
                      game,
                      player : int,
                      starter : bool):
        '''Update weights of loser.'''
        loss_sum = 0.0
        if starter:
            init_turn = 0
            eff_loss_penalty = self.loss_penalty 
        elif not starter:
            init_turn = 1
            eff_loss_penalty = self.loss_penalty + self.non_starter_bonus
        for turn in range(init_turn, game.turn_counter, 2):
            try:
                current_move = game.move_history[turn]
                y = np.ones(7) * self.neutral_bonus
                y[current_move[0]] = eff_loss_penalty
                y = th.tensor(y).unsqueeze(0).float()
                current_state = th.tensor(game.history[turn]*player).unsqueeze(0).unsqueeze(0).float()
                y_pred = self.model(current_state)
                loss = self.criterion(y_pred, y)
                loss_sum += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            except IndexError:
                pass
        return loss_sum / (game.turn_counter / 2)
        