'''
Class representing the game connect 4.
'''

import numpy as np

class Game:
    '''
    Class representing the game connect 4.
    '''
    def __init__(self) -> None:
        self.state : np.ndarray = np.zeros([6, 7])
        self.history : dict = {0 : self.state.copy()}
        self.move_history : list = []
        self.concluded : bool = False
        self.turn_counter : int = 0
        self.outcome : str = None # None for non concluded game, 0 for draw, 1 for winning game
        self.admissable_moves : np.ndarray = np.ones(7) # can play into col i if array[i] == 1
       
     
    def process_move(self, column : int, player : int):
        '''
        Update the game with a players moves.
        '''  
        # check if move is admissable
        if self.state[0, column] != 0:
            raise RuntimeError(f'column {column} already full!')
        if not player in(-1, 1):
            raise RuntimeError(f'player must be -1 or 1 and not {player}!')
        self.turn_counter += 1
        self.update_game(column, player)
        self.update_move_history(column, player)
        self.update_history()
        self.check_winner()
        self.check_draw()

        # update admissable moves
        if not np.any(self.state[:, column] == 0):
            self.admissable_moves[column] = 0
        
    def update_game(self, column, player) -> None:
        '''
        Puts the stone of the player on the state.
        '''
        # get top most stone index in column
        top_stone_ix : int = (self.state[:, column] != 0).argmax()
        # set stone
        if top_stone_ix == 0: # if column empty
            self.state[-1, column] = player
        elif top_stone_ix != 0:
            self.state[top_stone_ix-1, column] = player       

    def update_history(self) -> None:
        '''
        Adds the game state of turn x to the history dictionary with key x.
        '''
        self.history[self.turn_counter] = self.state.copy()

    def update_move_history(self, column, player) -> None:
        '''
        Adds the move of turn x to the move history list.
        '''
        self.move_history.append((column, player))

    def check_winner(self) -> None:
        '''
        Checks with the current game state if the game has a winner.
        '''
        # divide into rolling 4x4 sub matrices
        v : np.ndarray = np.lib.stride_tricks.sliding_window_view(self.state, [4, 4])
        # iterate over submatrix indices i, j
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                # check for diagonal win condition
                if np.trace(v[i, j, :, :]) == 4 or np.trace(v[i, j, :, :]) == -4:
                    self.concluded = True
                    self.outcome = 'diag'
                    return
                # check for anti-diag win condition
                if np.trace(np.fliplr(v[i, j, :, :])) == 4 or np.trace(np.fliplr(v[i, j, :, :])) == -4:
                    self.concluded = True
                    self.outcome = 'diag'
                    return
                
                for k in range(4):
                    # check for vertical win condition
                    if np.sum(v[i, j, :, k]) == 4 or np.sum(v[i, j, :, k]) == -4:
                        self.concluded = True
                        self.outcome = 'vert'
                        return
                    # check for horizontal win condition
                    if np.sum(v[i, j, k, :]) == 4 or np.sum(v[i, j, k, :]) == -4:
                        self.concluded = True
                        self.outcome = 'hori'
                        return
      
    def check_draw(self) -> None:
        '''
        Checks with the current game state if the game has ended with a draw, i.e. the game state is full.
        '''
        if not np.any(self.state == 0):
            self.concluded = True
            self.outcome = 'draw'

