import numpy as np

class Game:
    def __init__(self) -> None:
        self.state : np.ndarray = np.zeros([6, 7])
        self.history : dict = {0 : self.state.copy()}
        self.move_history : list = []
        self.concluded : bool = False
        self.turn_counter : int = 0
        self.outcome : int = None # None for non concluded game, 0 for draw, 1 for winning game
        self.admissable_moves : np.ndarray = np.ones(7) # can play into col i if array[i] == 1
        
        
    def move(self, column : int, player : int):
        # check if move is admissable
        if self.state[0, column] != 0:
           raise RuntimeError(f'column {column} already full!')  
        if not (player == -1 or player == 1):
            raise RuntimeError(f'player must be -1 or 1 and not {player}!')

        # get top most stone index in column
        top_stone_ix : int = (self.state[:, column] != 0).argmax()

        # set stone
        if top_stone_ix == 0:
            self.state[-1, column] = player
        elif top_stone_ix != 0:
            self.state[top_stone_ix -1, column] = player 
        # increment turn counter
        self.turn_counter += 1
        self.move_history.append((column, player))
        self.update_history()
        self.winner()
        self.draw()

        # update admissable moves
        if not np.any(self.state[:, column] == 0):
            self.admissable_moves[column] = 0

    def update_history(self) -> None:
        self.history[self.turn_counter] = self.state.copy()
    
    def winner(self) -> None: 
        # divide into rolling 4x4 sub matrices
        v : np.ndarray = np.lib.stride_tricks.sliding_window_view(self.state, [4, 4])
        # iterate over submatrix indices i, j
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                # check for diag win con
                if np.trace(v[i, j, :, :]) == 4 or np.trace(v[i, j, :, :]) == -4:
                    self.concluded = True
                    self.outcome = 1
                    return
                # check for anti-diag wincon
                if np.trace(np.fliplr(v[i, j, :, :])) == 4 or np.trace(np.fliplr(v[i, j, :, :])) == -4:
                    self.concluded = True 
                    self.outcome = 1
                    return

                for k in range(4):
                    # check for vertical wincon
                    if np.sum(v[i, j, :, k]) == 4 or np.sum(v[i, j, :, k]) == -4:
                        self.concluded = True
                        self.outcome = 1
                        return
                    # check for horizontal wincon
                    if np.sum(v[i, j, k, :]) == 4 or np.sum(v[i, j, k, :]) == -4:
                        self.concluded = True
                        self.outcome = 1
                        return
        
    def draw(self) -> None:
        if not np.any(self.state == 0):
            self.outcome = 0
            self.concluded = True


