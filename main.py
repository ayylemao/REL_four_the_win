import numpy as np
from game import Game
from model import CoolModel
import torch as th
import matplotlib.pyplot as plt
import random

model = CoolModel()

loss_arr = []
nn_win = 0
rp_win = 0
win_hist = []
for i in range(0, 20000):

    game = Game()
    while True:
        # Player 2
        #state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
        #model_out = model(state_input*(-1))

        #admissable_tens = th.tensor(game.admissable_moves)
        #nonzero_indices = th.nonzero(admissable_tens)
        #masked_tensor = model_out[0, nonzero_indices]
        #max_index = nonzero_indices[th.argmax(masked_tensor)]
        #choice = max_index.item()
        #game.move(choice, -1)

        # RANDOM PLAYER
        nonzero_indices = np.nonzero(game.admissable_moves)[0] 
        game.move(np.random.choice(nonzero_indices), -1) 
        
        if game.concluded:
            if game.outcome == 1: 
                winner = 'Random Player'
                rp_win += 1
            elif game.outcome == 0:
                winner = 'Draw'
                print('Draw')
            break
        # =============================================================================

        # Player 1
        state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
        model_out = model(state_input)

        # Since model_out[i] can be zero but the only admissable move,
        # we need to be smarter about choosing the next move:
        admissable_tens = th.tensor(game.admissable_moves)
        nonzero_indices = th.nonzero(admissable_tens)
        masked_tensor = model_out[0, nonzero_indices]
        max_index = nonzero_indices[th.argmax(masked_tensor)]
        choice = max_index.item()

        game.move(choice, 1)
        if game.concluded:
            if game.outcome == 1: 
                winner = 'Neural Network'
                nn_win += 1
            elif game.outcome == 0:
                winner = 'Draw'
                print('Draw')
            break


    criterion = th.nn.MSELoss()
    optimizer = th.optim.SGD(model.parameters(), lr=0.0001)
    player_id = np.empty(42)
    player_id[::2] = -1
    player_id[1::2] = 1
    if game.outcome == 1:
        loss_sum = 0.0
        for turn in range(0, game.turn_counter, 2):
            # WINNER        
            # y_true 
            current_move = game.move_history[turn]
            if winner == 'Random Player':
                # Winner prop
                current_move = game.move_history[turn]
                y =  np.zeros(7)
                y[current_move[0]] = 0.9
                y = th.tensor(y).unsqueeze(0).float()
                current_state = th.tensor(game.history[turn]*(-1)).unsqueeze(dim=0).unsqueeze(dim=0).float()
                y_pred = model(current_state)
                loss = criterion(y_pred, y) 
                loss_sum += loss.item() 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Loser prop
                try:
                    current_move = game.move_history[turn+1]
                    y =  np.ones(7) * 0.7
                    y[current_move[0]] = 0.2
                    y = th.tensor(y).unsqueeze(0).float()
                    current_state = th.tensor(game.history[turn+1]).unsqueeze(dim=0).unsqueeze(dim=0).float()
                    y_pred = model(current_state)
                    loss = criterion(y_pred, y) 
                    loss_sum += loss.item() 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except IndexError:
                    pass
            elif winner == 'Draw':
                pass 
            
            if winner == 'Neural Network':
                current_move = game.move_history[turn]
                y =  np.ones(7) * 0.7
                y[current_move[0]] = 0.2
                y = th.tensor(y).unsqueeze(0).float()
                current_state = th.tensor(game.history[turn]*(-1)).unsqueeze(dim=0).unsqueeze(dim=0).float()
                y_pred = model(current_state)
                loss = criterion(y_pred, y) 
                loss_sum += loss.item() 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                try:
                    current_move = game.move_history[turn+1]
                    y =  np.zeros(7)
                    y[current_move[0]] = 1.0
                    y = th.tensor(y).unsqueeze(0).float()
                    current_state = th.tensor(game.history[turn+1]).unsqueeze(dim=0).unsqueeze(dim=0).float()
                    y_pred = model(current_state)
                    loss = criterion(y_pred, y) 
                    loss_sum += loss.item() 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except IndexError:
                    pass
            elif winner == 'Draw':
                pass  
            loss_arr.append(loss_sum/game.turn_counter)
        #print(f'Game {i} loss: {loss_sum/game.turn_counter:.3f}, Winner {winner}')
        if i % 1000 == 0:
            mean_loss = np.mean(loss_arr)
            print('Game {i}:')
            print(f'Mean Loss: {mean_loss:.3f}')
            print(f'NN Win: {nn_win/10:.3f}%')
            print(f'RP Win: {rp_win/10:.3f}%')
            print(f'')
            loss_arr = []
            nn_win = 0
            rp_win = 0
            win_hist.append(nn_win)



fig, ax = plt.subplots(1,1)
ax.plot(loss_arr)
fig.savefig('loss.png', dpi=300)
game = Game()

nonzero_indices = np.nonzero(game.admissable_moves)[0] 
game.move(np.random.choice(nonzero_indices), -1) 

state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
model_out = model(state_input)

admissable_tens = th.tensor(game.admissable_moves)
nonzero_indices = th.nonzero(admissable_tens)
masked_tensor = model_out[0, nonzero_indices]
max_index = nonzero_indices[th.argmax(masked_tensor)]
choice = max_index.item()
game.move(choice, 1)
game.state
model_out
model_out.shape
