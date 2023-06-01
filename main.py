import numpy as np
from rel_con4 import Game, CoolModel, Propagation
from rel_con4.agents import RandomPlayer, NNPlayer
import torch as th
import matplotlib.pyplot as plt
import random

loss_move = -0.8
win_move = +0.6
neutral_move = 0.2
nonstarter_bonus = 0.2


model = CoolModel()
criterion = th.nn.MSELoss()
optimizer = th.optim.SGD(model.parameters(), lr=0.0001)
prop = Propagation(model=model, 
                   criterion=criterion, 
                   optimizer=optimizer,
                   win_bonus=win_move,
                   loss_penalty=loss_move,
                   neutral_bonus=neutral_move,
                   non_starter_bonus=nonstarter_bonus)

loss_arr = []
nn_win = 0
rp_win = 0
win_hist = []
PLAYER_A = 1
PLAYER_B = -1
STARTER_DICT = {PLAYER_A : True, PLAYER_B : False}
for i in range(0, 100000):
    game = Game()
    rndplayer = RandomPlayer(game=game)
    nnplayer = NNPlayer(game=game, model=model)

    
    # Game Loop
    # ================================= Game Loop ===================================
    while True:
        # Player 1
        rndplayer.move(player=PLAYER_A)
        
        # win check 
        if game.concluded:
            if game.outcome != 'draw': 
                winner = PLAYER_A
                loser = PLAYER_B
                rp_win += 1
            elif game.outcome == 'draw':
                winner = 'Draw'
                print('Draw')
            break

        # Player 2
        nnplayer.move(player=PLAYER_B) 
        
        
        if game.concluded:
            if game.outcome != 'draw': 
                winner = PLAYER_B
                loser = PLAYER_A
                nn_win += 1
            elif game.outcome == 'draw':
                winner = 'Draw'
                print('Draw')
            break
    # =================================================================================

    # Reenforcement Part
    if winner == PLAYER_A:
        win_loss = prop.rel_win_prop(game=game,
                                     player=winner,
                                     starter=STARTER_DICT[PLAYER_A])
    elif winner == PLAYER_B:
        win_loss = prop.rel_win_prop(game=game,
                                     player=winner,
                                     starter=STARTER_DICT[PLAYER_B])
    #print(game.state)

    #loss_sum = 0.0
    #for turn in range(0, 5, 2):
    #    # WINNER        
    #    # y_true 
    #    current_move = game.move_history[turn]
    #    if winner == 'Player A':
    #        # Winner prop
    #        current_move = game.move_history[turn]
    #        y =  np.zeros(7)
    #        y[current_move[0]] = (win_move + nonstarter_bonus)
    #        y = th.tensor(y).unsqueeze(0).float()
    #        current_state = th.tensor(game.history[turn]*(-1)).unsqueeze(dim=0).unsqueeze(dim=0).float()
    #        y_pred = model(current_state)
    #        loss = criterion(y_pred, y) 
    #        loss_sum += loss.item() 
    #        optimizer.zero_grad()
    #        loss.backward()
    #        optimizer.step()
    #        # Loser prop
    #        try:
    #            current_move = game.move_history[turn+1]
    #            y =  np.ones(7) * neutral_move
    #            y[current_move[0]] = loss_move
    #            y = th.tensor(y).unsqueeze(0).float()
    #            current_state = th.tensor(game.history[turn+1]).unsqueeze(dim=0).unsqueeze(dim=0).float()
    #            y_pred = model(current_state)
    #            loss = criterion(y_pred, y) 
    #            loss_sum += loss.item() 
    #            optimizer.zero_grad()
    #            loss.backward()
    #            optimizer.step()
    #        except IndexError:
    #            pass
    #    elif winner == 'Draw':
    #        pass 
    #    
    #    if winner == 'Player B':
    #        current_move = game.move_history[turn]
    #        y =  np.ones(7) * neutral_move
    #        y[current_move[0]] = (loss_move + nonstarter_bonus)
    #        y = th.tensor(y).unsqueeze(0).float()
    #        current_state = th.tensor(game.history[turn]*(-1)).unsqueeze(dim=0).unsqueeze(dim=0).float()
    #        y_pred = model(current_state)
    #        loss = criterion(y_pred, y) 
    #        loss_sum += loss.item() 
    #        optimizer.zero_grad()
    #        loss.backward()
    #        optimizer.step()
    #        try:
    #            current_move = game.move_history[turn+1]
    #            y =  np.zeros(7)
    #            y[current_move[0]] = win_move
    #            y = th.tensor(y).unsqueeze(0).float()
    #            current_state = th.tensor(game.history[turn+1]).unsqueeze(dim=0).unsqueeze(dim=0).float()
    #            y_pred = model(current_state)
    #            loss = criterion(y_pred, y) 
    #            loss_sum += loss.item() 
    #            optimizer.zero_grad()
    #            loss.backward()
    #            optimizer.step()
    #        except IndexError:
    #            pass
    #    elif winner == 'Draw':
    #        pass  
    #    loss_arr.append(loss_sum/game.turn_counter)
    ##if i % 1 == 100:
    #    #recent_loss = loss_arr[-11:-1]
    #    #print(f'Game {i} loss: {np.mean(recent_loss):.3f}, Winner: {winner}')
    #if i % 100 == 0:
    #    mean_loss = np.mean(loss_arr)
    #    print(f'Game {i}:')
    #    print(f'Mean Loss: {mean_loss:.3f}')
    #    print(f'NN Win: {nn_win:.3f}%')
    #    print(f'RP Win: {rp_win:.3f}%')
    #    print(f'Example State:')
    #    print(game.state)
    #    loss_arr = []
    #    win_hist.append(nn_win)
    #    nn_win = 0
    #    rp_win = 0



fig, ax = plt.subplots(1,1)
ax.plot(win_hist)
fig.savefig('loss.png', dpi=300)

th.save(model.state_dict(), 'model001.torch')

game = Game()
game.move(3, -1)

state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()

model_out = model(state_input)
model_out


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