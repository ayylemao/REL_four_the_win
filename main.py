import numpy as np
from game import Game
from model import CoolModel
import torch as th
import matplotlib.pyplot as plt


model = CoolModel()

loss_arr = []
for i in range(0, 1000):
    game = Game()
    while True:
        # Player 1
        state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
        model_out = model(state_input)
        choice = (th.tensor(game.admissable_moves) * model_out).argmax().item()
        game.move(choice, 1)
        if game.concluded:
            if game.outcome == 1: 
                pass
            elif game.outcome == 0:
                print('Draw')
            break
        # Player 2
        state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
        model_out = model(state_input*(-1))
        choice = (th.tensor(game.admissable_moves) * model_out).argmax().item()
        game.move(choice, -1)
        if game.concluded:
            if game.outcome == 1: 
                pass
            elif game.outcome == 0:
                print('Draw')
            break



    criterion = th.nn.MSELoss(reduction='sum')
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)
    if game.outcome == 1:
        loss_sum = 0.0
        for turn in range(game.turn_counter, 2, -2):
            # WINNER        
            # y_true 
            current_move = game.move_history[turn-1]

            y =  np.zeros(7)
            y[current_move[0]] = 1
            y = th.tensor(y).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).float()

            # y_pred 
            current_state = th.tensor(game.history[turn-2]).unsqueeze(dim=0).unsqueeze(dim=0).float()
            y_pred = model(current_state)

            loss = criterion(y_pred, y)
            
            loss_sum += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOSER
            # y_true 
            #current_move = game.move_history[turn-2]

            #y =  np.ones(7)
            #y[current_move[0]] = 0
            #y = y/np.sum(y)
            #y = th.tensor(y).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).float()

            ## y_pred 
            #current_state = th.tensor(game.history[turn-3]).unsqueeze(dim=0).unsqueeze(dim=0).float()
            #y_pred = model(current_state)

            #loss = criterion(y_pred, y)
 
            #loss_sum += loss.item()

            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
        loss_arr.append(loss_sum/game.turn_counter)
        print(f'Game {i} loss: {loss_sum/game.turn_counter:.3f}')


#loss_arr
#fig, ax = plt.subplots(1,1)
#ax.plot(loss_arr)
#fig.savefig('loss.png', dpi=300)
#
#game = Game()
#
#loss_arr[0]
#
#game.move(0, 1)
#game.state
#state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
#model_out = model(state_input*-1)
#choice = (th.tensor(game.admissable_moves) * model_out).argmax().item()
#game.move(choice, -1)
#game.state



