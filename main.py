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

win_arr = []
loss_arr = []
wins_a = 0
wins_b = 0
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
                wins_a += 1
            elif game.outcome == 'draw':
                print('Draw')
            break

        # Player 2
        nnplayer.move(player=PLAYER_B) 
        
        
        if game.concluded:
            if game.outcome != 'draw': 
                winner = PLAYER_B
                loser = PLAYER_A
                wins_b += 1
            elif game.outcome == 'draw':
                winner = 'Draw'
                print('Draw')
            break
    # =================================================================================

    # Reenforcement Part
    if game.outcome != 'draw':
        win_loss = prop.rel_win_prop(game=game,
                                        player=winner,
                                        starter=STARTER_DICT[winner])
        loss_loss = prop.rel_loss_prop(game=game,
                                    player=loser,
                                    starter=STARTER_DICT[loser])
        win_arr.append(win_loss) 
        loss_arr.append(loss_loss)
    #if i % 1 == 100:
        #recent_loss = loss_arr[-11:-1]
        #print(f'Game {i} loss: {np.mean(recent_loss):.3f}, Winner: {winner}')
    if i % 100 == 0 and i != 0:
        mean_loss_loss = np.mean(loss_arr)
        mean_win_loss = np.mean(win_arr)
        print(f'Game {i}:')
        print(f'Mean Win Loss: {mean_win_loss:.3f}')
        print(f'Mean Loss Loss: {mean_loss_loss:.3f}')
        print(f'A Win: {wins_a:.3f}%')
        print(f'B Win: {wins_b:.3f}%')
        print(f'Example State:')
        print(game.state)
        loss_arr = []
        win_arr
        win_hist.append(winner)
        wins_a = 0
        wins_b = 0



#fig, ax = plt.subplots(1,1)
#ax.plot(win_hist)
#fig.savefig('loss.png', dpi=300)
#
#th.save(model.state_dict(), 'model001.torch')
#
#game = Game()
#game.move(3, -1)
#
#state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
#
#model_out = model(state_input)
#model_out
#
#
#game = Game()
#
#nonzero_indices = np.nonzero(game.admissable_moves)[0] 
#game.move(np.random.choice(nonzero_indices), -1) 
#
#state_input = th.tensor(game.state).unsqueeze(dim=0).unsqueeze(dim=0).float()
#model_out = model(state_input)
#
#admissable_tens = th.tensor(game.admissable_moves)
#nonzero_indices = th.nonzero(admissable_tens)
#masked_tensor = model_out[0, nonzero_indices]
#max_index = nonzero_indices[th.argmax(masked_tensor)]
#choice = max_index.item()
#game.move(choice, 1)
#game.state
#model_out
#model_out.shape