from math import inf
from collections import Counter
import itertools
from time import time
import numpy as np

TIME_LIMIT = 10

#conversion from row, col coords to index coords
def index(x, y):
    x -= 1
    y -= 1
    return ((x//3)*27) + ((x % 3)*3) + ((y//3)*9) + (y % 3)

#index belonged of small boxes
def indices_of_box(b):
    return list(range(b*9, b*9 + 9))

#print the board in text
def show_board(state):
    for row in range(1, 10):
        row_str = ["|"]
        for col in range(1, 10):
            row_str += [state[index(row, col)]]
            if (col) % 3 == 0:
                row_str += ["|"]
        if (row-1) % 3 == 0:
            print("-"*(len(row_str)*2-1))
        print(" ".join(row_str))
    print("-"*(len(row_str)*2-1))

#add moves into the board
def update_board(state, move, player):
    if not isinstance(move, int):
        move = index(move[0], move[1])
    return state[: move] + player + state[move+1:]

#check boxes winner
def box_checker(box_str):
    global goals
    for idxs in goals:
        (x, y, z) = idxs
        if (box_str[x] == box_str[y] == box_str[z]) and box_str[x] != ".":
            return box_str[x]
    return "."

#temporary list of small boxes won 
def list_box_won(state):
    temp_box_win = ["."] * 9
    for b in range(9):
        idxs_box = indices_of_box(b)
        box_str = state[idxs_box[0]: idxs_box[-1]+1]
        tie = True
        for tile in box_str:
            if tile == '.':
                tie = False
                break
            if tie:
              temp_box_win[b] = "T"
        temp_box_win[b] = box_checker(box_str)
    return temp_box_win

#update the board with the box won in 9 * winner (for showing)
def show_box_won(state):
    box_win = list_box_won(state)
    for b in range(9):
        if(box_win[b]!="."):
            for i in range(9):
                m = b*9 + i
                state = state[: m] + box_win[b] + state[m+1:]
    return state

#possible moves in the next box
def possible_moves(state, last_move):
    box_won = list_box_won(state)
    if not isinstance(last_move, int):
        last_move = index(last_move[0], last_move[1])
    box_to_play = last_move % 9
    idxs = indices_of_box(box_to_play)
    if box_won[box_to_play] != ".":
        pi_2d = [indices_of_box(b) for b in range(9) if box_won[b] == "."] #creates a list of 2D indices of all the boxes that are empty or not yet won
        possible_indices = list(itertools.chain.from_iterable(pi_2d)) #flattens the nested list pi_2d into a single list of 1D indices
    else:
        possible_indices = idxs
    return possible_indices

#couples of results of boards and moves
def results(state, player, last_move):
    boards = []
    moves_idx = []
    possible_indexes = possible_moves(state, last_move)
    for idx in possible_indexes:
        if state[idx] == ".":
            moves_idx.append(idx)
            boards.append(update_board(state, idx, player))
    return zip(boards, moves_idx)

#opponent of the player
def opponent(p):
    return "O" if p == "X" else "X"

#evaluation to attract the AI to win
def evaluate_box(box_str, player):
    global goals
    score = 0
    three = Counter(player * 3)
    two = Counter(player * 2 + ".")
    one = Counter(player * 1 + "." * 2)
    three_opponent = Counter(opponent(player) * 3)
    two_opponent = Counter(opponent(player) * 2 + ".")
    one_opponent = Counter(opponent(player) * 1 + "." * 2)

    twoT1 = Counter(player * 2 + opponent(player))
    twoT2 = Counter(opponent(player) * 2 + player)
    twoT3 = Counter(opponent(player) + player * 2)
    twoT4 = Counter(player + opponent(player) + ".")

    for idxs in goals:
        (x, y, z) = idxs
        current = Counter([box_str[x], box_str[y], box_str[z]])

        if box_str == box_won and current == three:
            score += 300000
            return score
        if box_str == box_won and current == three_opponent:
            score -= 300000
            return score
        elif current == three:
            score += 300
        elif current == two:
            score += 20
        elif current == one:
            score += 5
        elif current == three_opponent:
            score -= 300
            return score
        elif current == two_opponent:
            score -= 20
        elif current == one_opponent:
            score -= 5
        elif current == twoT1 or current == twoT2 or current == twoT3 or current == twoT4:
            score -= 2

    for k in box_str:
        if k == "T":
            score -= 4
    
    return score

#sum of scores of small boxes and also big ones
def utility(state, depth, player):
    box_won = list_box_won(state)
    score = 0
    score += depth * 3
    score += evaluate_box(box_won, player) * 400
    for b in range(9):
        idxs = indices_of_box(b)
        box_str = state[idxs[0]: idxs[-1]+1]
        score += evaluate_box(box_str, player)
    return score

#minimax algorithm
def minimax(state, last_move, player, depth, start_time, beginer=False):
    if beginer:
        steps = results(state, player, 4) #if AI first, it will play in the middle box of the board
    else: 
        steps = results(state, player, last_move)
    best_move = (-inf, None)
    for step in steps:
        value = minimize(step[0], step[1], opponent(player), depth-1, start_time, -inf, inf)
        if value > best_move[0]:
            best_move = (value, step)
        #print("value = ", value)
        if time() >= start_time + TIME_LIMIT - 0.02:
            break
    return best_move[1]

#minimizing opponent of minimax
def minimize(state, last_move, player, depth, start_time, alpha, beta):
    if depth == 0 or box_checker(list_box_won(state)) != "." or is_full(state):
        #print(utility(state, depth, opponent(player)))
        #print(update_box_won(state))
        return utility(state, depth, opponent(player))
    steps = results(state, player, last_move)
    value = inf
    for step in steps:
        value = min(value, maximize(step[0], step[1], opponent(player), depth-1, start_time, alpha, beta))
        if value <= alpha:
            return value
        beta = min(beta, value)
        if time() >= start_time + TIME_LIMIT - 0.05:
            break
    return beta

#maximizing player of minimax
def maximize(state, last_move, player, depth, start_time, alpha, beta):
    if depth == 0 or box_checker(list_box_won(state)) != "." or is_full(state):
        #print(utility(state, depth, player))
        #print(update_box_won(state))
        return utility(state, depth, player)
    steps = results(state, player, last_move)
    value = -inf
    for step in steps:
        value = max(value, minimize(step[0], step[1], opponent(player), depth-1, start_time, alpha, beta))
        if value >= beta:
            return value
        alpha = max(alpha, value)
        if time() >= start_time + TIME_LIMIT - 0.05:
            break
    return alpha

#check correct inputs
def valid_input(state, move):
    global box_won
    if not (0 < move[0] < 10 and 0 < move[1] < 10):
        return False
    if box_won[index(move[0], move[1]) // 9] != ".": 
        return False
    if state[index(move[0], move[1])] != ".":
        return False
    return True

#conversion from box coords to row, col coords
def convert_box_coords(big_box, small_box):
    row = (big_box - 1) // 3 * 3 + (small_box - 1) // 3 + 1
    col = (big_box - 1) % 3 * 3 + (small_box - 1) % 3 + 1
    return (row, col)

#take correct inputs
def take_input(state, bot_move, piece):
    all_open_flag = False
    if bot_move == -1 or len(possible_moves(state, bot_move)) > 9 or list_box_won(state)[bot_move % 9] != ".":
        all_open_flag = True
    if all_open_flag:
        print("Play anywhere you want!")
        big_box = int(input("big_box = "))
        if big_box == -1:
            raise SystemExit
    else:
        box_dict = {0: "Upper Left", 1: "Upper Center", 2: "Upper Right",
                    3: "Center Left", 4: "Center", 5: "Center Right",
                    6: "Bottom Left", 7: "Bottom Center", 8: "Bottom Right"}
        print(f"Where would you like to place '{piece}' in ~"
              + box_dict[bot_move % 9] + "~ box?")
        big_box = bot_move % 9 + 1
        print("with big_box = " + str(big_box))
    
    small_box = int(input("small_box = "))
    (x,y) = convert_box_coords(big_box, small_box)
    print("#" * 60)

    if bot_move != -1 and index(x, y) not in possible_moves(state, bot_move):
        raise ValueError
    if not valid_input(state, (x, y)):
        raise ValueError
    return (x, y)

#check if the board is full
def is_full(state):
    for i in range(81):
        if state[i] == ".":
            return False
    return True

#conversion from index coords to box coords
def convert_index_to_box_coords(index):
    x = index // 9
    y = index % 9
    row = x // 3 * 3 + y // 3
    col = x % 3 * 3 + y % 3
    big_box = row // 3 * 3 + col // 3 + 1
    small_box = (row % 3) * 3 + col % 3 + 1
    return (big_box, small_box)

#game of Ultimate Tic Tac Toe
def game(state="." * 81, depth=6, start_player=1, piece="X"):
    global box_won, goals 
    goals = [(0, 4, 8), (2, 4, 6)] # Diagonals wins
    goals += [(i, i+3, i+6) for i in range(3)] # Columns wins
    goals += [(3*i, 3*i+1, 3*i+2) for i in range(3)] # Rows wins
    box_won = list_box_won(state)
    show_board(state)
    bot_move = -1
    i = 0
    c = 1
    if start_player == 2:
        print("Please wait, Bot is thinking...")
        start_time = time()
        bot_state, bot_move = minimax(state, bot_move, piece, depth, start_time, True)
        end_time = time()
        print("#" * 60)
        print(f"Step {c}")
        print(f"Bot placed '{piece}' on", convert_index_to_box_coords(bot_move))
        print(f"Bot's move took {end_time-start_time:.2f} seconds")
        show_board(bot_state)
        state = bot_state
        box_won = list_box_won(bot_state)
        piece = opponent(piece)
        i += 1
        c+=1

    while True:
        try:
            user_move = take_input(state, bot_move, piece)
        except ValueError:
            print("Invalid input or move not possible!")
            show_board(state)
            continue
        except SystemError:
            print("Game Stopped!")
            break
        
        print(f"Step {c}")
        print(f"User placed '{piece}' on ",convert_index_to_box_coords(index(user_move[0], user_move[1])))
        state = update_board(state, user_move, piece)
        state = show_box_won(state)
        show_board(state)
        box_won = list_box_won(state)
        if box_checker(box_won) != "." or is_full(state):
            break
        c += 1
        piece = opponent(piece)

        print("Please wait, Bot is thinking...")
        start_time = time()
        state, bot_move = minimax(state, user_move, piece, depth, start_time, False)
        end_time = time()

        print("#" * 60)
        print(f"Step {c}")
        print(f"Bot placed '{piece}' on", convert_index_to_box_coords(bot_move))
        print(f"Bot's move took {end_time-start_time:.2f} seconds")
        state = show_box_won(state)
        show_board(state)
        
        piece = opponent(piece)
        box_won = list_box_won(state)
        if box_checker(box_won) != "." or is_full(state):
            break
        c += 1
    game_won = box_checker(list_box_won(state))
    if game_won == "X":
        print("Congratulations YOU WIN!")
    elif game_won == "O":
        print("YOU LOSE!")
    elif is_full(state):
        print("DRAW!")
    print(f"With {c} steps")


    return state

#main function with depth 6
if __name__ == "__main__":

    INITIAL_STATE = "." * 81
    print("Choose 1 if you want to use X and 2 if you want to use O")
    piece_choice = int(input())
    print("Choose 1 if you want to start and 2 if you want to let bot start")
    start_choice = int(input())
    if piece_choice == 1:
        if start_choice == 1:
            game(INITIAL_STATE, depth=6, start_player=1, piece="X")
        elif start_choice == 2:
            game(INITIAL_STATE, depth=6, start_player=2, piece="O")
        else:
            print("Invalid choice. Please choose 1 or 2.")
    elif piece_choice == 2:
        if start_choice == 1:
            game(INITIAL_STATE, depth=6, start_player=1, piece="O")
        elif start_choice == 2:
            game(INITIAL_STATE, depth=6, start_player=2, piece="X")
        else:
            print("Invalid choice. Please choose 1 or 2.")
    else:
        print("Invalid choice. Please choose 1 or 2.")



