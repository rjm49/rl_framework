'''
Created on 26 Oct 2016

@author: Russell Moore
The TicTacToe example uses "temporal difference" learning to improve its victory-probability estimates
'''
import copy
import math
from random import randint
import random

p_win = {}

def generate_board_state(snum):
    c = i
    cout = []
    for _ in range(9):
        cout.append(c % 3)
        c = c // 3
    return(tuple(cout)) # hopefully splits on the empty string

def board_is_valid(s):
    num_o = s.count(1)
    num_x = s.count(2)
    #board state valid if there's the same number of x and o's or just one different between them
    return True if (num_o==num_x or abs(num_o-num_x)==1) else False

def skip_slicer(seq, take, skip):
    return [x for start in range(0, len(seq), take + skip)
              for x in seq[start:start + take]]

def board_is_win(s):
    n = 3
    rows = [s[i:i+n] for i in range(0, len(s), n)]
    
    for row in rows:
        #print("ROW", row)
        print(row)
        if row.count(1)==3: 
            return "o"
        if row.count(2)==3: 
            return "x"
    
    for c in range(0,n):
        col = [r[c] for r in rows]
        #print("COL", col)
        if col.count(1)==3: 
            return "o"
        if col.count(2)==3: 
            return "x"
    
    return False

def get_next_states(s, colour):
    poss_moves = [i for i, sq in enumerate(s) if sq==0]
    next_states = []
    for m in poss_moves:
        s_new = list(s)
        s_new[m] = colour
#         if board_is_valid(s_new): #throw away moves that break the rules
        next_states.append(tuple(s_new))
    return next_states

if __name__ == '__main__':
#    board = (0) * 9
    #use 1 for o, 2 for x as per "naughts and crosses"
    for i in range(19683):
        s = generate_board_state(i)
#        if board_is_valid(s):
        if True:
#             print(s)
            is_win = board_is_win(s)
            if is_win:
#                 print("is win==", is_win)
                p_win[s]=1 if is_win=="x" else 0 #we win with three x's else we cannot win
            elif s.count(0)==0:
                p_win[s]=0 #this is a draw position - there are no winners but no more free spaces
            else:
                p_win[s]=0.5 # we are unsure about non-winning non-draw states
    print("board states generated")
    
    alpha = 0.1 #alpha is the same thing as learning rate
    
    moves_to_vic = []
    
    for _ in range(1000):
        print("New Game...")
        mov_cnt = 0
        board = (0,0,0, 0,0,0, 0,0,0)
        while not board_is_win(board):
            max_moves = []
            subopt_moves = []
            curr_max = 0
            next_states = get_next_states(board, 2)
            for ns in next_states:
                estd_p_win = p_win[ns] #what's our estimated probability of winning from this state?
                #print(ns, estd_p_win)
                
                if estd_p_win > curr_max:
                    subopt_moves += max_moves
                    max_moves = [ns]
                    curr_max = estd_p_win
                elif(estd_p_win == curr_max):
                    max_moves.append(ns)
                else:
                    subopt_moves.append(ns)
                
                #print(subopt_moves, max_moves, curr_max)
                
            #roll a die
            die = randint(1,10)
            mov = None
            print("making move")
            if(die == 6 and subopt_moves):
                #exploratory step
                #choose randomly from moves thought to be sub-optimal
                mov = random.choice(subopt_moves)
                print("explorative move", mov)
            else:
                #exploitative step
                #choose randomly from the best moves List
                mov = random.choice(max_moves)
                curr_p_win = p_win[board]
                next_p_win = p_win[mov]
                #print(curr_p_win, next_p_win)
                new_v = curr_p_win + alpha*( next_p_win - curr_p_win ) #backup step
                print("greedy move", mov)
                print("updating",board,"from curr value", curr_p_win, "to new value",new_v)
                p_win[board] = new_v
            mov_cnt += 1
            board = mov
            #print(board, p_win[board])
                
        print("Victory! in", mov_cnt, "moves.")
        moves_to_vic.append(mov_cnt)
        
    print("Moves history:", moves_to_vic)