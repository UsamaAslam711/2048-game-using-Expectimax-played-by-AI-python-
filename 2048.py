from tkinter import Frame, Label, CENTER
from random import randint
import math
import time
import numpy as np
from numba import jit
from random import randint, seed

UP, DOWN, LEFT, RIGHT = range(4)

class EXPECTIMAX():

    def get_move(self, board):
        best_move, _ = self.maximize(board)
        return best_move

    def eval_board(self, board, n_empty): 
        grid = board.grid

        utility = 0
        smoothness = 0

        big_t = np.sum(np.power(grid, 2))
        s_grid = np.sqrt(grid)
        smoothness -= np.sum(np.abs(s_grid[::,0] - s_grid[::,1]))
        smoothness -= np.sum(np.abs(s_grid[::,1] - s_grid[::,2]))
        smoothness -= np.sum(np.abs(s_grid[::,2] - s_grid[::,3]))
        smoothness -= np.sum(np.abs(s_grid[0,::] - s_grid[1,::]))
        smoothness -= np.sum(np.abs(s_grid[1,::] - s_grid[2,::]))
        smoothness -= np.sum(np.abs(s_grid[2,::] - s_grid[3,::]))
        
        empty_w = 100000
        smoothness_w = 3

        empty_u = n_empty * empty_w
        smooth_u = smoothness ** smoothness_w
        big_t_u = big_t

        utility += big_t
        utility += empty_u
        utility += smooth_u

        return (utility, empty_u, smooth_u, big_t_u)

    def maximize(self, board, depth = 0):
        moves = board.get_available_moves()
        moves_boards = []

        for m in moves:
            m_board = board.clone()
            m_board.move(m)
            moves_boards.append((m, m_board))

        max_utility = (float('-inf'),0,0,0)
        best_direction = None

        for mb in moves_boards:
            utility = self.chance(mb[1], depth + 1)

            if utility[0] >= max_utility[0]:
                max_utility = utility
                best_direction = mb[0]

        return best_direction, max_utility

    def chance(self, board, depth = 0):
        empty_cells = board.get_available_cells()
        n_empty = len(empty_cells)

        if n_empty >= 6 and depth >= 3:
            return self.eval_board(board, n_empty)

        if n_empty >= 0 and depth >= 5:
            return self.eval_board(board, n_empty)

        if n_empty == 0:
            _, utility = self.maximize(board, depth + 1)
            return utility

        possible_tiles = []

        chance_2 = (.9 * (1 / n_empty))
        chance_4 = (.1 * (1 / n_empty))
        
        for empty_cell in empty_cells:
            possible_tiles.append((empty_cell, 2, chance_2))
            possible_tiles.append((empty_cell, 4, chance_4))

        utility_sum = [0, 0, 0, 0]

        for t in possible_tiles:
            t_board = board.clone()
            t_board.insert_tile(t[0], t[1])
            _, utility = self.maximize(t_board, depth + 1)

            for i in range(4):
                utility_sum[i] += utility[i] * t[2]

        return tuple(utility_sum)



dirs = [UP, DOWN, LEFT, RIGHT] = range(4)

@jit
def merge(a):
    for i in [0,1,2,3]:
        for j in [0,1,2]:
            if a[i][j] == a[i][j + 1] and a[i][j] != 0:
                a[i][j] *= 2
                a[i][j + 1] = 0   
    return a

@jit
def justify_left(a, out):
    for i in [0,1,2,3]:
        c = 0
        for j in [0,1,2,3]:
            if a[i][j] != 0:
                out[i][c] = a[i][j]
                c += 1
    return out

@jit
def get_available_from_zeros(a):
    uc, dc, lc, rc = False, False, False, False

    v_saw_0 = [False, False, False, False]
    v_saw_1 = [False, False, False, False]

    for i in [0,1,2,3]:
        saw_0 = False
        saw_1 = False

        for j in [0,1,2,3]:

            if a[i][j] == 0:
                saw_0 = True
                v_saw_0[j] = True

                if saw_1:
                    rc = True
                if v_saw_1[j]:
                    dc = True

            if a[i][j] > 0:
                saw_1 = True
                v_saw_1[j] = True

                if saw_0:
                    lc = True
                if v_saw_0[j]:
                    uc = True

    return [uc, dc, lc, rc]

class GameBoard:
    def __init__(self):
        self.grid = np.zeros((4, 4))#, dtype=np.int_)

    def clone(self):
        grid_copy = GameBoard()
        grid_copy.grid = np.copy(self.grid)
        return grid_copy

    def insert_tile(self, pos, value):
        self.grid[pos[0]][pos[1]] = value

    def get_available_cells(self):
        cells = []
        for x in range(4):
            for y in range(4):
                if self.grid[x][y] == 0:
                    cells.append((x,y))
        return cells

    def get_max_tile(self):
        return np.amax(self.grid)

    def move(self, dir, get_avail_call = False):
        if get_avail_call:
            clone = self.clone()

        z1 = np.zeros((4, 4))#, dtype=np.int_)
        z2 = np.zeros((4, 4))#, dtype=np.int_)

        if dir == UP:
            self.grid = self.grid[:,::-1].T
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid.T[:,::-1]
        if dir == DOWN:
            self.grid = self.grid.T[:,::-1]
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid[:,::-1].T
        if dir == LEFT:
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
        if dir == RIGHT:
            self.grid = self.grid[:,::-1]
            self.grid = self.grid[::-1,:]
            self.grid = justify_left(self.grid, z1)
            self.grid = merge(self.grid)
            self.grid = justify_left(self.grid, z2)
            self.grid = self.grid[:,::-1]
            self.grid = self.grid[::-1,:]

        if get_avail_call:
            return not (clone.grid == self.grid).all()
        else:
            return None

    def get_available_moves(self, dirs = dirs):
        available_moves = []
        
        a1 = get_available_from_zeros(self.grid)

        for x in dirs:
            if not a1[x]:
                board_clone = self.clone()

                if board_clone.move(x, True):
                    available_moves.append(x)

            else:
                available_moves.append(x)

        return available_moves

    def get_cell_value(self, pos):
        return self.grid[pos[0]][pos[1]]



UP, DOWN, LEFT, RIGHT = range(4)


SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        self.grid_cells = []

        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        self.EXPECTIMAX = EXPECTIMAX()

        self.run_game()
        self.mainloop()

    def run_game(self):
        while True:
            self.board.move(self.EXPECTIMAX.get_move(self.board))
            self.update_grid_cells()
            self.add_random_tile()
            self.update_grid_cells()

            if len(self.board.get_available_moves()) == 0:
                self.game_over_display()
                break

            self.update()
        
    def game_over_display(self):
        for i in range(4):
            for j in range(4):
                self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)

        self.grid_cells[1][1].configure(text="TOP",bg=BACKGROUND_COLOR_CELL_EMPTY)
        self.grid_cells[1][2].configure(text="4 TILES:",bg=BACKGROUND_COLOR_CELL_EMPTY)
        top_4 = list(map(int, reversed(sorted(list(self.board.grid.flatten())))))
        self.grid_cells[2][0].configure(text=str(top_4[0]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][1].configure(text=str(top_4[1]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][2].configure(text=str(top_4[2]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.grid_cells[2][3].configure(text=str(top_4[3]), bg=BACKGROUND_COLOR_DICT[2048], fg=CELL_COLOR_DICT[2048])
        self.update()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()

        for i in range(GRID_LEN):
            grid_row = []

            for j in range(GRID_LEN):

                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return randint(0, GRID_LEN - 1)

    def init_matrix(self):
        self.board = GameBoard()
        self.add_random_tile()
        self.add_random_tile()

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = int(self.board.grid[i][j])
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    n = new_number
                    if new_number > 2048:
                        c = 2048
                    else:
                        c = new_number

                    self.grid_cells[i][j].configure(text=str(n), bg=BACKGROUND_COLOR_DICT[c], fg=CELL_COLOR_DICT[c])
        self.update_idletasks()
        
    def add_random_tile(self):
        if randint(0,99) < 100 * 0.9:
            value = 2
        else:
            value = 4

        cells = self.board.get_available_cells()
        pos = cells[randint(0, len(cells) - 1)] if cells else None

        if pos is None:
            return None
        else:
            self.board.insert_tile(pos, value)
            return pos

gamegrid = GameGrid()
