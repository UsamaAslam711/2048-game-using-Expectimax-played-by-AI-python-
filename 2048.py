
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
