#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create latex source for complex tables using OOP.

Explanation of the 'CellStructure':
    You will supply nested lists containing the table entries.
        table_entries = [[row1_entry1, row1_entry2, row1_entry3, ...],
                         [row2_entry1, row2_entry2, row2_entry3, ...],
                          ...]
    The possible formats of an 'entry' are as follows ::
        -- ('int' or 'str') a single table entry
        -- [entry, shape, Optional: ident] (or same syntax with (...) instead of [...])
            --> here 'shape' refers to the shape of a "multicolumn/multirow"
            --> for example shape = [2, 3] will put the entry inside a region of
                2 rows and 3 columns inside the table (see examples below)
            --> if shape only contains one value (or is an int), a multicolumn is assumed
            --> Optional: ident == alignment and vertical line of a multi-box

    A single example row containing all variations is the following:
        (Warning, this is not a valid row, since the additional rows created by the
         multirows have gaps where there are *no* multirows!)
        ["simple", 156, ("entry", [2]), ("shorthand-multicolumn", 4), ["my-multi-box", [3, 2]],
         (1514, [1, 3], 'r|'), ["my-multirow-with-left-align-and-no-vlines", [2, 1], 'l']]
"""

import matplotlib.pyplot as plt
import numpy as np


def rgba2rgb(rgba, background=(1.0, 1.0, 1.0)):
    """Convert rgba to rgb based on a background color."""
    rgb = np.zeros((rgba.shape[0], rgba.shape[1], 3))
    alpha = rgba[:, :, 3]
    for i in range(3):
        rgb[:, :, i] = (1 - alpha) * background[i] + rgba[:, :, i] * alpha
    return np.round(rgb, 5)


def rgb2cmyk(rgb):
    """Convert rgb to cmyk (cyan, magenta, yellow, keyline/black)."""
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    black = 1 - np.max(rgb, axis=2)
    indx = np.where(black < 1 - 1e-5)

    cmyk = np.zeros((4, rgb.shape[0], rgb.shape[1]))

    cmyk[0] = 1 - red - black
    cmyk[1] = 1 - green - black
    cmyk[2] = 1 - blue - black
    cmyk[3] = black
    for i in range(3):
        cmyk[i] = np.reshape(cmyk[i][indx] / (1 - black[indx]), (rgb.shape[0], rgb.shape[1]))
    return np.round(cmyk, 5)


class HeatTable:
    """Special table supporting latex-cells with background colors based on a heatmap.
    """
    def __init__(self, cells, heat, colormap='viridis', mode='rgb', AutoSetup=True, alpha=None):
        """Initialize cells and heatmap. If 'AutoSetup', this also generates the table.
        """
        self.cells = cells
        self.heat = heat

        if AutoSetup:
            self.setup(colormap, mode, alpha=alpha)

    def setup(self, colormap='viridis', mode='rgb', alpha=None):
        """Setup the colors of each cell according to the given 'heat'-array."""
        if mode != 'rgb' and alpha is None:
            msg = "'alpha' must be given for colormode 'rgba' or 'cmyk'."
            raise ValueError(msg)

        cmap = plt.cm.get_cmap(colormap)
        rgba = cmap(self.heat)
        rgba[np.isnan(self.heat)] = np.ones(4)
        if mode == 'rgb':
            r, g, b = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
            self.l_cells = [
                [fr"\cellcolor[rgb]{{{r[row, col]}, {g[row, col]}, {b[row, col]}}}"
                 + f"{self.cells[row, col]}" for col in range(self.cells.shape[1])]
                for row in range(self.cells.shape[0])]

        elif mode == 'rgba':
            rgba[:, :, 3] = alpha
            rgb = rgba2rgb(rgba)
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            self.l_cells = [
                [fr"\cellcolor[rgb]{{{r[row, col]}, {g[row, col]}, {b[row, col]}}}"
                 + f"{self.cells[row, col]}" for col in range(self.cells.shape[1])]
                for row in range(self.cells.shape[0])]

        elif mode == 'cmyk':
            rgba[:, :, 3] = alpha
            rgb = rgba2rgb(rgba)
            cmyk = rgb2cmyk(rgb)
            c, m, y, k = cmyk
            self.l_cells = [
                [fr"\cellcolor[cmyk]{{{c[row, col]}, {m[row, col]}, {y[row, col]}, {k[row, col]}}}"
                 + f"{self.cells[row, col]}" for col in range(self.cells.shape[1])]
                for row in range(self.cells.shape[0])]

        else:
            msg = f"Color mode was {mode = } but must be one of {'rgb', 'rgba', 'cmyk'}."
            raise ValueError(msg)


class Table:
    """Provides methods for creating simple rectangular latex tables"""
    def __init__(self, cells):
        self.cells = self.__tolist(cells)
        self.__last_row = 0     # index of the last row that was added
        self.num_cols = 0       # number of columns in the table

        self.p_cells = [[]]     # np.ndarray containing the preprocessed cells
        self.l_cells = [[]]     # cells containing latex 'table-cell-strings'
        self.c_ind = []         # column indices of multi-cells
        self.__hlines_ind = []  # indices and shapes of multi-rows (for \cline{}'s)
        self.hlines = []        # boolean array for \cline{}'s


    def __tolist(self, cells):
        """Convert a numpy array to a list (passes if given a list)"""
        if isinstance(cells, np.ndarray):
            return cells.tolist()
        return cells


    def add_cells(self, cells, pos='below'):
        """Add cells at the specified position 'pos' relative to the existing 'Table'.
        'cells' must be a nested 2D-list, where entires may be arbitrary 'cell-entries'.
        """
        if pos not in ['below', 'right']:
            msg = f"Position was {pos = } but must be one of {'below', 'right'}."
            raise ValueError(msg)

        if isinstance(cells, (HeatTable, self.__class__)):
            cells = cells.l_cells
        else:
            cells = self.__tolist(cells)

        if pos == 'below':
            self.__last_row = len(self.cells)
            self.cells += cells

            # check for trailing multirow-cell
            # note that such a row can only contain exactly one element
            if isinstance(cells[-1][0], (tuple, list)):
                shape = cells[-1][0][1]    # last row, first entry, shape is at indx 1
                if len(shape) != 2:
                    msg = ("Found a trailing multirow in Table.add_cells with "
                           f" shape {shape} (should be ({shape[0]}, 1))")
                    raise IndexError(msg)

                num_trailing_rows = shape[0] - 1
                for i in range(num_trailing_rows):
                    self.cells.append([])

        elif pos == 'right':
            if isinstance(cells, np.ndarray):
                cells = [list(row) for row in cells]
            for i in range(len(self.cells) - self.__last_row):
                if i >= len(cells):
                    break
                self.cells[self.__last_row + i] += cells[i]


    def __shape_ident_from_elem(self, elem, j):
        """Return the shape and the identifier for a given multicell"""
        shape = elem[1]

        # identifier for multicolumn alignment and border
        if len(elem) == 3:
            ident = elem[-1]
        else:
            if j == 0:
                ident = '|c|'     # vertical line left to the first column
            else:
                ident = 'c|'

        # avoid TypeError if the shape was declared as (5) instead of (5,)
        if isinstance(shape, int):
            shape = (shape,)

        if len(shape) == 1:
            shape = (1, shape[0])       # reshape to (1, num_multicols)

        if shape[1] == -1:              # fill the entire row with this cell
            shape = (shape[0], self.num_cols)

        return shape, ident


    def __get_num_cols(self):
        """Compute the number of column in the table (respects multicells)"""
        num_cols = 0
        for elem in self.cells[0]:
            if isinstance(elem, (tuple, list)):
                num_cols += elem[1][-1]
            else:
                num_cols += 1
        return num_cols


    def __preprocess_cells(self):
        """Convert the list of 'cells' to a rectangular array to simplify the access.
        Redundant elemnts will remain 'np.inf' and can thus be removed later.
        """
        self.num_cols = self.__get_num_cols()
        self.p_cells = np.full((len(self.cells), self.num_cols), -np.inf, dtype=object)
        for i, row in enumerate(self.cells):
            j = 0
            for elem in row:

                # search for the first 'placeholder-entry' of value 'np.inf' in row 'i'
                for k in range(j, self.num_cols):
                    if self.p_cells[i, k] == -np.inf:
                        break
                    j += 1

                # simple table entries are already converted to strings here
                if not isinstance(elem, (tuple, list)):
                    self.p_cells[i, j] = str(elem)
                    j += 1

                else:
                    shape, ident = self.__shape_ident_from_elem(elem, j)
                    self.p_cells[i:i + shape[0], j:j + shape[1]] = np.inf
                    self.p_cells[i, j] = elem
                    self.c_ind.append([i, j, shape, ident])
                    j += shape[1]


    def __handle_multicol(self, i, j, value, shape, ident):
        """Generate the Latex-string for a multicolumn of specified parameters"""
        text = f"\\multicolumn{{{shape[1]}}}{{{ident}}}{{{value}}}"
        self.p_cells[i, j] = text


    def __handle_multirow(self, i, j, value, shape, ident):
        r"""Generate the Latex-string for a multirow of specified parameters.
        Also fill in the required'empty' Latex-strings below the multirow.

        This method adds the indices and sizes of multirows to '__hlines_ind'
        for later generating \cline{}'s.
        """
        if shape[1] == 1:                   # multirow only
            text = f"\\multirow{{{shape[0]}}}{{*}}{{{value}}}"
            empty = ""
        else:
            text = (f"\\multicolumn{{{shape[1]}}}{{{ident}}}" + "{"
                    + f"\\multirow{{{shape[0]}}}{{*}}{{{value}}}" + "}")
            empty = f"\\multicolumn{{{shape[1]}}}{{{ident}}}{{}}"

        # insert latex-text and the 'placeholders' for multirows below
        self.p_cells[i, j] = text
        self.p_cells[i + 1:i + shape[0], j] = empty

        # store the indices and shapes of multirows to handle crossing of \hline's
        self.__hlines_ind.append([i, j, shape])


    def create_table(self):
        r"""Steps:
        Preprocess the cells to get a rectangular array with correct table structure.
            This method also stores indices of any multicells encountered.
        For each multicell, generate the latex string and replace the
            fill-characters below multirows with the correct placeholders.
        Create the nested list containing latex-cells by eliminating
            redundant placeholders.
        Setup the boolean mask for table-indices with multirows such that
            \cline{}'s may be drawn later on.
        """
        self.__preprocess_cells()
        for ind in self.c_ind:
            i, j, shape, ident = ind
            elem = self.p_cells[i, j]
            value = elem[0]

            # handle different shapes of multicolumns/multirows
            if shape[0] == 1:               # multicolumn only
                self.__handle_multicol(i, j, value, shape, ident)
            else:
                self.__handle_multirow(i, j, value, shape, ident)

        self.l_cells = [[elem for elem in row if elem != np.inf]
                        for row in self.p_cells]
        self._setup_hlines()


    def _setup_hlines(self):
        """This sets up a boolean array 'hlines' as a LUT for drawing the horizontal lines."""
        self.hlines = np.ones((len(self.l_cells), self.num_cols), dtype=int)
        for ind in self.__hlines_ind:
            i, j, shape = ind
            rows = shape[0] - 1     # \hline below the box is fine, so only until -1
            cols = shape[1]
            self.hlines[i:i+rows, j:j+cols] = 0


    def __str__(self):
        r"""Join all the 'cells' using '&' and '\\'.
        If the boolean 2d-array 'hlines' is given, the horizontal lines will avoid
        any index where 'hlines[i, j] == 0' (and draw \cline{x-y} accordingly)
        """
        string = f"\\begin{{tabular}}{{|*{{{self.num_cols}}}{{c|}}}}\\hline\n"
        for i, row in enumerate(self.l_cells[:-1]):
            string += " & ".join(row) + " \\\\ "
            if np.any(self.hlines[i] == 0):
                first, last = 1, 1
                for j, val in enumerate(self.hlines[i]):
                    if val:
                        last += 1
                    else:
                        if self.hlines[i][max(j-1, 0)]:
                            string += f"\\cline{{{first}-{last-1}}}"
                            first = last

                        first += 1
                        last += 1

                if first < last:
                    string += f"\\cline{{{first}-{last-1}}}"

            else:
                string += "\\hline"

            string += "\n"
        return string + " & ".join(self.l_cells[-1]) + " \\\\ \\hline\n\\end{tabular}"

    def __repr__(self):
        return "Table(" + self.__str__() + ")\n"


def main():
    """Setup and print a complex example table"""
    print(__doc__)

    # a somewhat complicated example
    header = [
        [[r'$\substack{\mathrm{VHT}\\\mathrm{MCS}}$', (2,1)], ['Modulation', (2, 1)],
          ['Coding', (2,1)], r'\SI{20}{MHz}', r'\SI{40}{MHz}', r'\SI{80}{MHz}'],
        [[r'Airtime in \SI{}{\micro s}', (1,3)]],
        [[r'Multicell(2,2)', (2,2)], "text", [r'Multicell2(2,2)', (2,2)], "text2"],
        ["nothing", "nothing2"],
        ["test", [r'Multicell(2,4)', (2,4)], "test2"],
        ["test3", "test4"]
    ]

    # create the table
    table = Table(header)
    table.add_cells([
        [[r'BIGCELL', (2, 5)]], []
        ] * 3, 'right')
    table.add_cells([[['1 Spatial Stream', (1,-1)]]])

    table.add_cells([
        ["P", "x", ["Multicell(3,4)", (3,4)], "ttt", "666", "777", ["Testcell(2,2)", (2,2)]],
        ["P2", ["Testrow(2,1)", (2,1)], ["Testcol(1,3)", (1,3)]],
        ["P3", "T1", "T2", "T3", ["TCol(1,2)", (1,2)]]
        ])
    table.add_cells([[['2 Spatial Streams', (1,-1)]]])

    table.create_table()
    print(table)
    return 0

if __name__ == "__main__":
    main()
