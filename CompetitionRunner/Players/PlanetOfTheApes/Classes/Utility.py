from __future__ import print_function
import sys
from itertools import count
import Classes.Config as conf

def parse_board(tmp):
    arr = tmp.split("\n")
    board = []
    start = 1
    for i in range(0, conf.N):
        tmp = []
        for j in range(0, conf.N):
            tmp.append(arr[start + i][j + 1])
        board.append(tmp)
    return board

def print_board(arr):
    str = ""
    n = len(arr)
    for i in range(0, n):
        for j in range(0, n):
            str += arr[i][j]
        str += "\n"
    print(str)


def print_pos(pos, f=sys.stderr, owner_map=None):
    """ print visualization of the given board position, optionally also
    including an owner map statistic (probability of that area of board
    eventually becoming black/white) """
    if pos.n % 2 == 0:  # to-play is black
        board = pos.board.replace('x', 'O')
        Xcap, Ocap = pos.cap
    else:  # to-play is white
        board = pos.board.replace('X', 'O').replace('x', 'X')
        Ocap, Xcap = pos.cap
    print('Move: %-3d   Black: %d caps   White: %d caps  Komi: %.1f' % (pos.n, Xcap, Ocap, pos.komi), file=f)
    pretty_board = ' '.join(board.rstrip()) + ' '
    if pos.last is not None:
        pretty_board = pretty_board[:pos.last*2-1] + '(' + board[pos.last] + ')' + pretty_board[pos.last*2+2:]
    rowcounter = count()
    pretty_board = [' %-02d%s' % (conf.N-i, row[2:]) for row, i in zip(pretty_board.split("\n")[1:], rowcounter)]

    print("\n".join(pretty_board), file=f)
    print('    ' + ' '.join(conf.colstr[:conf.N]), file=f)
    print('', file=f)

def parse_coord(s):
    if s == 'pass':
        return None
    return conf.W+1 + (conf.N - int(s[1:])) * conf.W + conf.colstr.index(s[0].upper())

def str_coord(c):
    if c is None:
        return 'pass'
    row, col = divmod(c - (conf.W+1), conf.W)
    return '%c%d' % (conf.colstr[col], conf.N - row)

def isDone(pos):
    for i in range (0, len(pos)):
        if pos.board[i] == '.':
            if pos.checkmove(i):
                return False
    return True
