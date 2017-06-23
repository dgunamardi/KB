import time
from sys import argv

# --- GO MAIN PROGRAM DEPENDENCIES ---

from collections import namedtuple
from itertools import count
import re

# === EXECUTABLE AND TIMEOUT DEPENDENCIES ---
import time
from multiprocessing.pool import ThreadPool
from multiprocessing import TimeoutError
import subprocess as sp
import os
import string

# variables

N = 9
W = N + 2
empty = "\n".join([(N + 1) * ' '] + N * [' ' + N * '.'] + [(N + 2) * ' ']
                  )
colstr = 'ABCDEFGHIJKLMNOPQRST'


def empty_position(a):
    """ Return an initial board position """
    return Position(board=a, cap=(0, 0), n=0, ko=None, last=None, last2=None, komi=7.5)


def parse_coord(s):
    if s.strip() == "pass":
        return "pass"
    else:
        try:
            res = W + 1 + (N - int(s[1:])) * W + colstr.index(s[0].upper())
        except ValueError:
            res = "wi"
        return res


# board string routines
def neighbors(c):
    """ generator of coordinates for all neighbors of c """
    return [c - 1, c + 1, c - W, c + W]


def diag_neighbors(c):
    """ generator of coordinates for all diagonal neighbors of c """
    return [c - W - 1, c - W + 1, c + W - 1, c + W + 1]


def board_put(board, c, p):
    return board[:c] + p + board[c + 1:]


def floodfill(board, c):
    """ replace continuous-color area starting at c with special color # """
    # This is called so much that a bytearray is worthwhile...
    byteboard = bytearray(board)
    p = byteboard[c]
    byteboard[c] = ord('#')
    fringe = [c]
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if byteboard[d] == p:
                byteboard[d] = ord('#')
                fringe.append(d)
    return str(byteboard)


# Regex that matches various kind of points adjecent to '#' (floodfilled) points
contact_res = dict()
for p in ['.', 'x', 'X']:
    rp = '\\.' if p == '.' else p
    contact_res_src = ['#' + rp,  # p at right
                       rp + '#',  # p at left
                       '#' + '.' * (W - 1) + rp,  # p below
                       rp + '.' * (W - 1) + '#']  # p above
    contact_res[p] = re.compile('|'.join(contact_res_src), flags=re.DOTALL)


def contact(board, p):
    """ test if point of color p is adjecent to color # anywhere
    on the board; use in conjunction with floodfill for reachability """
    m = contact_res[p].search(board)
    if not m:
        return None
    return m.start() if m.group(0)[0] == p else m.end() - 1


def is_eyeish(board, c):
    """ test if c is inside a single-color diamond and return the diamond
    color or None; this could be an eye, but also a false one """
    eyecolor = None
    for d in neighbors(c):
        if board[d].isspace():
            continue
        if board[d] == '.':
            return None
        if eyecolor is None:
            eyecolor = board[d]
            othercolor = eyecolor.swapcase()
        elif board[d] == othercolor:
            return None
    return eyecolor


def is_eye(board, c):
    """ test if c is an eye and return its color or None """
    eyecolor = is_eyeish(board, c)
    if eyecolor is None:
        return None

    # Eye-like shape, but it could be a falsified eye
    falsecolor = eyecolor.swapcase()
    false_count = 0
    at_edge = False
    for d in diag_neighbors(c):
        if board[d].isspace():
            at_edge = True
        elif board[d] == falsecolor:
            false_count += 1
    if at_edge:
        false_count += 1
    if false_count >= 2:
        return None

    return eyecolor


class Position(namedtuple('Position', 'board cap n ko last last2 komi')):
    """ Implementation of simple Chinese Go rules;
    n is how many moves were played so far """

    def move(self, c):
        """ play as player X at the given coord c, return the new position """

        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(self.board, c, 'X')
        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)

            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        # Update the position and return

        return Position(board=board.swapcase(), cap=(self.cap[1], capX),
                        n=self.n + 1, ko=ko, last=c, last2=self.last, komi=self.komi)

    def move2(self, c):
        """ play as player X at the given coord c, return the new position """

        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(self.board, c, 'X')
        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)

            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        # Update the position and return
        return Position(board=board.swapcase(), cap=(self.cap[1], capX),
                        n=self.n + 1, ko=ko, last=c, last2=self.last, komi=self.komi)

    def atari(self, c):
        """ play as player X at the given coord c, return the new position """

        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(self.board, c, 'X')

        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None
        capX = capX * 10
        return capX

    def atari2(self, c, board):
        """ play as player X at the given coord c, return the new position """
        # ---------------------PLAYER 1
        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(board, c, 'X')

        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None
        captured = 0
        if c - 1 is not IndexError and c - 1 < 109 and c - 1 > 11:
            captured = self.atari3(c - 1, board)

        if captured is not None:
            capX += captured
        if c + 1 is not IndexError and c + 1 < 109 and c + 1 > 11:
            captured = self.atari3(c + 1, board)
        if captured is not None:
            capX += captured
        if c - W is not IndexError and c - W < 109 and c - W > 11:
            captured = self.atari3(c - W, board)

        if captured is not None:
            capX += captured
        if c + W is not IndexError and c + W < 109 and c + W > 11:
            captured = self.atari3(c + W, board)

        if captured is not None:
            capX += captured

        if c - 1 - W is not IndexError and c - 1 - W < 109 and c - 1 - W > 11:
            captured = self.atari3(c - 1 - W, board)
        if captured is not None:
            capX += captured
        if c + 1 + W is not IndexError and c + 1 + W < 109 and c + 1 + W > 11:
            captured = self.atari3(c + 1 + W, board)
        if captured is not None:
            capX += captured
        if c - 1 + W is not IndexError and c - 1 + W < 109 and c - 1 + W > 11:
            captured = self.atari3(c - 1 + W, board)
        if captured is not None:
            capX += captured
        if c + 1 - W is not IndexError and c + 1 - W < 109 and c + 1 - W > 11:
            captured = self.atari3(c + 1 - W, board)
        if captured is not None:
            capX += captured

        return capX

    def atari3(self, c, board):
        """ play as player X at the given coord c, return the new position """
        # ---------------------PLAYER 2
        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(board, c, 'x')

        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        captured = 0
        if c - 1 is not IndexError and c - 1 < 109 and c - 1 > 11:
            captured = self.atari4(c - 1, board)

        if captured is not None:
            capX += captured
        if c + 1 is not IndexError and c + 1 < 109 and c + 1 > 11:
            captured = self.atari4(c + 1, board)

        if captured is not None:
            capX += captured
        if c - W is not IndexError and c - W < 109 and c - W > 11:
            captured = self.atari4(c - W, board)

        if captured is not None:
            capX += captured
        if c + W is not IndexError and c + W < 109 and c + W > 11:
            captured = self.atari4(c + W, board)

        if captured is not None:
            capX += captured

        return capX

        return capX

    def atari4(self, c, board):
        """ play as player X at the given coord c, return the new position """
        # ---------------------PLAYER 1
        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(board, c, 'X')

        # Test for captures, and track ko       capX = self.cap[0]
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        return capX

    def atari5(self, c, board):
        """ play as player X at the given coord c, return the new position """
        # ---------------------PLAYER 2
        # Test for ko
        capY = 0
        capY = self.cap[0]
        singlecaps = []

        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capY += capcount
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        board = board_put(board, c, 'x')

        # Test for captures, and track ko       capX = self.cap[0]
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None
        capY = capY * 100
        capX = capX * 100

        return capX

    def pass_move(self):
        """ pass - i.e. return simply a flipped position """
        return Position(board=self.board.swapcase(), cap=(self.cap[1], self.cap[0]),
                        n=self.n + 1, ko=None, last=None, last2=self.last, komi=self.komi)

    def checkforward(self, c):
        """ play as player X at the given coord c, return the new position """
        # Test for ko

        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        poin = 0
        temp = []
        board = board_put(self.board, c, 'x')

        captured = self.atari5(c, board)
        if captured is not None:
            poin += captured

        global makan
        for i in range(0, 9):
            satubaris = []
            for j in range(0, 9):
                c = parse_coord(str(string.lowercase[i]) + str(j + 1))
                if self.board[c] == '.':
                    captured = self.atari2(c, board)

                    satubaris.append(captured)

                else:
                    satubaris.append(-100)
            temp.append(satubaris)

        for i in range(0, 9):
            for j in range(0, 9):
                if temp[i][j] != -100:
                    if temp[i][j] is not None:
                        poin += temp[i][j]

        return poin


jalan = None


def lala(sim):
    global makan2
    makan2 = []
    i = 0
    j = 0
    for i in range(0, 9):
        satubaris = []
        for j in range(0, 9):
            c = parse_coord(str(string.lowercase[i]) + str(j + 1))
            if c < len(sim.pos.board):
                if sim.pos.board[c] == '.' and sim.pos.board[c] is not IndexError:
                    captured = sim.pos.atari(c)
                    satubaris.append(captured)
                else:
                    satubaris.append(-100)
            else:
                satubaris.append(-100)
        makan2.append(satubaris)

    for i in range(0, 9):
        for j in range(0, 9):
            if makan2[i][j] is None:
                makan2[i][j] = 0


class Simulation:
    def __init__(self, pos):
        self.pos = pos


def execute2(sim):
    # -- input test (manual input to check rule violation) --
    # --- program ---
    # proc = sp.Popen([cmd, prog, sim.pos.board], stdout=sp.PIPE)
    # sc = proc.stdout.read()
    err = False
    cont = True
    while (cont):
        cont = False
        args = None
        global makan
        makan = []
        for i in range(0, 9):
            satubaris = []
            for j in range(0, 9):
                c = parse_coord(str(string.lowercase[i]) + str(j + 1))
                if c < len(sim.pos.board):
                    if sim.pos.board[c] == '.':
                        captured = sim.pos.atari(c)
                        satubaris.append(captured)

                    else:
                        satubaris.append(-100)
                else:
                    satubaris.append(-100)
            makan.append(satubaris)
        for i in range(0, 9):
            for j in range(0, 9):
                if makan[i][j] is None:
                    makan[i][j] = 0
        terbesar = -100000

        average = 0

        for i in range(0, 9):
            for j in range(0, 9):
                average += makan[i][j]
        average = average / 81
        global makan2
        for i in range(0, 9):
            for j in range(0, 9):

                if makan[i][j] is not None:

                    if makan[i][j] > average:
                        makan[i][j] += makan2[i][j] * 2
                else:

                    makan[i][j] = makan2[i][j]
        terkecil = 0
        indexawal = 0
        indexkedua = 0
        for i in range(0, 9):
            for j in range(0, 9):
                c = parse_coord(str(string.lowercase[i]) + str(j + 1))

                a = c
                if c < len(sim.pos.board):
                    poin = sim.pos.checkforward(c)
                else:
                    poin = 0
                makan[i][j] -= poin
                if makan[i][j] < terkecil:
                    terkecil = makan[i][j]
                    indexawal = i
                    indexkedua = j

        if indexawal - 1 > 0:
            makan[indexawal - 1][indexkedua] += 1

        if indexawal + 1 < 9:
            makan[indexawal + 1][indexkedua] += 1
        if indexawal - 1 > 0 and indexkedua - 1 > 0:
            makan[indexawal - 1][indexkedua - 1] += 1

        if indexawal + 1 < 9 and indexkedua + 1 < 9:
            makan[indexawal + 1][indexkedua + 1] += 1

        if indexawal + 1 < 9 and indexkedua - 1 > 0:
            makan[indexawal + 1][indexkedua - 1] += 1

        if indexawal - 1 > 0 and indexkedua + 1 < 9:
            makan[indexawal - 1][indexkedua + 1] += 1

        if indexkedua - 1 > 0:
            makan[indexawal][indexkedua - 1] += 1

        if indexawal - 1 > 0 and indexkedua + 1 < 9:
            makan[indexawal][indexkedua + 1] += 1

        # -----------------------------------------------------------------------------------------------------------------------------------------------
        for i in range(0, 9):
            for j in range(0, 9):
                z = parse_coord(str(string.lowercase[i]) + str(j + 1))
                if z < len(sim.pos.board):
                    if sim.pos.board[z] == ".":
                        if makan[i][j] > terbesar:
                            terbesar = makan[i][j]
                            indexawal = i
                            indexkedua = j
        while err == True:

            makan[indexawal][indexkedua] -= 500
            terbesar = -100000
            for i in range(0, 9):
                for j in range(0, 9):
                    z = parse_coord(str(string.lowercase[i]) + str(j + 1))
                    if z < len(sim.pos.board):
                        if sim.pos.board[z] == ".":
                            if makan[i][j] > terbesar:
                                terbesar = makan[i][j]
                                indexawal = i
                                indexkedua = j
            err = False
            sc = (str(string.lowercase[indexawal]) + str(indexkedua + 1))

            c = parse_coord(sc)
            if not sim.pos.move(c):
                # rule violation
                cont = True
                args = "err2"
                err = True
            if c < len(sim.pos.board):

                if c is "wi":
                    # wrong input
                    cont = True
                    args = "err0"
                    err = True
                elif sim.pos.board[c] != '.':
                    # position not empty
                    cont = True
                    args = "err1"
                    err = True
                elif not sim.pos.move(c):
                    # rule violation
                    cont = True
                    args = "err2"
                    err = True
                    # -- program --
                    # proc = sp.Popen([cmd, prog, args], stdout=sp.PIPE)
                    # sc = proc.stdout.read()

                    # -- manual input to check rule violation -
            else:
                err = True
                cont = True

        sc = (str(string.lowercase[indexawal]) + str(indexkedua + 1))
        c = parse_coord(sc)
        if not sim.pos.move(c):
            # rule violation
            cont = True
            args = "err2"
            err = True
        if c < len(sim.pos.board):
            if c is "wi":
                # wrong input
                cont = True
                args = "err0"
                err = True
            elif sim.pos.board[c] != '.':
                # position not empty
                cont = True
                args = "err1"
                err = True
            elif not sim.pos.move(c):
                # rule violation
                cont = True
                args = "err2"
                err = True
                # -- program --
                # proc = sp.Popen([cmd, prog, args], stdout=sp.PIPE)
                # sc = proc.stdout.read()

                # -- manual input to check rule violation -
        else:
            err = True
            cont = True
    print(sc)
    return c


def aa(sim):
    print("d2")


a = argv[1]
sim = Simulation(pos=empty_position(a))
sim = Simulation(pos=sim.pos.pass_move())
lala(sim)
sim = Simulation(pos=sim.pos.pass_move())

execute2(sim)


