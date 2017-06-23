
import Classes.Base as base
from Classes.Base import Simulation
import Classes.Utility as util
import Classes.Config as conf
import sys

import Functions.func_dummy as p

def game_io(runtype = "def"):
    """ A simple minimalistic text mode UI. """

    sim = Simulation(pos=base.empty_position())
    owner_map = conf.W*conf.W*[0]
    res = []
    while True:
        util.print_pos(sim.pos, sys.stdout, owner_map)

        # board = parse_board(sim.pos.board)
        # print_board(board)

        sc = ''
        if(runtype == 'func'):
            if sim.pos.n % 2 == 0:
                sc = p.foo(sim.pos.board)
            else:
                sc = p.bar(sim.pos.board)
        elif(runtype == 'exec'):
            continue
        else:
            sc = raw_input("Your move: ")

        c = util.parse_coord(sc)
        if c is not None:
            # Not a pass
            if sim.pos.board[c] != '.':
                print('err1 Bad move (not empty point)')
                continue

            if not sim.pos.move(c):
                print('err2 Bad move (rule violation)')
                continue
            sim.pos = sim.pos.move(c)
        else:
            # Pass move
            sim = Simulation(pos=sim.pos.pass_move())

        res = sim.pos.Area()

        if sim.pos.n % 2:
            print("player1 score = ", res[0], "| player2 score = ", res[1])
        else:
            print("player1 score = ", res[1], "| player2 score = ", res[0])

        if util.isDone(sim.pos):
            break

    print('\n')

    util.print_pos(sim.pos, sys.stdout, owner_map)
    if sim.pos.n % 2:
        print("player1 score = ", res[0], "| player2 score = ", res[1])
    else:
        print("player1 score = ", res[1], "| player2 score = ", res[0])

    print("Thanks For Playing")