import time
from sys import argv


# simple executable giving checks on wrong input and tile not empty
time.sleep(2)
if(len(argv) > 1):
    if argv[1] == "err1" or argv[1] == "err0":
        print("a1")
    else:
        print("wow")

