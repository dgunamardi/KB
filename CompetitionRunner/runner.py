import os
import subprocess as sp
import sys


cmd = "python"
cwd = os.getcwd()
prog = cwd + "\\main.py"
url1 = [
    "\\Players\\RE\\bukanre.py",
    "\\Players\\Her\\go-block.py" #5
    # "\\Players\\Jarvis\\jarvisgo.py", #8
    # "\\Players\\Baymax\\baymax.py", #1
    # "\\Players\\Biljo\\biljo.py", #2
    # "\\Players\\Doraemon\\doraemon.py", #3
    # "\\Players\\ExMachina\\exmachina.py", #4
    # "\\Players\\ImitationGame\\imitationgame.py", #6
    # "\\Players\\Ironman\\ironman.py", #7
    # "\\Players\\MetalSlug\\SlugGo.py", #9
    # "\\Players\\Singularity\\singularity.py", #10
    # "\\Players\\Spongebob\\ai.py", #11
    # "\\Players\\StarWars\\Chess_Go.py", #12
    # "\\Players\\WallE\\walle.py", #13
    # "\\Players\\iRobot\\iRobot.py", #14
    # "\\Players\\Terminator\\terminator.py",
    # "\\Players\\Tron\\tron.py",
    # "\\Players\\Chappie\\chappie.py"
]

url2 = [
    "\\Players\\RE\\bukanre.py",
    "\\Players\\Her\\go-block.py", #5
    "\\Players\\Jarvis\\jarvisgo.py", #8
    "\\Players\\Baymax\\baymax.py", #1
    "\\Players\\Biljo\\biljo.py", #2
    "\\Players\\Doraemon\\doraemon.py", #3
    "\\Players\\ExMachina\\exmachina.py", #4
    "\\Players\\ImitationGame\\imitationgame.py", #6
    "\\Players\\Ironman\\ironman.py", #7
    "\\Players\\MetalSlug\\SlugGo.py", #9
    "\\Players\\Singularity\\singularity.py", #10
    "\\Players\\Spongebob\\ai.py", #11
    "\\Players\\StarWars\\Chess_Go.py", #12
    "\\Players\\WallE\\walle.py", #13
    "\\Players\\iRobot\\iRobot.py", #14
    "\\Players\\Terminator\\terminator.py",
    "\\Players\\Tron\\tron.py",
    "\\Players\\Chappie\\chappie.py"
]

start = 0

for i in range(start, len(url1)):
    for j in range(i+1, len(url2)):
        print(i,j)
        print(url1[i] + " vs " + url2[j])
        proc = sp.Popen([cmd, prog, cmd, cwd + url1[i], cwd + url2[j]])
        # while proc.poll() is None:
        #     l = proc.stdout.readline().strip()  # This blocks until it receives a newline.
        #     print l.strip()
        #     sys.stdout.flush()
        #
        # print proc.stdout.read()
        # sys.stdout.flush()
        proc.wait()
        proc.kill()
