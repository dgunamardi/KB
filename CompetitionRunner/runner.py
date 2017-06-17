import os
import subprocess as sp


cmd = "python"
prog = "C:\\Users\\VR-03\\PycharmProjects\\AICompetition_NUTSHELL\\main.py"
urls = ["C:\\Users\\VR-03\\PycharmProjects\\AICompetition_NUTSHELL\\test.py", "C:\\Users\\VR-03\\PycharmProjects\\AICompetition_NUTSHELL\\test.py"]


for i in range(0, len(urls)):
    for j in range(i+1, len(urls)):
        print(urls[i], j)
        proc = sp.Popen([cmd, prog, cmd, urls[i], urls[j]], stdout=sp.PIPE)
        proc.wait()
        #sc = proc.stdout.read()
        #print(sc)