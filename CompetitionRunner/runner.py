import os
import subprocess as sp
import sys


cmd = "python"
cwd = os.getcwd()
prog = cwd + "\\main.py"
urls = ["\\Dummy\\test.py", "\\Dummy\\test.py", "\\Dummy\\test.py"]

start = 0

for i in range(start, len(urls)):
    for j in range(i+1, len(urls)):
        print(i,j)
        print(urls[i] + " vs " + urls[j])
        proc = sp.Popen([cmd, prog, cmd, cwd + urls[i], cwd + urls[j]])
        # while proc.poll() is None:
        #     l = proc.stdout.readline().strip()  # This blocks until it receives a newline.
        #     print l.strip()
        #     sys.stdout.flush()
        #
        # print proc.stdout.read()
        # sys.stdout.flush()
        proc.wait()
