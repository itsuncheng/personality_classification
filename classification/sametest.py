import csv
import sys

def sametest():
    lines = []
    lines2 = []
    with open("train.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if len(line) == 2:
                lines.append(line)
    with open("dev.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if len(line) == 2:
                lines2.append(line)

    cntr = 0
    for i in lines2: 
        if i in lines:
            print(i)
            cntr += 1
    print(cntr)

sametest()