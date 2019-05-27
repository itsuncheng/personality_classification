import sys
import csv
import re
import os
from cleaning import clean

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if len(line) == 2:
                lines.append(line)
        return lines

lines = read_tsv("../train.csv")

def create_files(lines):
    for line in lines:
        text = line[1]
        text = clean(text)

        label = line[0]
        label = re.sub("[^a-zA-Z]", '', label)

        if (len(label) > 4):
            continue

        directory_name = 'data/input/' + label + '/'
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        f = open(directory_name + "train.txt", "a+")
        f.write(text)

create_files(lines)
