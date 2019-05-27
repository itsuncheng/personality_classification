import os
import csv
import sys
from sklearn.model_selection import train_test_split

def combine(directory):
	lines = []
	for filename in os.listdir(directory):
		input_file = os.path.join(directory, filename)
		with open(input_file, "r", encoding="utf-8") as f:
			reader = csv.reader(f, delimiter="\t")
			cntr = 0
			for line in reader:
				if sys.version_info[0] == 2:
					line = list(unicode(cell, 'utf-8') for cell in line)
				if len(line) == 2 and len(line[1]) > 50 and line not in lines:
					lines.append(line)
					cntr += 1
				if cntr >= 5000: break
	return lines

def create_combined_dataset(lines):
	train, dev = train_test_split(lines, test_size = 0.15)
	return train, dev

def dataset_exporter(train, dev):
	with open('train.csv', 'w', newline='') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(train)
	csvFile.close()
	with open('dev.csv', 'w', newline='') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(dev)
	csvFile.close()

lines = combine("./dataset")
train, dev = create_combined_dataset(lines)
dataset_exporter(train, dev)