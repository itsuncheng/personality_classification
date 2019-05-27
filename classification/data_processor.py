import os
import csv
import sys
import re
from cleaning import clean

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
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

class PersonalityProcessor(DataProcessor):
    def __init__(self, mode):
        self.mode = mode
        self.mode = self.mode.upper()

    def get_train_examples(self, data_dir):
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self.create_examples(self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self, data_dir):
        labels_list = []
        train_examples = self.get_train_examples(data_dir)
        for i in train_examples: 
            if i.label not in labels_list:
                labels_list.append(i.label)
        return labels_list

    def create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if (i == 0): continue
            id_num = "%s-%s" % (set_type, i)
            text = line[1]
            text = clean(text)

            label = line[0]
            label = re.sub("[^a-zA-Z]", '', label)
            label = label.lower()
            if (len(label) > 4): continue
            
            if (self.mode == "E/I" or self.mode == "I/E"): label = label[0]
            elif (self.mode == "N/S" or self.mode == "S/N"): label = label[1]
            elif (self.mode == "T/F" or self.mode == "F/T"): label = label[2]
            elif (self.mode == "J/P" or self.mode == "P/J"): label = label[3]

            examples.append(InputExample(guid=id_num, text=text, label=label))
        return examples

