__author__ = 'tylerfolkman'

'''
This file provides the code to read in the raw data and create test and train files by word

Exclude -p, bet-n, bet-v, consume-v, deaf-a
Deaf-a has a .adj thing
Consume-v, bet-v, bet-n have missing ids
-p things are strange...


'''

import re
import pandas as pd
from os import walk
import sys


def import_train(filein="../data/Sval1to2.fix/train-fix.xml"):
    with open(filein, 'r') as infile:
        raw_doc = infile.read()
    instances = raw_doc.split("</instance>")
    data = convert_instances(instances[:-1])
    return data


def import_test(filein="../data/Sval1to2.fix/test-fix.xml"):
    with open(filein, 'r') as infile:
        raw_doc = infile.read()
    instances = raw_doc.split("</instance")
    gold_dict = gold_standards()
    mapping = get_mapping()
    data = convert_test_instances(instances[:-1], gold_dict, mapping)
    return data


def convert_test_instances(instances, gold_dict, mapping):
    data = []
    for instance in instances:
        lextype = re.search(r'<instance id="([a-zA-z-]+).*', instance).group(1).strip()
        if "-p" in lextype or lextype == 'bet-n' or lextype == 'bet-v' or lextype == "consume-v" or lextype == 'deaf-a':
            continue
        id = int(re.search(r'instance id="[a-zA-z-]+\.([0-9]+).*', instance).group(1))
        senseid = get_sense(lextype, id, gold_dict, mapping)
        context = re.search(r'<context>(.+)</context>', instance, re.DOTALL).group(1).strip().replace('\n', ' ')
        woi = re.search(r'<head>(.+)</head>', context).group(1).strip()
        context = context.replace('<head>', '').replace("</head>", '')
        data.append([lextype, id, woi, senseid, context])
    return data


def get_sense(lextype, id, gold_dict, mapping):
    senses = []
    lex_split = lextype.split("-")
    lex = lex_split[0].strip()
    if lex == "floating":
        lex = "float"
    sub_map = mapping[mapping['lex'] == lex]
    lex_dict = gold_dict[lextype]
    gold_senses = lex_dict[id]
    for sense in gold_senses:
        if sense == "P":
            val = 999999
        elif sense == "T":
            val = 999998
        elif sense == "U":
            val = 999997
        else:
            val = int(sub_map[sub_map['gold'] == sense]['senseid'].values)
        senses.append(val)
    return senses


def convert_instances(instances):
    data = []
    for instance in instances:
        lextype = re.search(r'<instance id="([a-zA-z-]+).*', instance).group(1).strip()
        id = int(re.search(r'instance id="[a-zA-z-]+\.([0-9]+).*', instance).group(1))
        senseid = int(re.search(r'<answer instance=".* senseid="[a-zA-z-\.]*([0-9]+).*', instance).group(1))
        context = re.search(r'<context>(.+)</context>', instance, re.DOTALL).group(1).strip().replace('\n', ' ')
        woi = re.search(r'<head>(.+)</head>', context).group(1).strip()
        context = context.replace('<head>', '').replace("</head>", '')
        data.append([lextype, id, woi, senseid, context])
    return data

def gold_standards(dirin="../data/GOLD"):
    (_, _, filenames) = walk(dirin).next()
    gold_dict = {}
    for f in filenames:
        with open("{0}/{1}".format(dirin, f), 'r') as filein:
            iter_dict = {}
            for line in filein:
                colon_split = line.split(":")
                key = int(colon_split[0].strip())
                answers = [v.strip() for v in colon_split[1].split(' or ')]
                iter_dict[key] = answers
        gold_dict[f] = iter_dict
    return gold_dict


def get_mapping(filein="../data/mapping.txt"):
    return pd.read_csv(filein, sep=" ", header=None, names=['lex', 'senseid', 'gold', 'type', 'other1', 'other2'])


cols = ['lextype', 'id', 'woi', 'senseid', 'context']
remove_lexs = ['-p', 'bet-n', 'bet-v', 'consume-v', 'deaf-a']
remove_lexs_p = '|'.join(remove_lexs)
print("Extracting training data to pandas df...")
train_data = import_train()
train_df = pd.DataFrame(train_data, columns=cols)
train_df = train_df[~train_df['lextype'].str.contains(remove_lexs_p)]
train_df.to_pickle("../data/clean/train_df.pkl")

print("Extracting testing data to pandas df...")
test_data = import_test()
test_df = pd.DataFrame(test_data, columns=cols)
test_df.to_pickle("../data/clean/test_df.pkl")

print("Results")
print(train_df['lextype'].value_counts())
print(test_df['lextype'].value_counts())
