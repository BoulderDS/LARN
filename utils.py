from constants import *

import pickle
import numpy as np
import csv
import matplotlib.pyplot as plt
#from cycler import cycler
from collections import Counter
from collections import defaultdict
#import pandas as pd
import seaborn as sns

import nltk
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

np.random.seed(2017)

def load_data(load_path):
    return pickle.load(open(load_path, 'rb'))

def generate_negative_samples(num_traj, span_size, negs, span_data):
    inds = np.random.randint(0, num_traj, negs)
    neg_words = np.zeros((negs, span_size)).astype('int32')
    neg_masks = np.zeros((negs, span_size)).astype('float32')
    for index, i in enumerate(inds):
        rand_ind = np.random.randint(0, len(span_data[i][2]))
        neg_words[index] = span_data[i][2][rand_ind]
        neg_masks[index] = span_data[i][3][rand_ind]
    return neg_words, neg_masks

# parse learned descriptors into a dict
def read_descriptors(desc_file):
    desc_map = {}
    f = open(desc_file, 'r')
    for i, line in enumerate(f):
        line = line.split()
        desc_map[i] = ', '.join(line)
    return desc_map

# read learned trajectories file, adapted from Mohit's RMN code
def read_csv(csv_file):
    reader = csv.reader(open(csv_file, 'r'))
    all_traj = {}
    prev_book = None
    prev_c1 = None
    prev_c2 = None
    total_traj = 0
    for index, row in enumerate(reader):
        if index == 0:
            continue
        book, c1, c2, month, span_index = row[:5]
        if prev_book != book or prev_c1 != c1 or prev_c2 != c2:
            prev_book = book
            prev_c1 = c1
            prev_c2 = c2
            if book not in all_traj:
                all_traj[book] = {}
            all_traj[book][c1+' AND '+c2] = {'distributions': [], 'months': [], 'span_index': []}
            total_traj += 1
        all_traj[book][c1+' AND '+c2]['distributions'].append(np.array(row[5:], dtype='float32'))
        all_traj[book][c1+' AND '+c2]['months'].append(int(month))
        all_traj[book][c1+' AND '+c2]['span_index'].append(int(span_index))
    return all_traj

def month_to_str(month, year_base):
    month -= 1
    if month % 12 + 1 < 10:
        str_month = str(year_base + int(month / 12)) + '-0' + str(month % 12 + 1)
    else:
        str_month = str(year_base + int(month / 12)) + '-' + str(month % 12 + 1)
    return str_month

def str_to_month(str_month, year_base):
    str_month_split = str_month.split('-')
    year = int(str_month_split[0])
    month = int(str_month_split[1])
    return (year - year_base) * 12 + month

def desc_query(desc_sample_file_name, rel, desc_i, month):
    desc_selected_sample_dict = pickle.load(open(desc_sample_file_name, 'rb'))
    month_i = str_to_month(month, year_base)
    for doc in desc_selected_sample_dict['Internation'][rel][(desc_i, month_i)]:
        for doc_sent in doc:
            print('\n\n', doc_sent, '\n\n')

def attn_query(attn_sample_file_name, rel, desc_i, month, word):
    attn_selected_sample_dict = pickle.load(open(attn_sample_file_name, 'rb'))
    month_i = str_to_month(month, year_base)
    for doc in attn_selected_sample_dict['Internation'][rel][(desc_i, month_i)][word]:
        for doc_sent in doc:
            print('\n\n', doc_sent, '\n\n')
        
def calc_desc_sentiment(desc_list):
    sia = SentimentIntensityAnalyzer()
    vader_sentiment_result = sia.polarity_scores(' '.join(desc_list))
    return vader_sentiment_result['compound']
