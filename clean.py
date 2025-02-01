
#%%
import regex as re
import numpy as np
import csv
import copy
import joblib
import utils
import nltk
from nltk.corpus import stopwords
from contextlib import ExitStack
from we_have_ml_at_home.deep_learning import no_resources
import sqlite3
import json
import os

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


#%%

## GLOBAL VARIABLES


# for use in v1 cleaning using special tokens
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
          "January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]

R_MONTHS = "|".join(MONTHS)

R_PLACEHOLDERS = ["<WEBSITE>", 
                  "<TIMETIME>", 
                  "<DATETIME>", 
                  "<YEARTIME>", 
                  "<CURRENCY>",
                  "<PERCENTAGE>",
                  "<ORDINAL>",
                  "<QUANTITY>"] 

def create_currency_dict():
    """Retrieves currency html table from google developer docs and creates a {symbol:iso_code} dict
    
    Returns:
        cur_dict (dict): {currency symbol: ISO code} dict
    """
    src = "https://developers.google.com/public-data/docs/canonical/currencies_csv"
    
    response = requests.get(src)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table (modify 'table' class if necessary)
    table = soup.find('table')
    
    # Read the table into a DataFrame
    df = pd.read_html(str(table))[0]
    df = df.dropna(subset = "symbol")
    
    cur_dict = {}
    for index, row in df.iterrows():
        symbol = row["symbol"]
        currency = row["currency"]
        if symbol and currency and not symbol.isalpha() and "." not in symbol:
            cur_dict[symbol] = currency
            
    # make sure $ is USD (there are multiple currencies that use $)
    cur_dict["$"] = "USD"
    
    return cur_dict

def load_currency_dict(potential_path):
    """Load the currency dict if available or create a new one and save at potential path"""
    
    try:
        currency_dict = joblib.load(potential_path)
    except FileNotFoundError:
        currency_dict = create_currency_dict()
        joblib.dump(currency_dict, potential_path)
    
    return currency_dict

cur_dict_path = "joblib_objects/currency_dict" 
currency_dict = load_currency_dict(cur_dict_path)

#print(currency_dict)


def format_ticker_strs(text_str):
    """ Standardizes yahoo finance ticker indicators

    ex. "... (NYSE: AAPL) ..." -> "... (AAPL) ..."

    Args:
        text_str (str) : str (or generic text string) that has ticker indicator to format
    """
    ticker_str = re.sub(r"\([A-Z]+:\s*([A-Z]+\S*)\)", r'(\1)', text_str)
    ticker_str = re.sub(r"\(([A-Z]+),\s[^a-z\(\(]+\)", r'(\1)', ticker_str)

    return ticker_str

def replace_numeric(text_str):
    
    # replace all digit with pound sign
    text_str = re.sub(r"\d","#", text_str)
    
    # separate nondigit from numbers (based on the pound sign)
    text_str = re.sub(r"(#(\S*#+)?)", r' \1 ' , text_str)
    
    # replace percent with " percent "
    text_str = text_str.replace("%", " percent ")
    
    # replace common currency symbols
    ## get the currency symbols first
    for symbol, currency in currency_dict.items():
        text_str = text_str.replace(symbol, currency)
        
    return text_str

def misc_cleaning(text_str):
    
    # get rid of possessives # THIS IS BIG
    text_str = re.sub(r"[\'\’]s", "", text_str)
    
    # plural possessives
    text_str = re.sub(r"s[\'\’]", "s", text_str)
    
    # deal with title of address
    text_str = re.sub(r'(Mr)\.|(Mrs)\.|(Ms)\.|(Dr)\.', r'\1\2\3\4', text_str)
    
    # make sure apostrophes only sandwiched between letters
    text_str = re.sub(r"(?<![a-z])'|'(?![a-z])", "", text_str)
    
    # replace sentence fragments in parentheses with just the sentence fragment
    text_str = re.sub(r"\(([^\(\)]*[a-z][^\(\)]*)\)", r'\1', text_str)
    
    # replace ampersands with "and"
    text_str = re.sub(r"\&"," and ",text_str)
    
    # replace queries and connectors with spaces
    # text_str = re.sub(r"(?<![\s#A-Z])[-/:\.\?\=]+(?![A-Z\s#])", r" " ,text_str)
    text_str = re.sub(r"(?<![\s#(\(A-Z)])[-/:\.\?\=]+(?![(A-Z\))\s#])", r" " ,text_str)
    
    # make sure acronyms are joined together
    text_str = re.sub(r"(?<!\w)([A-Z])\.", r"\1", text_str)

    # remove everything else that is not a punctuation and ()
    text_str = re.sub(r"(?<!#)[^\.\'\?\!;:\w\d\(\)\s#]", "", text_str)
    
    # bring together parentheses
    #text_str = re.sub(r"(?<=\([A-Z]+)\s(?=[A-Z]+\))", r"", text_str)
    # result = re.sub(r"\(([A-Z ]+)\)", lambda m: f"({m.group(1).replace(' ', '')})", text)
    text_str = re.sub(r"\(([A-Z# ]+)\)", lambda m: f"({m.group(1).replace(' ', '')})", text_str)
    
    return text_str


def clean_finance_article(text_str):
    """Performs three functions on article to clean before processing"""
    return misc_cleaning(replace_numeric(format_ticker_strs(text_str)))

def fragment_sentences(text_str): # might be something wrong with this
    """Splits body of text (string) into sentences
    
    Args:
        text_str (str) : String to break into sentences

    Returns:
        sentence (list) : Array of sentences
    """
    # Split by sentence using lookbehind
    sentences = re.split(r'(?<=\S[\w>\)]\s*[\.\?!:;])\s(?!#)|>>>',text_str) 
    
    return sentences

def clean_sentence(sentence):
    """Removes the punctuation from the string
    Args:
        sentence (str) : string to clean

    Returns:
        clean_text_str (str) : cleaned string
    """

    # remove punctuation
    sentence = re.sub(r"(?<!#)[\.\?\!;:](?!#)",' ',sentence)
    
    sentence = re.sub(r"\(([A-Z# ]+)\)", lambda m: f"({m.group(1).replace(' ', '')})", sentence)

    # only remove periods at the end of strings
    #sentence = re.sub(r'\.\Z', '', sentence)
    
    # there are really only hyphenated 

    # Remove excess spaces
    clean_sentence = re.sub(r'\s{2,}',' ',sentence)

    return clean_sentence

def clean_company_name(company_name):
    """Format company name in the same way as sentences are formatted for easier lookup"""
    return clean_finance_article(company_name)

def get_ticker_dict(nsdq_path="data/preprocessing/nsdq_screener.csv", 
                    otc_path="data/preprocessing/otc_screener.csv", 
                    mf_path="data/preprocessing/mf_names.csv",
                    sec_source="data/preprocessing/sec_tickers.json"):
    """Creates a dictionary {ticker: company_name} using yfinance, SEC, and OTC data.
    
    Args:
        nsdq_path (str): path to csv file with nasdaq tickers
            see: https://www.nasdaq.com/market-activity/stocks/screener
        otc_path (str) : path to csv file with otc tickers
            see: https://www.otcmarkets.com/research/stock-screener/api/downloadCSV?greyAccess=false&expertAccess=false
        mf_path (str) : path to csv file with mf names
        sec_source(str) : path to sec json file
            
    Returns:
        tickers (dict): {ticker: company name }
    """
    # instantiate list for tickers
    ticker_dict = {}

    amend_ticker_dict_csv(ticker_dict, nsdq_path)

    # add on SEC stock data
    #source = "https://www.sec.gov/files/company_tickers.json"
    #json_dict = utils.get_json_dict(sec_source)
    
    with open(sec_source) as json_file:
        json_dict = json.load(json_file)
    
    for company_dict in json_dict.values():
        ticker = company_dict["ticker"]
        if ticker not in ticker_dict:
            ticker_dict[ticker] = company_dict["title"].title()
    
    # add on SEC mutual fund data
    amend_ticker_dict_csv(ticker_dict, mf_path)

    # add on OTC data
    amend_ticker_dict_csv(ticker_dict, otc_path, make_title=True)

    return ticker_dict

def amend_ticker_dict_csv(ticker_dict, csv_path, ticker_col=0, name_col=1, header=True, make_title=False):
    """Add ticker and name information from csv files to ticker dictionary
    
    Args:
        ticker_dict (dict) : ticker_dict to amend to
        csv_path (str) : path to csv with information
        ticker_col (int, default=0) : column number of ticker
        name_col (int, default=1) : columns number of company name
        header (bool, default=True) : whether csv has header or not, skip if it does have one
        make_title (bool, default=False) : whether or not to transform company name into tutle
    """

    with open(csv_path) as file:
        csv_reader = csv.reader(file)
        if header:
            next(csv_reader) # skip headers
        for line in csv_reader:
            ticker = line[ticker_col]
            name = line[name_col]
            if make_title:
                name = name.title()
            # check if already in the ticker 
            if ticker not in ticker_dict and ticker != "":
                ticker_dict[ticker] = name

def get_power_sequences(article_sentence_lists, ticker_dict, stopwords_list):
    """Searches through tokenized sentences in article for power sequences 

    Args:
        article_sentence_lists (list) : list of list of split sentences (each list corresponds to a sentence)
        ticker_dict (dict) : {ticker: company name} dictionary 
        stopwords (list) : list of common words to exclude from no ticker sents

    Return:
        power_sequences (list) : tokenized lists of how public companies are mentioned 
        no_ticker_sentence_list (list) : article_sentence_list without the ticker indicators
    """
    # instantiate lists to hold power sequences and sentences without tickers
    power_sequences = []
    no_ticker_sentence_list = []

    for sentence_list in article_sentence_lists:
        no_ticker_sentence = []
    
        # search for the ticker indicator by looking through sentence
        for i in range(len(sentence_list)):
            word = sentence_list[i]
            # identify the ticker indicator if it exists
            ticker_form = check_ticker_ind(word)
            
            # get company name if the ticker is real, or return
            if ticker_form:
                company_name = ticker_dict.get(ticker_form, False)
                if company_name:
                    # break the official company name into tokens for search
                    name_words = company_name.split()
                    # find how the company is referred to in the article
                    # set a search flag that will become False once a part of the company name
                    # matches a word before the ticker indicator
                    # set a name_idx that selects words in the company name and searches for them
                    search_flag = True
                    name_idx = 0 
                    
                    while search_flag and name_idx < len(name_words): # should not search out more words than company
                        # get the company name words
                        name_word = name_words[name_idx]
                        # create a new pointer starting at the the word before
                        j=i-1
                        candidate_sequence = []
                        while j >= 0 and search_flag:
                            cur_word = sentence_list[j]
                            candidate_sequence.insert(0,cur_word)

                            # if we've found the earliest word with a company word
                            if name_word == cur_word:
                                search_flag = False
                            # if mentioned with an apostrophe (ex. Apple's)
                            elif name_word + "'s" == cur_word:
                                # get rid of the apostrophe (will search for this when getting context)
                                candidate_sequence[-1] = name_word
                                search_flag = False
                            else:
                                # move to an earlier word
                                j-=1
                            if not search_flag:
                                if candidate_sequence not in power_sequences:
                                    # append s to the final word or first word in context
                                    power_sequences.append(candidate_sequence)
                                if [ticker_form] not in power_sequences:
                                    power_sequences.append([ticker_form])
                            
                        # count up the name_idx if there hasn't been a match
                        # ex. Eastman Kodak is commonly referred to as Kodak, so have to go through words in company name
                        name_idx += 1
            else:
                # if not a ticker indicator append to a new list (without the ticker indicators)
                if word.lower() not in stopwords_list:
                    no_ticker_sentence.append(word)
        # look at algo below to get 
        no_ticker_sentence_list.append(no_ticker_sentence)
    
    power_sequences = maintain_longest_sequences(power_sequences)

    return power_sequences, no_ticker_sentence_list

def search_title_for_companies(title_word_list, ticker_dict, max_search=2):
    """Search for company names in the title_word_list using the names of the company in ticker dict. 
    Return power sequences if sufficient match to company name

    Args:
        title_word_list (list) : list of the words of 
        ticker_dict (dict) : {ticker: company name} dictionary
        max_search (int, default = 2) : how many words have to match before search is stopped
    Returns:
        power_sequences (list) : tokenized lists of how public companies are mentioned 
    """

    power_sequences = []
    start_title_idx = 0
    
    comp_tuples = [(ticker, name.split()) for ticker, name in ticker_dict.items()]
    while start_title_idx < len(title_word_list):
        cand_comp_tuples = []
        cand_seq = []
        # get the title word
        start_title_word = title_word_list[start_title_idx]

        # a company name has to be completed
        if start_title_word[0].isupper():
            
            for ticker, name_list in comp_tuples:
                if start_title_word.lower() == name_list[0].lower():
                    cand_comp_tuples.append((ticker, name_list))

            if len(cand_comp_tuples) > 0:
                seq_idx = 1
                cand_seq.append(start_title_word)

            # if you've gotten to two words, just take the first
            while len(cand_comp_tuples) > 1 and start_title_idx + seq_idx < len(title_word_list):
                if seq_idx == max_search:
                    cand_comp_tuples = [cand_comp_tuples[0]]
                    break
                cur_title_word = title_word_list[start_title_idx + seq_idx]
                cand_seq.append(cur_title_word)

                cand_comp_tuples_past = copy.deepcopy(cand_comp_tuples)
                cand_comp_tuples = []

                for ticker, name_list in cand_comp_tuples_past:
                    if seq_idx < len(name_list):
                        if cur_title_word.lower() == name_list[seq_idx].lower():
                            cand_comp_tuples.append((ticker, name_list))
                seq_idx += 1
                # if only one company append sequence and ticker to power sequence
            if len(cand_comp_tuples) == 1:
                comp_tuple = cand_comp_tuples[0]
                ticker = comp_tuple[0]
                name_list = comp_tuple[1]

                unloaded = False
                # unload rest of the sequence
                while start_title_idx + seq_idx < len(title_word_list) and not unloaded and seq_idx < len(name_list): 
                    cur_title_word = title_word_list[start_title_idx + seq_idx]
                    name_word = name_list[seq_idx]
                    if cur_title_word.lower() == name_word.lower():
                        cand_seq.append(cur_title_word)
                        seq_idx += 1
                    else:
                        unloaded = True
                power_sequences.append(cand_seq)
                power_sequences.append([ticker])
                start_title_idx = start_title_idx + seq_idx
            else:
                start_title_idx += 1
                
            seq_idx = 0

        else:
            start_title_idx += 1

    power_sequences = maintain_longest_sequences(power_sequences)  
    return power_sequences

def maintain_longest_sequences(power_sequences):
    """Maintain longest encompassing sequences 
    ex. if have [['AGNC', 'Investment'], ['AGNC', 'Investment', 'Corp']]
        should just leave [['AGNC', 'Investment', 'Corp']]

    Args:
        power_sequences (list) : list of the power sequences

    """

    # get rid of apostrophes
    candidate_sequences = []
    for sequence in power_sequences:
        if sequence[-1][-2:] == "'s":
            sequence1 = sequence[:]
            sequence1[-1] = sequence[-1][:-2]
            if sequence1 not in candidate_sequences:
                candidate_sequences.append(sequence1)
        else:
            candidate_sequences.append(sequence)

    # new power sequences
    new_power_sequences = []
    
    # subsequences will be listed before sequences
    sorted_sequences = sorted(candidate_sequences, key=lambda x: len(x))

    # subsequences have to be shorter than the longest sequence 
    sorted_candidate_subsequences = []
    for sequence in sorted_sequences:
        if len(sequence) < len(sorted_sequences[-1]):
            sorted_candidate_subsequences.append(sequence)
        else:
            new_power_sequences.append(sequence)
    
    for idx, sequence1 in enumerate(sorted_candidate_subsequences):
        sequence_length = len(sequence1)
        add_flag = True
        for sequence2 in sorted_sequences[idx+1:]:
            if len(sequence2) >= len(sequence1) + 1:
                if sequence1 == sequence2[:sequence_length]:
                    add_flag = False
                    break
        if add_flag:
            new_power_sequences.append(sequence1)

    return new_power_sequences

def check_ticker_ind(word):
    """Checks if word is a ticker indicator in the form (TICKER)
    
    Args:
        word (str) : The word to check

    Return:
        ticker (str) : string (possible ticker) or "" if not in ticker form
    """
    #find_result = re.findall(r"\(([A-Z]+[\S]*)\)", word)
    find_result = re.findall(r"\(([^a-z\(\)]+)\)", word)

    try:
        ticker = find_result[0]
    except: 
        ticker = False
    return ticker

# def develop_vocabulary(no_ticker_sentence_list, vocabulary, case_ambiguity=False):
#     """Add any newly encountered words in the article to the vocabulary 

#     Args:
#         no_ticker_sentence_list (list) : list of list of split sentences (each list corresponds to a sentence) 
#             where ticker indicators are removed
#         vocabulary (dict) : current vocabulary dictionary in the form {word: index_no}
#         case_ambiguity (bool, default=False) : whether to add a lowercase form or capitalized word or not
#     """
#     # Gather current word_idx to start counting
#     if vocabulary:
#         word_idx = max(vocabulary.values()) + 1
#     else:
#         word_idx = 0

#     # loop through words and add to vocabulary
#     for sentence in no_ticker_sentence_list:
#         for word in sentence:
#             # add the word if it is not in the vocab
#             if word not in vocabulary:
#                 vocabulary[word] = word_idx
#                 word_idx += 1
#             # add a lower case version of the word if title
#             if case_ambiguity and word.lower() not in vocabulary:
#                 vocabulary[word.lower()] = word_idx
#                 word_idx += 1
                
                
def develop_vocab(word_list, vocab_dict, case_ambiguity=False):
    """Add any newly encountered words in the article to the vocabulary frequency dictionary, 
    keeps track of word index, frequency, and if part of positive examples
    
    Not a vocab frequency dict because don't want to count before split into train/test sets to avoid data leakage

    Args:
        word_list (list) : list words to add
            where ticker indicators are removed
        vocab_dict (dict) : {word: word_idx}
        case_ambiguity (bool, default=False) : whether to add a lowercase form or capitalized word or not
        
    
    """
    # Gather current word_idx to start counting
    if vocab_dict:
        word_idx = len(vocab_dict)
    else:
        word_idx = 0

    # loop through words and add to vocabulary
    for word in word_list:
        if case_ambiguity:
            if word.lower() not in vocab_dict and word not in vocab_dict: # for specialized tokens
                vocab_dict[word.lower()] = word_idx
                word_idx += 1
        else:
        # add the word if it is not in the vocab
            if word not in vocab_dict:
                vocab_dict[word] = word_idx
                word_idx += 1
        # add a lower case version of the word if title

def create_context_master(context_window_radius=4):
    """create df to to hold examples
    
    Args:
        context_window_radius (int, default=4) : length of context on either side of word of interest. 
    
    Returns:
        context_master : list with column names
    """
    cols = ["target_word"] + [f"word_{position}" for position in range(-context_window_radius, context_window_radius+1) if position != 0] + ["label"]
    return [cols]

def create_context_examples(vocabulary, no_ticker_sentence_list, power_sequences, ticker_dict, case_ambiguity, min_sen_len=0, pack_len=None):
    """For each sentence, convert words into indices and for each word label if a 
        at this point ticker indicators have been removed so any tickers found refers directly to the public company/security
    Finds where companies are not mentioned (negative examples) through lowercase words
        WARNING! Company pronouns will list as negative examples 
        (while having similar context to positive examples)

    Args:
        
        vocabulary (dict) : current vocabulary dictionary in the form of {word: index_no} used to quantify words
        no_ticker_sentence_list (list) : list of list of split sentences (each list corresponds to a sentence) 
            where ticker indicators are removed
        power_sequences (list) : tokenized lists of how public companies are mentioned
        ticker_dict (dict) : {ticker: company name} dictionary
        case_ambiguity (bool, default=False) : lower-case everything
        min_sen_len (int, default=0) : minimum sentence length to add to setences tuple
        pack_len (int, default=None) : max allowed sentence length, compresses into multiple list s
    
    Returns:
        
        sentences_idxs
        sentences_labels
        
        
    """
    
    sentences_idxs = []
    sentences_labels = []

    # Loop through words in sentences
    for sentence in no_ticker_sentence_list:
        if len(sentence) >= min_sen_len:
        
            # convert the sentence to list of corresponding vocab idxs, list for labels
            vocab_idxs = []
            lower_idxs = [] # for lower case company names
            
            labels = []
            
            # loop through the words in the sentence
            cur_word_idx = 0
    
            selected_power_sequences = copy.deepcopy(power_sequences)
            
            # index for the power sequences
            seq_idx = 0
            while cur_word_idx < len(sentence):
                
                cur_word = sentence[cur_word_idx]
                label = 0
                
                # can't search for a title word to be a company if no power sequences
                if (cur_word[0].isupper() or seq_idx > 0) and power_sequences:
                    surviving_power_sequences = []
                    # find power sequences if not on a power sequences
                    for sequence in selected_power_sequences:
                        # check if past the max index
                        if seq_idx <= len(sequence)-1:
                            if cur_word == sequence[seq_idx]:
                                surviving_power_sequences.append(sequence)
                                label = 1
                            elif cur_word.replace("'s", "") == sequence[seq_idx]:
                                # not appending any power sequences because assuming ends after 's
                                label = 1
                    
                    if surviving_power_sequences == []:
                        selected_power_sequences = copy.deepcopy(power_sequences)
                        seq_idx = 0
                    else:
                        selected_power_sequences = copy.deepcopy(surviving_power_sequences)
                        seq_idx += 1
                
                # if label != 1:
                #     if seq_idx != 0:
                #         raise Exception
                #     if cur_word.islower():
                #         label = 0
                #         # reset power sequences
                #         selected_power_sequences = copy.deepcopy(power_sequences)
                #         seq_idx = 0
    
                #     if cur_word.isupper():
                #         # not safe to label 1s in these instances because 
                #         # can have common acronym
                #         # check if it's a stock ticker
                #         selected_power_sequences = copy.deepcopy(power_sequences)
                #         seq_idx = 0
    
                #         company_name = ticker_dict.get(cur_word, False)
                #         if not company_name:
                #             label = 0
    
                # print(cur_word)
                # print(label)
                # print(f"{cur_word} {label}")
                
                # add to the data if can gather a label
                # if label is not None:
                
                # add lower case positive label
                if not case_ambiguity:
                    if label == 1 and not cur_word.isupper(): # make sure not add lowercase tickers
                        # add to vocab if necessary
                        develop_vocab([cur_word.lower()], vocabulary)
                        if not lower_idxs:
                            lower_idxs = vocab_idxs[:] # copy over if the first time
                        lower_idxs.append(vocabulary[cur_word.lower()])
                    elif lower_idxs:
                        lower_idxs.append(vocabulary[cur_word])
                    
                labels.append(label)
                if case_ambiguity and not(cur_word[0]=="<" and cur_word[-1]==">"):
                    cur_word = cur_word.lower()
                vocab_idxs.append(vocabulary[cur_word])
                
                # onto the next word
                cur_word_idx += 1
        
        # pack list if pack_len passed
        # if pack_len is not None:  
        #     for start_idx in range(0, len(labels), pack_len):
        #         end_idx = min(start_idx+pack_len, len(labels))
        #         sentences_idxs.append(vocab_idxs[start_idx:end_idx])
        #         sentences_labels.append(labels[start_idx:end_idx])
        # else:
        # -2 is beginning of sentence and -1 is the end
        if vocab_idxs:
            sentences_idxs.append([-2] + vocab_idxs + [-1])
            sentences_labels.append([-2] + labels + [-1])
            
            # add lower case names for poistives for more robust model
            if lower_idxs:
                sentences_idxs.append([-2] + lower_idxs + [-1])
                sentences_labels.append([-2] + labels + [-1])
        
        # print(sentence)
        # print(test_sentence)
        # print(vocab_idxs)
        # print(labels)
        # print()

    return sentences_idxs, sentences_labels


def format_ticker(ticker_str):
    """Edit a ticker string to make searchable in nasdaq screener
    
    Args:
        ticker_str (str): ticker found in text

    Return:
        formatted_ticker (str): ticker ready for nasdaq search
    """
    return ticker_str.replace(".","/")

def get_article(data_path, i_start):
    """amalgamates article in text data based on starting line in the text file
    
    Args:
        data_path (str) : path to the text file
        i_start (int) : line number 
    
    Returns:
        article (str) : The article as a string
        i_placeholder (int) : Location where the next article begins
        end_flag (bool) : If the article is the last one in the text file or not
    """
    with open(data_path) as data_file:
        
        for _ in range(i_start):
            next(data_file)

        i = i_start

        last_title = next(data_file)[:-1]
        i+=1
        first_trending = next(data_file)
        i+=1
        while True:
            try:
                article_list = []
                title = last_title
                i+=1
                # how many times hit on trending
                TRENDING_counter = 0
                trend_stop = False
                while not trend_stop:
                    # skip to i place holder
                    line = next(data_file)[:-1]
                    i+=1
                    # append punctuation to the article title
                    if line == "TRENDING":
                        i-=1
                        TRENDING_counter +=1
                        last_title = article_list.pop()
                        trend_stop = True
                    elif len(line.split()) >= 4:
                        article_list.append(line)
                        # if reach trending, we know article ends two lines up 
                article = " ".join(article_list) 
                    # join the text data 
            except StopIteration:
                article = " ".join(article_list) 
                yield title, article, i-1
                break

            yield title, article, i-1
        

def create_data_set(data_path, example_path, label_path, vocab_dict, stopwords_list, i_start = 0, case_ambiguity=False):

    ticker_dict = get_ticker_dict()
    clean_ticker_dict = {ticker: clean_company_name(name) for ticker, name in ticker_dict.items()}
    
    counter = 0
    i = i_start
    old_i_place = i
    
    if i == 0:
        mode = "w"
    else:
        mode = "a"

    with open(example_path, mode, newline='') as example_file, open(label_path, mode, newline='') as label_file:
        
        example_writer = csv.writer(example_file)
        label_writer = csv.writer(label_file)
        
        try:
            for title, article, i_place in get_article(data_path, i): # this stays open because in a generator
            
                clean_art = clean_finance_article(article)
                clean_title = clean_sentence(clean_finance_article(title)).split()
                
                clean_s = [clean_sentence(sentence).split() for sentence in fragment_sentences(clean_art)]

                ps_body, nts_body = get_power_sequences(clean_s, clean_ticker_dict, stopwords_list)
                ps_title, nts_title = get_power_sequences([clean_title], clean_ticker_dict, stopwords_list)
                
                # if no companies marked with tickers, search title for plain company names
                if not (ps_title or ps_body):
                    ps_title_search = search_title_for_companies(nts_title[0], clean_ticker_dict)
                    if ps_title_search:
                        ps_title = ps_title_search
                
                ps = list(set([tuple(p) for p in ps_title]).union(set([tuple(p) for p in ps_body]))) # power sequences
                nts = nts_title + nts_body # no ticker sentences (sentences cleaned of tickers)
        
                # develop the vocabulary
                for sentence in nts:
                    develop_vocab(sentence, vocab_dict, case_ambiguity=case_ambiguity)
                
                # get the indices of the words and corresponding labels
                sentences_idxs, sentences_labels = create_context_examples(
                                                        vocabulary=vocab_dict, 
                                                        no_ticker_sentence_list=nts, 
                                                        power_sequences=ps, 
                                                        ticker_dict=ticker_dict,
                                                        case_ambiguity=case_ambiguity
                                                        )
                
                
                # discharge the context master every so often to relieve memory
                example_writer.writerows(sentences_idxs)
                label_writer.writerows(sentences_labels)
                
                old_i_place = i_place

                counter += 1
                if counter % 1000 == 0:
                    #example_file.flush()
                    print(f"Finished {counter} articles")
                
        except:
            print(f"old i place : {old_i_place}")
            print(f"i_place: {i_place}")
    
def unique_examples(examples_csv_path: str, labels_csv_path: str, examples_target_path: str, labels_target_path: str, header_flag=False):
    """Finds unique examples and labels among the data, adds them to a new csv (Removes duplicate example, label pairs), 
    sql used to enforce efficient uniqueness"""
    
    conn = sqlite3.connect(":memory2")
    cur = conn.cursor()
    
    cur.execute("CREATE TABLE IF NOT EXISTS rows (row TEXT PRIMARY KEY)")
    
    with open(examples_csv_path, mode="r") as examples_csv, open(labels_csv_path, mode="r") as labels_csv:
        
        example_reader = csv.reader(examples_csv)
        labels_reader = csv.reader(labels_csv)
            
        source_iterable = zip(example_reader, labels_reader)
        
        with open(examples_target_path, "w") as examples_target, open(labels_target_path, "w") as labels_target:
            examples_writer = csv.writer(examples_target)
            labels_writer = csv.writer(labels_target)
            
            if header_flag:
                header1, header2 = next(source_iterable)
                examples_writer.writerow(header1)
                labels_writer.writerow(header2)
                
            for example, label in source_iterable:
                concat = ",".join(example+label)
                try:
                    cur.execute("INSERT INTO rows (row) VALUES (?)", (concat,))
                    examples_writer.writerow(example)
                    labels_writer.writerow(label)
                except:
                    pass
    conn.close()
    
    os.remove(":memory2")

              
def get_len_longest_row(csv_source):

    len_longest_row = 0

    with open(csv_source) as source_file:
        source_reader = csv.reader(source_file)

        for line in source_reader:
            len_line = len(line)
            if len_line > len_longest_row:
                len_longest_row = len_line
    
    return len_longest_row

def cut_and_pack_csv_rows(csv_source, csv_target, pack_len, max_len=None):
    """Discard lines longer than certain lengths (max_len) 
    and then break them down into multiple lines for efficient storage
    
    Args:
        csv_source (file-like) : source csv file
        csv_target (file-like) : file to write to
        pack_len (int) : max size of line segments when breaking down one example
        max_len (int, default = None) : max example length to include
    
    """
    
    if max_len is not None and pack_len > max_len:
        raise Exception("Pack len should be less than the max length of an example line")
    
    with ExitStack() as stack:

        source_file = stack.enter_context(open(csv_source))
        source_reader = csv.reader(source_file)
        target_file = stack.enter_context(open(csv_target, "w+"))
        target_writer = csv.writer(target_file)
        
        for line in source_reader:
            if max_len is None or len(line) <= max_len:
                
                for start_idx in range(0, len(line), pack_len):
                    end_idx = min(start_idx+pack_len, len(line))
                    
                    target_writer.writerow(line[start_idx:end_idx])

def pad_uneven_csv_rows(csv_source, csv_target, pad_val=np.nan, pad_min=None):
    """Pad the lines shorter than the longest line or pad_min with pad_val
    to stretch to length of longest line or pad min when written to a new csv
    
    Args:
        csv_source (file-like) : source csv file
        csv_target (file-like) : file to write to
        pad_val (variable,default=np.nan) : what value to pad lines with
        pad_min (int, default=None) : how long each line should be after padding (>= len of longest line)
    
    """
    
    len_longest_row = get_len_longest_row(csv_source)
    
    if pad_min is not None:
        if pad_min < len_longest_row:
            raise Exception("If a pad min is passed it must be longer than the length of the longest row")
        len_longest_row = pad_min

    with ExitStack() as stack:

        source_file = stack.enter_context(open(csv_source))
        source_reader = csv.reader(source_file)
        target_file = stack.enter_context(open(csv_target, "w+"))
        target_writer = csv.writer(target_file)

        for line in source_reader:
            pad_vals = (len_longest_row - len(line)) * [pad_val]
            new_line = line + pad_vals
            target_writer.writerow(new_line)


def create_csv_transform_path(directory, prefix, suffix):
    
    return f"{directory}/{prefix}{suffix}.csv"

def main():
    create_bool = True
    data_bool = True

    data_path = "./data/news.txt"
    
    examples_prefix = "sequential_v2_examples"
    labels_prefix = "sequential_v2_labels"
    
    seq_dir = "./data/sequential/v2_stopwords"
    
    example_path = create_csv_transform_path(seq_dir, examples_prefix, "")
    label_path = create_csv_transform_path(seq_dir, labels_prefix, "")
    
    example_unique_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique")
    label_unique_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique")
    
    # example_shuffled_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique_shuffled")
    # label_shuffled_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique_shuffled")
    
    # example_packed_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique_packed")
    # label_packed_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique_packed")
    
    # example_padded_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique_padded")
    # label_padded_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique_padded")
    
    
    vocab_save_path = "./joblib_objects/seq_vocab_stopwords"
    # ticker_dict_path = "./joblib_objects/ticker_dict"
    i_start = 0
    
    if create_bool:
        vocab_dict = {}
        # add special tokens first to vocab
        #develop_vocab(R_PLACEHOLDERS, vocab_dict)
        
        #nltk.download("stopwords")
        #stopwords_list = stopwords.words("english")
        # trying out no stopwords because think will be important
        stopwords_list = []
        
        create_data_set(data_path, example_path, label_path, vocab_dict, stopwords_list, i_start, case_ambiguity=False)
        
        joblib.dump(vocab_dict, vocab_save_path)

    if data_bool:
        # get unique entries (based both on example and label)
        unique_examples(example_path, label_path, example_unique_path, label_unique_path)
        
        train_prob, dev_prob, test_prob = .99, .005, .005
        #train_prob, dev_prob, test_prob = .9, .05, .05
        
        # shuffle and distribute
        no_resources.chop_up_csv(example_unique_path, {f"{examples_prefix}_train.csv":train_prob, f"{examples_prefix}_dev.csv":dev_prob, f"{examples_prefix}_test.csv":test_prob}, header_flag=False, seed=100)
        no_resources.chop_up_csv(label_unique_path, {f"{labels_prefix}_train.csv":train_prob, f"{labels_prefix}_dev.csv":dev_prob, f"{labels_prefix}_test.csv":test_prob}, header_flag=False, seed=100)
        
        # perform same task to each of the segmented data sets
        
        for prefix in (examples_prefix, labels_prefix):
            for suffix in ("_train", "_dev", "_test"):
                
                orig = create_csv_transform_path(seq_dir, prefix, suffix)
                packed = create_csv_transform_path(seq_dir, f"{prefix}{suffix}", "_packed")
                padded = create_csv_transform_path(seq_dir, f"{prefix}{suffix}", "_final")
                
                #pack
                cut_and_pack_csv_rows(orig, packed, pack_len=30, max_len=60)
                
                os.remove(orig)
                
                # pad with None to
                pad_uneven_csv_rows(packed, padded, None, pad_min=30)
                
                os.remove(packed)
                
        # #pack
        # cut_and_pack_csv_rows(example_shuffled_path, example_packed_path, 30, max_len=60)
        # cut_and_pack_csv_rows(label_shuffled_path, label_packed_path, 30, max_len=60)
        
        # # pad with none to
        # pad_uneven_csv_rows(example_packed_path, example_padded_path, None, pad_min=30)
        # pad_uneven_csv_rows(label_packed_path, label_padded_path, None, pad_min=30)
        
        
if __name__ == "__main__":
    main()
   
#%%

# data check to make sure number of sentences add up
def count_starts(csv_name):
  with open(csv_name) as source_file:
    reader = csv.reader(source_file)
    count = 0
    for line in reader:
      if line[0] == "-2":
        count += 1
  return count

uni_examples = "/Users/Tighe_Clough/Desktop/Programming/Projects/i-spy-tickers/data/sequential/sequential_v1_examples_unique.csv"
uni_labels = "/Users/Tighe_Clough/Desktop/Programming/Projects/i-spy-tickers/data/sequential/sequential_v1_labels_unique.csv"

train_examples_final = "/Users/Tighe_Clough/Desktop/Programming/Projects/i-spy-tickers/data/sequential/sequential_v1_examples_test_final.csv"
train_labels_final ="/Users/Tighe_Clough/Desktop/Programming/Projects/i-spy-tickers/data/sequential/sequential_v1_labels_test_final.csv"

raw_examples_train_proto = "/Users/Tighe_Clough/Desktop/Programming/Projects/i-spy-tickers/data/sequential/sequential_v1_examples_proto_train_final.csv"
raw_labels_train_proto = "/Users/Tighe_Clough/Desktop/Programming/Projects/i-spy-tickers/data/sequential/sequential_v1_labels_proto_train_final.csv"


print(count_starts(train_examples_final))
print(count_starts(train_labels_final))





