
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


#%%


# TODO Ideas
## I hate the way you are getting rid of similar power seqeunces (maintain_longest_sequence)

# COMPLETED
# add protected tickers to power sequences if match a sequence
## get rid of stop words for greater word effect and
## include the cur word in the data for reconstruction
# Extract power sequences from titles
# reel in window size maybe to 3

# REJECTED
# make all lowercase
# get rid of sentences, (no fragment sentences), because can get these weird separations


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

def format_ticker_strs(text_str):
    """ Standardizes yahoo finance ticker indicators

    ex. "... (NYSE: AAPL) ..." -> "... (AAPL) ..."

    Args:
        text_str (str) : str (or generic text string) that has ticker indicactor to format
    """
    ticker_str = re.sub(r"\([A-Z]+:\s*([A-Z]+\S*)\)", r'(\1)', text_str)
    ticker_str = re.sub(r"\(([A-Z]+),\s[^a-z\(\(]+\)", r'(\1)', ticker_str)

    return ticker_str

def replace_numeric(text_str):
    """Replace numeric with more meaningful substitutes"""

    # replace time
    time_pattern = r"[\d]{1,2}:\d\d ([apAP]\.?[Mm]\.?)? ([A-Z]{1,4}T)?" 
    text_str = re.sub(time_pattern, " <TIMETIME> ", text_str)

    date_pattern = r"(" + R_MONTHS + r")(\s\d{1,2}(st|th|nd|rd)?)? \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(" + R_MONTHS + r") \d{1,2}(st|th|nd|rd)?( \d{4})?"
    # replace a date with DATE
    text_str = re.sub(date_pattern, " <DATETIME> ", text_str)

    # replace a year with year
    text_str = re.sub(r"\s\d{4}\b", " <YEARTIME> ", text_str)

    # replace a currency with CURRENCY # this is evil
    currency_pattern = r"-?([A-Z]{3,}[^\w\s\.:,]|[^\w\s\.:,]|[A-Z]{3,})-?\d[\d,]*\.?\d*(B|b|M|MM|m|K|k)?"
    text_str = re.sub(currency_pattern, " <CURRENCY> ", text_str)

    # replace a percentage with PERCENTAGE
    text_str = re.sub(r"(\s|^)\S+%", " <PERCENTAGE> ", text_str)

    # replace ordinal with ORDINAL
    text_str = re.sub(r"\d+(st|th|nd|rd)", " <ORDINAL> ", text_str)
        
    # replace hyphen with spaces, separate word conjunctions
    text_str = re.sub(r"[-/]",r" " ,text_str)
    
    # replace the rest of the numbers with QUANTITY
    text_str = re.sub(r"(\s|^)[\d\,]+(\.\d+)?\s"," <QUANTITY> ", text_str) # does nto
    
    return text_str

def misc_cleaning(text_str):
    """Miscellaneous text cleaning"""
    
    # get rid of possessives # THIS IS BIG
    text_str = re.sub(r"\'s", "", text_str)

    # deal with title of address
    text_str = re.sub(r'(Mr)\.|(Mrs)\.|(Ms)\.|(Dr)\.', r'\1\2\3\4', text_str)

    # make sure only apostrophes in the form of single quotes are removed
    text_str = re.sub(r"\s\'|\'\s",' ',text_str)
    
    # separate conjunctions to a prefix and a suffix
    text_str = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])", r" \'" , text_str)

    # replace sentence fragments in parentheses with just the sentence fragment
    text_str = re.sub(r"\(([^\(\)]*[a-z][^\(\)]*)\)", r'\1', text_str)

    # replace ampersands with "and"
    text_str = re.sub(r"\&","and",text_str)

    # website
    website_pattern = r"(http[s]?://)?(www\.)?[a-zA-Z0-9-\.]+\.(net|com)"
    text_str = re.sub(website_pattern, " <WEBSITE> ", text_str)

    # make sure acronyms are joined together
    text_str = re.sub(r"(?<!\w)([A-Z])\.", r"\1", text_str)

    # remove everything else that is not a punctuation and ()
    text_str = re.sub(r"[^\.\?\!;:\w\d\(\)\s<>\']", "", text_str)
    
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
    sentences = re.split(r'(?<=\S[\w>]\s*[\.\?!:;])\s(?=[A-Z])|>>>',text_str) 
    
    return sentences

def clean_sentence(sentence):
    """Removes the punctuation from the string
    Args:
        sentence (str) : string to clean

    Returns:
        clean_text_str (str) : cleaned string
    """

    # remove punctuation
    sentence = re.sub(r"[\.\?\!;:]",' ',sentence)

    # only remove periods at the end of strings
    #sentence = re.sub(r'\.\Z', '', sentence)

    # Remove excess spaces
    clean_sentence = re.sub(r'\s{2,}',' ',sentence)

    return clean_sentence

def clean_company_name(company_name):
    """Format company name in the same way as sentences are formatted for easier lookup"""
    return clean_finance_article(company_name)

def get_ticker_dict(nsdq_path="data/nsdq_screener.csv", 
                    otc_path="data/otc_screener.csv", 
                    mf_path="data/mf_names.csv",
                    sec_source="data/sec_tickers.json"):
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

    return power_sequences, no_ticker_sentence_list, 

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
    
    Not a vocab freq dict because don't want to count before split for data leakagee

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
        
        
        
    """
    
    sentences_idxs = []
    sentences_labels = []

    # Loop through words in sentences
    for sentence in no_ticker_sentence_list:
        if len(sentence) >= min_sen_len:
        
            # convert the sentence to list of corresponding vocab idxs, list for labels
            vocab_idxs = []
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
                #         # can have common acronum
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
                labels.append(label)
                if case_ambiguity and not(cur_word[0]=="<" and cur_word[-1]==">"):
                    cur_word = cur_word.lower()
                vocab_idxs.append(vocabulary[cur_word])
                
                # onto the next word
                cur_word_idx += 1
        
        # pack list if pack_len passed
        if pack_len is not None:  
            for start_idx in range(0, len(labels), pack_len):
                end_idx = min(start_idx+pack_len, len(labels))
                sentences_idxs.append(vocab_idxs[start_idx:end_idx])
                sentences_labels.append(labels[start_idx:end_idx])
        else:
            sentences_idxs.append(vocab_idxs)
            sentences_labels.append(labels)
        
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

                if not (ps_title or ps_body):
                    ps_title_search = search_title_for_companies(nts_title[0], clean_ticker_dict)
                    if ps_title_search:
                        ps_title = ps_title_search
                
                ps = ps_title + ps_body # power sequences
                nts = nts_title + nts_body# not ticker sentences

                # develop the vocabulary
                for sentence in nts:
                    develop_vocab(sentence, vocab_dict, case_ambiguity=case_ambiguity)
                
                # get the indices of the words and corresponding labels
                sentences_idxs, sentences_labels = create_context_examples(vocabulary=vocab_dict, 
                                                        no_ticker_sentence_list=nts, 
                                                        power_sequences=ps, 
                                                        ticker_dict=ticker_dict,
                                                        case_ambiguity=case_ambiguity,
                                                        pack_len=30)
                
                
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

def unique_examples(source_csv_path: str, target_csv_path:str, header_flag=True):
    """Transfer unique examples to a new target_csv_path, many articles are cookie cutter, and replace """

    conn = sqlite3.connect(":memory")
    cur = conn.cursor()

    cur.execute("CREATE TABLE IF NOT EXISTS rows (row TEXT PRIMARY KEY)")

    with open(source_csv_path, mode="r") as source_csv:

        source_reader = csv.reader(source_csv)
        
        if header_flag:
            header = next(source_reader)

        with open(target_csv_path, "w") as target_csv:
            target_writer = csv.writer(target_csv)
            if header_flag:
                target_writer.writerow(header)

            for source_line in source_reader:
                source_line_str = ','.join(source_line)
                try:
                    cur.execute("INSERT INTO rows (row) VALUES (?)", (source_line_str,))
                    target_writer.writerow(source_line)
                except sqlite3.IntegrityError:
                    continue
                
    conn.close()
    
def unique_examples_2(examples_csv_path: str, labels_csv_path: str, examples_target_path: str, labels_target_path: str, header_flag=False):
    """"""
    
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

def pad_uneven_csv_rows(csv_source, csv_target, pad_val=np.nan):
    
    len_longest_row = get_len_longest_row(csv_source)

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
    
    
    examples_prefix = "sequential_v1_examples"
    labels_prefix = "sequential_v1_labels"
    
    seq_dir = "./data/sequential"
    
    example_path = create_csv_transform_path(seq_dir, examples_prefix, "")
    label_path = create_csv_transform_path(seq_dir, labels_prefix, "")
    
    example_unique_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique")
    label_unique_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique")
    
    example_padded_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique_padded")
    label_padded_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique_padded")
    
    example_shuffled_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique_padded_shuffled")
    label_shuffled_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique_padded_shuffled")
    
    vocab_save_path = "./joblib_objects/sequential_vocab"
    # ticker_dict_path = "./joblib_objects/ticker_dict"
    i_start = 0
    
    if create_bool:
        vocab_dict = {}
        # add special tokens first to vocab
        develop_vocab(R_PLACEHOLDERS, vocab_dict)
        
        #nltk.download("stopwords")
        # stopwords = stopwords.words("english")
        # trying out no stopwords because think will be important
        stopwords_list = []
        
        create_data_set(data_path, example_path, label_path, vocab_dict, stopwords_list, i_start, case_ambiguity=True)
        
        joblib.dump(vocab_dict, vocab_save_path)

    if data_bool:
        # get unique entries (based both on example and label)
        unique_examples_2(example_path, label_path, example_unique_path, label_unique_path)
    
        # pad with none to shortest length
        pad_uneven_csv_rows(example_unique_path, example_padded_path, None)
        pad_uneven_csv_rows(label_unique_path, label_padded_path, None)
        
                            
        # shuffle
        rng = np.random.default_rng(seed=100)
        no_resources.shuffle(source=example_padded_path,
                              output=example_shuffled_path, 
                              line_limit=10000,
                              header_bool=False, 
                              rng=rng)
        
        rng = np.random.default_rng(seed=100)
        no_resources.shuffle(source=label_padded_path,
                              output=label_shuffled_path, 
                              line_limit=10000,
                              header_bool=False, 
                              rng=rng)
        
        no_resources.chop_up_csv(example_shuffled_path, {"seq_v1_examples_train.csv":.99, "seq_v1_examples_dev.csv":.005, "seq_v1_examples_test.csv":.005}, header_flag=False, seed=100)
        no_resources.chop_up_csv(label_shuffled_path, {"seq_v1_labels_train.csv":.99, "seq_v1_labels_dev.csv":.005, "seq_v1_labels_test.csv":.005}, header_flag=False, seed=100)
        
if __name__ == "__main__":
    main()
   
#%%
def count_lines(source):
    with open(source) as data_file:
        reader = csv.reader(data_file)
        data_lines = sum(1 for line in reader)
    
    return data_lines

examples_prefix = "sequential_v1_examples"
labels_prefix = "sequential_v1_labels"

seq_dir = "./data/sequential"

example_path = create_csv_transform_path(seq_dir, examples_prefix, "")
label_path = create_csv_transform_path(seq_dir, labels_prefix, "")

example_unique_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique")
label_unique_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique")

example_padded_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique_padded")
label_padded_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique_padded")

example_shuffled_path = create_csv_transform_path(seq_dir, examples_prefix, "_unique_padded_shuffled")
label_shuffled_path = create_csv_transform_path(seq_dir, labels_prefix, "_unique_padded_shuffled")

print(count_lines("data/sequential/seq_v1_examples_train.csv"))
print(count_lines("data/sequential/seq_v1_labels_train.csv"))


#%%

# gathering some sentence that won't break

test = "Insider Sell: EVP, Chief Marketing & Strategy Officer Paul Mcnab Sells 15,004 Shares of .... The reported revenue represents a surprise of +2.04% over the Zacks Consensus Estimate of $45.56 million. With the consensus EPS estimate being $0.06, the EPS surprise was 33.33%. Our team is managing this business effectively, and we are building momentum that we believe will result in performance improvements in 2024. For the year was broadly in line with the market average, at 29%. Continuing our 55-year history of paying dividends back to March 1968. This concludes our prepared remarks. And now we're going to talk."
clean_art = clean_finance_article(test)
splitted = [clean_sentence(sentence).split() for sentence in fragment_sentences(clean_art)]

print(clean_art)
print(splitted)


# perecentages can't do negative
# numbers create excess space


#%%






#%%

examples_prefix = "sequential_v1_examples"
seq_dir = "./data/sequential"
example_path = create_csv_transform_path(seq_dir, examples_prefix, "")

vocab = joblib.load("./joblib_objects/sequential_vocab")

_, long_line = get_len_longest_row(example_path)


#%%

vocab = joblib.load("./joblib_objects/sequential_vocab_test")
rev_vocab = {val:key for key, val in vocab.items()}

# 1,0,0,0,0,1,1,1
# 


for i in [397,195,122,346,46,522,445,162]:
    print(rev_vocab[i])




# %%
rev_vocab[19998]


#%%


print(([(i, rev_vocab[int(word_idx)]) for i, word_idx in enumerate(long_line)]))


#%%

example = [508,509,9,138,512,445,33,515,516,9,518,28,520,407,418,409,508,17,445,33,528,529,27,339,532]
print([rev_vocab[int(word_idx)] for word_idx in example])


#%%

# print this longest line 





#%%

# example_path = "./data/sequential_v1_examples_test.csv"
# label_path = "./data/sequential_v1_labels_test.csv"
# example_unique_path = "./data/sequential_examples_v1_test_unique.csv"
# label_unique_path = "./data/sequential_labels_v1_test_unique.csv"

# unique_examples_2(example_path, label_path, example_unique_path, label_unique_path, header_flag=False)


# #%%
# import numpy as np

# rng = np.random.default_rng(seed=100)

# example_path = "./data/sequential_v1_examples_test.csv"
# label_path = "./data/sequential_v1_labels_test.csv"
# shuffled_example_path = "./data/sequential_v1_examples_shuffled_test.csv"
# shuffled_label_path = "./data/sequential_v1_labels_shuffled_test.csv"


# no_resources.shuffle(example_path, shuffled_example_path, line_limit=10, rng=rng)

# print()

# rng = np.random.default_rng(seed=100)

# no_resources.shuffle(label_path, shuffled_label_path, line_limit=10, rng=rng)


# #%%
# # test chop-up csv
# no_resources.chop_up_csv(shuffled_example_path, {"seq_examples_train.csv":.8, "seq_examples_dev.csv":.1, "seq_examples_test.csv":.1}, header_flag=False, seed=100)
# no_resources.chop_up_csv(shuffled_label_path, {"seq_labels_train.csv":.8, "seq_labels_dev.csv":.1, "seq_labels_test.csv":.1}, header_flag=False, seed=100)


#%%

# ticker_dict = get_ticker_dict()
# clean_ticker_dict = {ticker: clean_company_name(name) for ticker, name in ticker_dict.items()}
# #nltk.download("stopwords")
# stopwords_en = stopwords.words("english")
# data_path = "./data/news.txt"
# i = 0
# counter = 0

# for title, article, i in get_article(data_path, i):
#     print(title)
#     counter += 1

#     # if counter % 1000 == 0:
#     #     print(f"Finished {counter} articles")

#     # clean_art = clean_finance_article(article)
#     # clean_title = clean_sentence(clean_finance_article(title)).split()

#     # clean_s = [clean_sentence(sentence).split() for sentence in fragment_sentences(clean_art)]

#     # ps_body, nts_body = get_power_sequences(clean_s, clean_ticker_dict, stopwords_en)
#     # ps_title, nts_title = get_power_sequences([clean_title], clean_ticker_dict, stopwords_en)
    
#     # if not (ps_title or ps_body):
#     #     ps_title_search = search_title_for_companies(nts_title[0], clean_ticker_dict)
#     #     if ps_title_search:
#     #         ps_title = ps_title_search

#     # ps = ps_title + ps_body
#     # nts = nts_title + nts_body

#     # print(ps)
#     # print(nts)

# print(counter)

#%%
# csv_path = "./data/examples2.csv"
# vocab_path = "./joblib_objects/vocab2"

# vocab = joblib.load(vocab_path)
# vocab_rev = {y:x for x,y in vocab.items()}

# with open(csv_path) as file:
#     reader = csv.reader(file)
#     next(reader)
#     for line in reader:
#         word_idxs = line[1:4] + [line[0]] + line[4:7]
#         label = line[7]
#         print(word_idxs)
#         words = [vocab_rev.get(int(idx), "") for idx in word_idxs]
#         print(vocab_rev[int(line[0])])
#         print(words)
#         print(label)


#%%

# title = "We really don't know what happened with Nvidia"

# clean_title = clean_sentence(clean_finance_article(title)).split()
# ps_title, nts_title = get_power_sequences([clean_title], clean_ticker_dict, stopwords_en)

# print(nts_title)

# ps = search_title_for_companies(nts_title[0], clean_ticker_dict)

# print(ps)
#%%


#%%
## ARTICLE TEST

# ticker_dict = get_ticker_dict()

# data_path = "./data/news.txt"

# raw_article, i, end_flag = get_article(data_path,2206)

# clean_art = clean_finance_article(raw_article)

# clean_s = [clean_sentence(sentence).split() for sentence in fragment_sentences(clean_art)]

# ps, nts = get_power_sequences(clean_s, ticker_dict)

# print(ps)
# print(nts)

# vocab = {}

# develop_vocabulary(nts, vocab)

# write_context_examples(csv_path = "data/examples.csv", 
#                             vocabulary = vocab, 
#                             no_ticker_sentence_list = nts, 
#                             power_sequences = ps, 
#                             ticker_dict = ticker_dict)


#%%
# ticker_article = clean_finance_article(raw_article)

# # print(ticker_article)

# sentences = fragment_sentences(ticker_article)

# # #%%
# # # clean the sentences
# # for sentence in sentences:
# #     print(sentence)
# #     clean = clean_sentence(sentence)
# #     print(clean)

# for sentence in fragment_sentences(raw_article):
#     print(sentence)
#     print(replace_numeric(sentence))

#%%

# this = clean_finance_article("With over (DLTH) $1024 million in 20.2% capital as of (maybe a month earlier) December 2023")

# print(this)

# #%%
# date_pattern = r"(" + R_MONTHS + r") \d{1,2} \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(" + R_MONTHS + r") \d{4}"
# date_pattern2 = r"(" + R_MONTHS + r")(\s\d{1,2}(st|th|nd|rd)?)? \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(" + R_MONTHS + r") \d{1,2}(st|th|nd|rd)?( \d{4})?"
# # replace a date with DATE
# this = re.sub(date_pattern2, "DATE", "We are leaving Sep 20th 2024.")

# print(this)
# #%%
# ticker_dict = get_ticker_dict()
# test_sentence = "Canadian-based mining company B2Gold Corp (NYSE:BTG) is among the high-yield dividend stocks popular among hedge funds. But 20 of these men."

# test = format_ticker_strs(test_sentence)

# print(test)
# test = re.sub(r"\W\d+\W"," QUANTITY ", test)


# print(test)

#%%

# ## PATTERN TEST
# import re

# text = "Our average active PaperPie brand partners for the first quarter totaled 23,200 compared to 32,200 in the first quarter last year, a decrease of 9,000 or 28%."

# currency_pattern = r"-?([A-Z]{3,}[^\w\s\.:,]|[^\w\s\.:,]|[A-Z]{3,})-?\d[\d,]*\.?\d*(B|b|M|MM|m|K|k)?"
# text_str = re.sub(currency_pattern, " CURRENCY ", text)

# print(text_str)


#%%

# # test_sentence = "Old Dominion (NASDAQ: ODFL) also had an easier y/y tonnage comp in February (down 12.4% a year ago) compared to January (down 7.8%)."

# test_sentence = article

# clean_art = clean_finance_article(test_sentence)

# clean_s = [clean_sentence(sentence).split() for sentence in fragment_sentences(clean_art)]

# ps, nts = get_power_sequences(clean_s, ticker_dict)

# print(ps)
# print(nts)


#%%

# test_str= "and there were 42 people there"
# print(re.sub(r"(\s|^)\d+(\.\d+)?\s"," <QUANTITY> ", test_str)) # does nto





#%%
# amalgamate the article (the ending is  up from TRENDING)
## \n replace with ' '
## Don't append TRENDING or Related Qu0ote

# # get rid of quotes
# # capitalization fo the first word of a sentence 

# IDEAS
## try model matching company name you find in article next to ticker, with all mentions of that company name
## 's at end fo power sequences
## have to get rid of .'s and ,'s in ticker names when search
## That match word in your get_company_name_idxs is important


# end of article in this form

# TODO
# how to deal with websites (put in misc cleaning)

# Article title 
# TRENDING
# BODY
#"Related Quotes" (end)


# Developing vocabulary and adding context
# We're going to start with case preserving and just adding words as they are seen
# and if it fails we can do case ambiguous 


## a different vocabulary will be needed for this where every capital word seen also gets a lowercase
# and can create a 
# %%
