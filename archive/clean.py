
#%%
import regex as re
import csv
import requests
import json
import copy
import joblib
import pandas as pd
import utils


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
          "January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]

R_MONTHS = "|".join(MONTHS)

R_PLACEHOLDERS = ["WEBSITE", 
                  "TIMETIME", 
                  "DATETIME", 
                  "YEARTIME", 
                  "CURRENCY",
                  "PERCENTAGE",
                  "ORDINAL",
                  "QUANTITY"] 

def format_ticker_strs(text_str):
    """ Standardizes yahoo finance ticker indicators

    ex. "... (NYSE: AAPL) ..." -> "... (AAPL) ..."

    Args:
        text_str (str) : str (or generic text string) that has ticker indicactor to format
    """
    ticker_str = re.sub(r"\([A-Z]+:\s*([A-Z]+\S*)\)", r'(\1)', text_str)
    ticker_str = re.sub(r"\(([A-Z]+),\s[^a-z\(\(]+\)", r'(\1)', text_str)

    return ticker_str

def misc_cleaning(text_str):
    """Miscellaneous text cleaning"""
    
    # get rid of possessives
    # text_str = re.sub(r"\'s", "", text_str)

    # deal with title of address
    text_str = re.sub(r'(Mr)\.|(Mrs)\.|(Ms)\.|(Dr)\.', r'\1\2\3\4', text_str)

    # make sure only apostrophes in the form of single quotes are removed
    text_str = re.sub(r"\s\'|\'\s",' ',text_str)

    # replace sentence fragments in parentheses with just the sentence fragment
    text_str = re.sub(r"\(([^\(\)]*[a-z][^\(\)]*)\)", r'\1', text_str)

    # replace ampersands with "and"
    text_str = re.sub(r"\&","and",text_str)

    # website
    website_pattern = r"(http[s]?://)?(www\.)?[a-zA-Z0-9-\.]+\.(net|com)"
    text_str = re.sub(website_pattern, " WEBSITE ", text_str)

    # make sure acronyms are joined together
    text_str = re.sub(r"(?<!\w)([A-Z])\.", r"\1", text_str)

    # replace hyphen with spaces
    text_str = re.sub(r"[-/]",r" " ,text_str)

    # remove everything else that is not a punctuation and () '
    text_str = re.sub(r"[^\.\?\!;:\w\d\'\(\)\s]", "", text_str)
    
    return text_str

def replace_numeric(text_str):
    """Replace numeric with more meaningful substitutes"""

    # replace time
    time_pattern = r"[\d]{1,2}:\d\d ([apAP]\.?[Mm]\.?)? ([A-Z]{1,4}T)?" 
    text_str = re.sub(time_pattern, " TIMETIME ", text_str)

    date_pattern = r"(" + R_MONTHS + r")(\s\d{1,2}(st|th|nd|rd)?)? \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(" + R_MONTHS + r") \d{1,2}(st|th|nd|rd)?( \d{4})?"
    # replace a date with DATE
    text_str = re.sub(date_pattern, " DATETIME ", text_str)

    # replace a year with year
    text_str = re.sub(r"\s\d{4}\b", " YEARTIME ", text_str)

    # replace a currency with CURRENCY # this is evil
    currency_pattern = r"-?([A-Z]{3,}[^\w\s\.:,]|[^\w\s\.:,]|[A-Z]{3,})-?\d[\d,]*\.?\d*(B|b|M|MM|m|K|k)?"
    text_str = re.sub(currency_pattern, " CURRENCY ", text_str)

    # replace a percentage with PERCENTAGE
    text_str = re.sub(r"\w+%", " PERCENTAGE ", text_str)

    # replace ordinal with ORDINAL
    text_str = re.sub(r"\d+(st|th|nd|rd)", " ORDINAL ", text_str)

    # replace the rest of the numbers with QUANTITY
    text_str = re.sub(r"(\s|^)\d+\s"," QUANTITY ", text_str)
    
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
    sentences = re.split(r'(?<=\S\w[\.\?!:;])\s(?=[A-Z])|>>>',text_str) 
    
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
    return clean_sentence(company_name)

def get_ticker_dict(nsdq_path="data/nsdq_screener.csv", 
                    otc_path="data/otc_screener.csv", 
                    mf_path="data/mf_names.csv"):
    """Creates a dictionary {ticker: company_name} using yfinance, SEC, and OTC data.
    
    Args:
        nsdq_path (str): path to csv file with nasdaq tickers
            see: https://www.nasdaq.com/market-activity/stocks/screener
        otc_path (str) : path to csv file with otc tickers
            see: https://www.otcmarkets.com/research/stock-screener/api/downloadCSV?greyAccess=false&expertAccess=false
        mf_path (str) : path to csv file with mf names
            
    Returns:
        tickers (dict): {ticker: company name }
    """
    # instantiate list for tickers
    ticker_dict = {}

    amend_ticker_dict_csv(ticker_dict, nsdq_path)

    # add on SEC stock data
    source = "https://www.sec.gov/files/company_tickers.json"

    json_dict = utils.get_json_dict(source)
        
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

def get_power_sequences(article_sentence_lists, ticker_dict):
    """Searches through tokenized sentences in article for power sequences 

    Args:
        article_sentence_lists (list) : list of list of split sentences (each list corresponds to a sentence)
        ticker_dict (dict) : {ticker: company name} dictionary 

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
                    # get the name of the company and format to match article cleaning
                    clean_name = clean_company_name(company_name)
                    # break the official company name into tokens for search
                    name_words = clean_name.split()
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

                            ## IF CANDIDATE SEQUENCE IN POWER SEQUENCES

                            # if we've found the earliest word with a company word
                            if name_word.lower() == cur_word.lower():
                                if candidate_sequence not in power_sequences:
                                    power_sequences.append(candidate_sequence)
                                search_flag = False
                            # if mentioned with an apostrophe (ex. Apple's)
                            # don;t 
                            elif name_word.lower() + "'s" == cur_word.lower():
                                # get rid of the apostrophe (will search for this when getting context)
                                candidate_sequence[-1] = candidate_sequence[-1][:-2]
                                if candidate_sequence not in power_sequences:
                                    # append s to the final word or first word in context
                                    power_sequences.append(candidate_sequence)
                                search_flag = False
                            else:
                                # move to an earlier word
                                j-=1
                        # count up the name_idx if there hasn't been a match
                        # ex. Eastman Kodak is commonly referred to as Kodak, so have to go through words in company name
                        name_idx += 1
            else:
                # if not a ticker indicator append to a new list (without the ticker indicators)
                no_ticker_sentence.append(word)

        # look at algo below to get 
        no_ticker_sentence_list.append(no_ticker_sentence)
    
    power_sequences = maintain_longest_sequences(power_sequences)

    return power_sequences, no_ticker_sentence_list, 

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

def develop_vocabulary(no_ticker_sentence_list, vocabulary, case_ambiguity=False):
    """Add any newly encountered words in the article to the vocabulary 

    Args:
        no_ticker_sentence_list (list) : list of list of split sentences (each list corresponds to a sentence) 
            where ticker indicators are removed
        vocabulary (dict) : current vocabulary dictionary in the form {word: index_no}
        case_ambiguity (bool, default=False) : whether to add a lowercase form or capitalized word or not
    """
    # Gather current word_idx to start counting
    if vocabulary:
        word_idx = max(vocabulary.values()) + 1
    else:
        word_idx = 0

    # loop through words and add to vocabulary
    for sentence in no_ticker_sentence_list:
        for word in sentence:
            # add the word if it is not in the vocab
            if word not in vocabulary:
                vocabulary[word] = word_idx
                word_idx += 1
            # add a lower case version of the word if title
            if case_ambiguity and word.lower() not in vocabulary:
                vocabulary[word.lower()] = word_idx
                word_idx += 1

def create_context_master(context_window_radius=4):
    """create df to to hold examples
    
    Args:
        context_window_radius (int, default=4) : length of context on either side of word of interest. 
    
    Returns:
        context_master : list with column names
    """
    cols = [f"word_{position}" for position in range(-context_window_radius, context_window_radius+1) if position != 0] + ["label"]
    return [cols]

def write_context_examples(vocabulary, no_ticker_sentence_list, power_sequences, ticker_dict, context_master, context_window_radius=4):
    """Add context word numbers to context_master list, and label as 0 (no company mention) or 1 (company mention)
    Find where companies are mentioned (positive examples) using power sequences, or finding a raw ticker symbol
        at this point ticker indicators have been removed so any tickers found refer directly to the public company/security
    Find where companies are not mentioned (negative examples) through lowercase words
        WARNING! Company pronouns will list as negative examples 
        (while having similar context to positive examples)

    Args:
        
        vocabulary (dict) : current vocabulary dictionary in the form of {word: index_no} used to quantify words
        no_ticker_sentence_list (list) : list of list of split sentences (each list corresponds to a sentence) 
            where ticker indicators are removed
        power_sequences (list) : tokenized lists of how public companies are mentioned
        ticker_dict (dict) : {ticker: company name} dictionary
        context_master (list) : list 
        context_window_radius (int, default=4) : length of context on either side of word of interest. 
            ex. context_window of 4 will look at maximum four words on either side of word of interest
    """

    # min sentence length - small sentences do not give much information
    min_sentence_length = context_window_radius

    # Loop through words in sentences
    for sentence in no_ticker_sentence_list:
        if len(sentence) >= min_sentence_length:
            # convert the sentence to list of corresponding vocab idxs
            sentence_vocab_idxs = [vocabulary[word] for word in sentence]
            # pad the sentence vocab idxs with -1s (meaning null space)
            padded_list = [-1]*context_window_radius + sentence_vocab_idxs + [-1]*context_window_radius
            # loop through the words in the sentence
            cur_word_idx = 0

            selected_power_sequences = copy.deepcopy(power_sequences)
            seq_idx = 0
            while cur_word_idx < len(sentence):
                # keep a running window of the word numbers to match the context radius
                # cur_word_idx - index of the word in the original sentence 
                # context_idx - index in the padded list that corresponds to the cur_word_idx
                context_idx = cur_word_idx + context_window_radius
                # set boundaries for the window on the padded list
                min_context_idx = context_idx - context_window_radius
                max_context_idx = context_idx + context_window_radius + 1
                # select context
                context = padded_list[min_context_idx:context_idx] + padded_list[context_idx+1:max_context_idx]
                # select the current word/string in the sentence
                cur_word = sentence[cur_word_idx]
                # label is None by default
                label = None
                # if word is lowercase we treat as negative example
                
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
                
                if cur_word.islower() and label != 1:
                    label = 0
                    # reset power sequences
                    selected_power_sequences = copy.deepcopy(power_sequences)
                    seq_idx = 0

                if cur_word.isupper() and label != 1:
                    # not safe to label 1s in these instances because 
                    # can have common
                    # check if it's a stock ticker
                    selected_power_sequences = copy.deepcopy(power_sequences)
                    seq_idx = 0

                    company_name = ticker_dict.get(cur_word, False)
                    if not company_name:
                        label = 0

                # print(cur_word)
                # print(label)
                # print(f"{cur_word} {label}")
                # add to the data if can gather a label
                if label is not None:
                    # use pandas for efficient handling
                    context_master.append(context + [label])
                    
                # onto the next word
                cur_word_idx += 1

    return context_master

def context_master_to_csv(context_master, csv_path):
    """
    Args:
        csv_path (str) : path to existing data"""
    with open(csv_path, "a", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(context_master)

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
    # placeholder to mark the start of the file
    with open(data_path, mode="r") as file:
        article_list = []
        # end_flag, communicates if end of text file has been reached
        end_flag = True
        # how many times hit on trending
        TRENDING_counter = 0
        # skip to i place holder
        for i, line in enumerate(file):
            line = line[:-1]
            # make sure to only start recording at proper line
            if i >= i_start:
                # append punctuation to the article title
                if i == i_start:
                    line += "."
                if line == "TRENDING":
                    TRENDING_counter +=1
                elif len(line.split()) >= 4:
                    article_list.append(line)
                # if reach trending, we know article ends two lines up 
                if TRENDING_counter == 2:
                    # remove the next article title from the list
                    article_list = article_list[:-1]
                    i_placeholder = i - 1
                    end_flag = False
                    break
                    
            # if the algo has gotten this far without getting to "TRENDING"
            # the end of the text file is reached

    if end_flag: 
        i_placeholder = i_start

    # join the text data 
    article = " ".join(article_list)

    return article, i_placeholder, end_flag

def main():
    data_path = "./data/news.txt"
    csv_path = "./data/examples2.csv"
    vocab_path = "./joblib_objects/vocab2"

    ticker_dict = get_ticker_dict()
    context_master = create_context_master()

    vocab = {}
    counter = 0
    i = 0
    end_flag = False
    #print_flag = True

    while not end_flag:
        i_begin = i 
        article, i, end_flag = get_article(data_path, i)

        clean_art = clean_finance_article(article)

        clean_s = [clean_sentence(sentence).split() for sentence in fragment_sentences(clean_art)]

        ps, nts = get_power_sequences(clean_s, ticker_dict)

        develop_vocabulary(nts, vocab)

        # if "canalWEBSITE" in vocab and print_flag:
        #     print(i_begin)
        #     print_flag = False

        context_master = write_context_examples(vocabulary = vocab, 
                                                no_ticker_sentence_list = nts, 
                                                power_sequences = ps, 
                                                ticker_dict = ticker_dict, 
                                                context_master=context_master)

        counter += 1

        # discharge the context master every so often to relieve memory
        if counter % 1000 == 0:
            context_master_to_csv(context_master, csv_path)
            context_master = []
            print(f"Finished {counter} articles")

    # discharge rest of context master
    context_master_to_csv(context_master, csv_path)
    joblib.dump(vocab, vocab_path)

if __name__ == "__main__":
    main()


#%%
import joblib
my_vocab = joblib.load("joblib_objects/vocab")


 #%%

# for key in my_vocab.keys():
#     for placeholder in R_PLACEHOLDERS:
#         if placeholder in key:
#             print(key)

#%%
# i = 0
# for key in my_vocab:
#     if key[0].isupper() and key.lower() in my_vocab:
#         i +=1

# print(i/len(my_vocab)) # comes out to .141 of dict which isn't crazy

#%%


#%% TODO
# improves stock dictioanry to include yfinance + sec and otc data
# improve currency



# YEARTIME as well
# get rid of 's, without getting rid of it's, everyone's ...
## get rid of it if it starts with a captial
# get rid of case sensitivity
# anything with ant parenthesis is axed

# for context, dictionary.get(word, -1), so if you did axe it you wont get errors 
# at the end anything that isnumeric gets flushed out (not added to dict)
# anything that is not .isalpha


#%%

# ticker_dict = get_ticker_dict()
# data_path = "./data/news.txt"
# i = 0
# counter = 0
# while i < 1000:
#     print(i)
#     article, i, end_flag = get_article(data_path, i)
#     counter +=1

#     clean_art = clean_finance_article(article)

#     clean_s = [clean_sentence(sentence).split() for sentence in fragment_sentences(clean_art)]
    
#     ps, nts = get_power_sequences(clean_s, ticker_dict)

#     print(ps)
#     print(nts)


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
# amalgamate the article (the ending is  up from TRENDING)
## \n replace with ' '
## Don't append TRENDING or Related Qu0ote

# # get rid of quotes
# # capitalization fo the first word of a sentence 

# IDEAS
## try model with getting rid of stop words
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
