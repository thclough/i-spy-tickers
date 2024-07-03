# My strategy is to start simple and layer on complexity as needed
# I first start with a start with a a simple Naive Bayes model to:
## 1. Model the problem and gather a baseline for more complex methods
## 2. Gather an indication if the distribution of positive examples contains indicate words, i.e. are there unique words in the
## context window of a stock word that really indicate the word is part of a stock name?
## Does removing stop words lead to greater variation between the 

import joblib
import csv
import numpy as np
from we_have_ml_at_home import deep_learning as dl
import time


def fit_naive_bayes_model(train_csv_path, vocab_len, header=False):

    phi_k_0_counts = np.zeros(vocab_len+1)
    phi_k_1_counts = np.zeros(vocab_len+1)

    # instantiate counters
    num_pos_examples = 0
    num_examples = 0
    num_neg_words = 0
    num_pos_words = 0
    
    with open(train_csv_path) as file:
        reader = csv.reader(file)
        if header:
            next(reader)

        for line in reader:
            num_examples+=1
            label = int(line.pop())
            if label == 0:
                for word_idx in line:
                    word_idx = int(word_idx)
                    if word_idx > -1:
                        num_neg_words += 1
                        phi_k_0_counts[word_idx] += 1
            elif label == 1:  
                num_pos_examples += 1
                for word_idx in line:
                    word_idx = int(word_idx)
                    if word_idx > -1:
                        num_pos_words += 1
                        phi_k_1_counts[word_idx] += 1
    
    phi_1 = num_pos_examples/num_examples

    phi_k_0 = (1+phi_k_0_counts) / (vocab_len + num_neg_words)
    phi_k_1 = (1+phi_k_1_counts) / (vocab_len + num_pos_words)

    phi_k_0[-1] = 1e-15
    phi_k_1[-1] = 1e-15

    return phi_1, phi_k_0, phi_k_1

def predict_from_nb_model(model, idx_matrix):

    phi_1, phi_k_0, phi_k_1 = model 

    # use logarithms to compare the probabilities

    logprob_0 = np.sum(np.log(phi_k_0)[idx_matrix],axis=1) + np.log(1 - phi_1)
    logprob_1 = np.sum(np.log(phi_k_1)[idx_matrix],axis=1) + np.log(phi_1)

    labels = (logprob_1 > logprob_0).astype(int)

    return labels

def nb_class_report(eval_chunk, model):
    # hold classification matrix coordinates (true label, predicted label) -> count
    report_dict = {}

    # separate report dict and labels in case labels are not 0 indexes
    labels = set()

    for X_eval, y_eval in eval_chunk.generate():
        y_pred_eval = predict_from_nb_model(model, X_eval)

        for true_label, pred_label in zip(y_eval, y_pred_eval):
            true_label = true_label[0]
            pred_label
            true_label = int(true_label)
            
            pred_label = int(pred_label)
            report_dict[(true_label, pred_label)] = report_dict.get((true_label, pred_label), 0) + 1

            labels.add(true_label)
            labels.add(pred_label)

    num_labels = len(labels)
    sorted_labels = sorted(list(labels))
    sorted_labels_key = {label: idx for idx, label in enumerate(sorted_labels)}
    
    report = np.zeros((num_labels, num_labels))

    for true_label, pred_label in report_dict:
        pair_count = report_dict[(true_label, pred_label)]
        report_idx = (sorted_labels_key[true_label], sorted_labels_key[pred_label])

        report[report_idx] = pair_count

    return sorted_labels_key, report

def version1():
    vocab_path = "./joblib_objects/vocab"
    train_csv_path = "./data/examples_shuffled_train.csv"
    dev_csv_path = "./data/examples_shuffled_dev.csv"
    test_csv_path = "./data/examples_shuffled_test.csv"

    model_path = "./i-spy-tickers/model/nb_model"
    
    # load in dictionary to for vocab
    word_dict = joblib.load(vocab_path)

    try:
        nb_model = joblib.load(model_path)
    except FileNotFoundError:
        # fit the model
        print("Creating new model")
        nb_model = fit_naive_bayes_model(train_csv_path, len(word_dict), header=True)
        joblib.dump(nb_model, model_path)

    # create chunks
    train_chunk = dl.no_resources.Chunk(chunk_size=10000, train_chunk=True)
    train_chunk.set_data_input_props(train_csv_path, data_selector=np.s_[:-1], skiprows=1)
    train_chunk.set_data_output_props(train_csv_path, data_selector=np.s_[[-1]], skiprows=1)

    dev_chunk = train_chunk.create_linked_chunk(dev_csv_path, dev_csv_path)
    test_chunk = train_chunk.create_linked_chunk(test_csv_path, test_csv_path)

    # get classification reports
    train_report = nb_class_report(train_chunk, nb_model)
    print(train_report[1])
    dev_report = nb_class_report(dev_chunk, nb_model)
    print(dev_report[1])

def version2():
    vocab_path = "./joblib_objects/vocab2"
    train_csv_path = "/Users/Tighe_Clough/Desktop/Programming/Projects/i-spy-tickers/data/examples_v2_shuffled_unique.csv"

    model_path = "./model/nb_model_2"
    
    # load in dictionary to for vocab
    word_dict = joblib.load(vocab_path)

    try:
        nb_model = joblib.load(model_path)
    except FileNotFoundError:
        # fit the model
        print("Creating new model")
        nb_model = fit_naive_bayes_model(train_csv_path, len(word_dict), header=True)
        joblib.dump(nb_model, model_path)

    # create chunks
    train_chunk = dl.no_resources.Chunk(chunk_size=10000, train_chunk=True)
    train_chunk.set_data_input_props(train_csv_path, data_selector=np.s_[1:-1], skiprows=1)
    train_chunk.set_data_output_props(train_csv_path, data_selector=np.s_[[-1]], skiprows=1)

    # get classification reports
    train_report = nb_class_report(train_chunk, nb_model)
    print(train_report[1:])

def main():
    #version1()
    version2()

if __name__ == "__main__":
    main()








