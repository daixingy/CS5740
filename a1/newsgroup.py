import pandas as pd
import re
import string
from collections import defaultdict
import heapq
import numpy as np

# Select n-gram for bag of word
n_gram_bow = (1, 2)
# Select word bank size
bank_size = 10000

def keep_top_k_words(featurized_data, word_bank_dict, count, k = bank_size):
    """ To filter the word bank dictionary to only keep the top k occurances words.
        Returns a word_bank_dict and the new train data features
    
    Inputs:
        featurized_data: The featurized train data that is being converted
        word_bank_dict: The original word bank that was built from training data
        count: The number of occurances of the words in the original word bank
        k: top k to keep
    """
    # The new word bank keeping only top k
    new_word_bank_dict = {}
    
    # Convert map that stores old word bank to new word bank
    store = {}
    
    # Convert the count dictionary to lists of tuple to sort by occurances
    occurances = [(-count[x], x) for x in list(count.keys())]
    # Heapify occurances in decending order
    heapq.heapify(occurances)
    
    # Create new dictionary to only keep top k
    
    # Incremental ID
    word_id = 0
    
    while k > 0 and occurances:
        # Pop the current most occuranced word
        _, word = heapq.heappop(occurances)
        new_word_bank_dict[word] = word_id
        store[word_bank_dict[word]] = word_id
        k -= 1
        word_id += 1
        
    # Recreate the featurized train data with new word bank
    new_featurized_data = []
    for data in featurized_data:
        new_data = []
        for w in data:
            # If the word is kept, append it to the new featurized list
            if w in store:
                new_data.append(store[w])
                
        new_featurized_data.append(new_data)
        
    return new_featurized_data, new_word_bank_dict
    
    
def text_extract(text, stop_words):
    """ Convert one string to useful information in list of strings
    
    Inputs:
        text: one line of text from the data
        stop_words: a set of stop words
    """
    # list of str to return
    res = []
    # Split the text by lines
    lines = text.splitlines()
    
    # Punctuations to be removed by using re.sub
    to_remove = "[" + string.punctuation + "\t" + "]"
    
    # Weird words to be removed
    weird_words = set(["maxaxaxaxaxaxaxaxaxaxaxaxaxaxax", "db", "pts", "pt", "oo", "aaa", "aa"])
    
    # Skip all the lines before the "Lines: ..." line as they are not as useful
    i = 0
    while i < len(lines) and lines[i][:5] != 'Lines':
        i += 1
    i += 1
    
    # Run a loop over the rest of the lines and append str of words into the return list
    while i < len(lines):
        # Strip punctations
        line = re.sub(to_remove, '', lines[i])
        # Split line by space into words
        words = line.split(' ')
        
        # Loop over all words and append the ones to keep to res
        for word in words:
            # Turn into lower case
            lower_word = word.lower()
            # If not a stop word, append to res
            if len(lower_word) > 1 and lower_word.isalpha() and lower_word not in stop_words and lower_word not in weird_words:
                res.append(lower_word)
        i += 1
                
    return res


def vectorize(data, m):
    """ Turns data into a vectorized numpy array
    
    Inputs:
        data: The data being vectorized
        m: Total length of the word bank for data or total number of classes for labels
    """
    # Create numpy array with all 0 in the final vector shape
    data_vec = np.zeros((len(data), m))
    # Loop over each data to create final vector
    for i in range(len(data)):
        for x in data[i]:
            data_vec[i, x] = 1
    
    return data_vec


def n_gram_bag_of_words_featureize(input_data, stop_words, word_bank_dict, n_gram_bow = n_gram_bow):
    """ Convert all data in the input data into vectors, also return word bank
    
    Inputs:
        input_data: The input data
        stop_words: a set of stop words
        word_bank_dict: The word vocab bank
        n_gram_bow: (from, to) n-gram range to use
    """
    
    # If this is the training set and word bank dict is not given. Create the word bank dict
    build_word_bank = word_bank_dict == {}
    # To count the number of occurance, only used when building word bank
    count = defaultdict(int)
    # Incremental ID
    word_id = 0
    
    # The list storing all the (input, label)
    featurized_result = []
    
    # Loop over all the input data
    for i in range(len(input_data)):
        # list of numbers corresponding to the n-gram words
        featurized_line = []
        
        # Convert text to list of words
        words = text_extract(input_data[i], stop_words)
        
        
        # Run a sliding window to do n-gram bag of word on this list of words
        for j in range(len(words) - n_gram_bow[-1] + 1):
            
            # Loop over all the n-gram range
            for n in range(n_gram_bow[0], n_gram_bow[1] + 1):
                # Current sliding window n words
                cur_words = tuple([words[k] for k in range(j, j + n)])

                # If we are building the word bank for training data
                if build_word_bank:
                    if cur_words not in word_bank_dict:
                        # Add the n-gram word to the word bank
                        word_bank_dict[cur_words] = word_id
                        word_id += 1
                    # Count the number of occrance
                    count[cur_words] += 1

                # If the current sliding window words is in the word bank, append to list
                if cur_words in word_bank_dict:
                    featurized_line.append(word_bank_dict[cur_words])
        
        # Append the current data to result list
        featurized_result.append(featurized_line)
        
    # If we are building the word bank, only keep top k occurances words
    if build_word_bank:
        return keep_top_k_words(featurized_result, word_bank_dict, count)

    return featurized_result, word_bank_dict


def newsgroup_featurize(input_data, labels, labels_dict, stop_words, word_bank_dict = {}):
    """ Featurizes an input for the newsgroup domain. Returns a dictionary mapping word to number if not given one.
    Inputs:
        input_data: The input data
    """
    # Make all the labels into class number
    feature_labels = [labels_dict[labels[i]] for i in range(len(labels))]
    
    
    # Using bag of words
    data, word_bank_dict = n_gram_bag_of_words_featureize(input_data, stop_words, word_bank_dict)
    
    # Vectorize the data
    data_vec = vectorize(data, len(word_bank_dict))
    
    # Vectorize the label
    # label_vec = vectorize(feature_labels, len(labels_dict))
    
    return (data_vec, feature_labels), word_bank_dict
    
    
def newsgroup_data_loader(train_data_filename,
                          train_labels_filename,
                          dev_data_filename,
                          dev_labels_filename,
                          test_data_filename):
    """ Loads the data.
    Inputs:
        train_data_filename (str): The filename of the training data.
        train_labels_filename (str): The filename of the training labels.
        dev_data_filename (str): The filename of the development data.
        dev_labels_filename (str): The filename of the development labels.
        test_data_filename (str): The filename of the test data.
    Returns:
        Training, dev, and test data, all represented as a list of (input, label) format.
        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.
    # Load train data and labels
    train_data = pd.read_csv(train_data_filename)['text'].to_list()
    train_labels = pd.read_csv(train_labels_filename)['newsgroup'].to_list()
    
    # Create label dictonary
    unique_labels = list(set(train_labels))
    labels_dict = {unique_labels[i] : i for i in range(len(unique_labels))}
    
    # Load dev data and labels
    dev_data = pd.read_csv(dev_data_filename)['text'].to_list()
    dev_labels = pd.read_csv(dev_labels_filename)['newsgroup'].to_list()
    
    # Load test data
    test_data = pd.read_csv(test_data_filename)['text'].to_list()
    
    # Load stop words
    stop_words = set(pd.read_table("stopwords.txt", header = None).iloc[:,0].to_list())
    
    # TODO: Featurize the input data for all three splits.
    
    # Build word bank and vectorize train data
    train, word_bank_dict = newsgroup_featurize(train_data, train_labels, labels_dict, stop_words)
    
    # Vectorize the dev data
    dev, _ = newsgroup_featurize(dev_data, dev_labels, labels_dict, stop_words, word_bank_dict)
    
    # Vectorize the test data with dummy labels
    dummy = [train_labels[0]] * len(test_data)
    test, _ = newsgroup_featurize(test_data, dummy, labels_dict, stop_words, word_bank_dict)
    
    # Generate a reversed label look up
    reverse_labels_dict = {v:k for k, v in labels_dict.items()}
    
    return train, dev, test, reverse_labels_dict

if __name__ == "__main__":

    train, dev, test, reverse_labels_dict = newsgroup_data_loader("./data/newsgroups/train/train_data.csv",
                                            "./data/newsgroups/train/train_labels.csv",
                                            "./data/newsgroups/dev/dev_data.csv",
                                            "./data/newsgroups/dev/dev_labels.csv",
                                            "./data/newsgroups/test/test_data.csv")

    # print(sorted(reverse_labels_dict.items()))

