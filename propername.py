import csv
import os 
import sys
from collections import Counter 

def propername_featurize(input_data, n, type, word_bank, m=0):
    """ Featurizes an input for the proper name domain.
        type could be 'bow' or 'char'
    Inputs:
        input_data: The input data.
    """
    # open stopwords
    f = open("./stopwords.txt")
    stopwords = [x.strip().lower() for x in f]

    # TODO: Implement featurization of input.
    label_list = ["person","place","movie","drug","company"]
    feature_labels = {}
    # tag 5 labels respectively with numbers from 0 to 4 in order
    for i in range(len(label_list)):
        feature_labels[label_list[i]] = i

    if type == 'char':
        return char_ngrams_featurize(input_data, feature_labels, n, word_bank)
    elif type == 'bow':
        return bow_featurize_ngrams(input_data, feature_labels,stopwords, n,word_bank)
    else:
        return mixed_mn_gram(input_data,feature_labels,m,n,type, word_bank)

def mixed_mn_gram(input_data,feature_labels, m,n,type,word_bank):

    # print(word_bank)

    all_characters,label_tag = [],[]
    #create a list of strings with only characters
    for i in input_data:
        if i and i[1] in feature_labels:
            label_tag.append(feature_labels[i[1]])
        else: 
            label_tag.append(-1)
        pure_char = ''.join(filter(str.isalpha, i[0]))
        all_characters.append(pure_char)
    # create the list of tuples where each tuple consists of vectorized data and label tag
    featurized_vectors = []
    for a,b in zip(all_characters, label_tag):
        single_vector = [0]*len(word_bank)
        for l in range(m,n+1):
            j = 0
            temp = ''     
            while j < len(a)-(l-1):
                temp = ''
                for c in range(l):
                    temp = temp + a[j+c]
                if temp in word_bank:
                    index = word_bank.index(temp) 
                    single_vector[index] = 1
                # single_vector[index] += 1
                j+=1
        featurized_vectors.append((single_vector,b))
    return featurized_vectors



def get_wordbank(input_data, n, k, type, m=0):

    # open stopwords
    f = open("./stopwords.txt")
    stopwords = [x.strip().lower() for x in f]
    count = {}

    if type == 'mixed':  
        all_characters = []
        for l in range(m,n+1):
            #create a list of strings with only characters
            for i in input_data:
                pure_char = ''.join(filter(str.isalpha, i[0]))
                all_characters.append(pure_char)
            # create a comprehensive list of n grams
            for i in all_characters:
                j = 0
                temp = ''
                while j < len(i)-(n-1):
                    temp = ''
                    for c in range(l):
                        temp = temp+i[j+c]
                    if temp not in count:
                        count[temp] = 1
                    else:
                        count[temp] += 1
                    j += 1         
    elif type == 'char' :
        all_characters = []
        #create a list of strings with only characters
        for i in input_data:
            pure_char = ''.join(filter(str.isalpha, i[0]))
            all_characters.append(pure_char)
        # create a comprehensive list of n grams
        for i in all_characters:
            j = 0
            temp = ''
            while j < len(i)-(n-1):
                temp = ''
                for c in range(n):
                    temp = temp+i[j+c]
                if temp not in count:
                    count[temp] = 1
                else:
                    count[temp] += 1
                j += 1
    else:
        for i in range(len(input_data)):
            temp = input_data[i][0].split()
            for j in temp:
                curr = ''.join(filter(str.isalpha, j)).lower()   
                if curr not in stopwords:
                    if (curr not in count):
                        count[curr] = 1
                    else:
                        count[curr] += 1
    print(len(count.keys()))
    kt = Counter(count)
    print(kt.most_common(4))
    word_bank = [i for (i,k) in kt.most_common(k)]
    return word_bank


def bow_featurize_ngrams(input_data, feature_labels, stopwords, n,word_bank):

    all_characters,featurized_vectors,label_tag = [[]]*len(input_data),[],[]
    
    for i in range(len(input_data)):
        if input_data[i] and input_data[i][1] in feature_labels.keys():
            label_tag.append(feature_labels[input_data[i][1]])
        else:
            label_tag.append(-1)
        temp = input_data[i][0].split()

        all_characters[i] = []
        for j in temp:
            curr = ''.join(filter(str.isalpha, j)).lower()
            all_characters[i].append(curr)

    for (a,b) in zip(all_characters,label_tag):
        j = 0
        temp = ''
        single_vector = [0]*len(word_bank)
        for j in a:
            if j not in stopwords and j in word_bank:
                index = word_bank.index(j)
                single_vector[index] = 1
        featurized_vectors.append((single_vector,b))

    return featurized_vectors


def char_ngrams_featurize(input_data, feature_labels, n,word_bank):

    all_characters,label_tag = [],[]
    #create a list of strings with only characters
    for i in input_data:
        if i and i[1] in feature_labels:
            label_tag.append(feature_labels[i[1]])
        else: 
            label_tag.append(-1)
        pure_char = ''.join(filter(str.isalpha, i[0]))
        all_characters.append(pure_char)
    # create the list of tuples where each tuple consists of vectorized data and label tag
    featurized_vectors = []
    for a,b in zip(all_characters, label_tag):
        j = 0
        temp = ''
        single_vector = [0]*len(word_bank)
        while j < len(a)-(n-1):
            temp = ''
            for c in range(n):
                temp = temp + a[j+c]
            if temp in word_bank:
                index = word_bank.index(temp) 
                single_vector[index] = 1
            # single_vector[index] += 1
            j+=1
        featurized_vectors.append((single_vector,b))
    return featurized_vectors


def propername_data_loader(train_data_filename,
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
        Training, dev, and test data, all represented as (input, label) format.
        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.

    train_data = read_csv(train_data_filename)
    train_labels = read_csv(train_labels_filename)
    dev_data = read_csv(dev_data_filename)
    dev_labels = read_csv(dev_labels_filename)
    test_data = read_csv(test_data_filename)
    test_labels = ["what"] * len(test_data)
    
    
    train,dev,test = [],[],[]
    for a,b in zip(train_data,train_labels):
        train.append((a[1],b[1]))
    for a,b in zip(dev_data,dev_labels):
        dev.append((a[1],b[1]))
    for a,b in zip(test_data,test_labels):
        test.append((a[1],b))
    # print(len(train))
    # TODO: Featurize the input data for all three splits.
    # return train,dev,test
    m_gram = 2
    n_gram = 3
    word_bank = get_wordbank(train,n_gram,10000, 'mixed', m_gram)
    final_train_data = propername_featurize(train,n_gram, 'mixed',word_bank,m_gram)
    final_dev_data = propername_featurize(dev,n_gram,'mixed',word_bank,m_gram)
    final_test_data = propername_featurize(test,n_gram,'mixed',word_bank,m_gram)
    return  final_train_data,final_dev_data, final_test_data
    # return propername_featurize(train)
    

def read_csv(path):
    file_values = open(path,"r")
    reader= csv.reader(file_values)
    data= []
    for a in reader:
        if a[0] != 'id':
            data.append(a)
    return data


if __name__ == "__main__":
    # # print(read_csv("./data/propernames/dev/dev_data.csv"))
    train, dev, test = propername_data_loader("./data/propername/train/train_data.csv","./data/propername/train/train_labels.csv","./data/propername/dev/dev_data.csv","./data/propername/dev/dev_labels.csv","./data/propername/test/test_data.csv")
    # # input_data = [('abcd','place'),('efgh','person'),('pqrs xyz mn','drug'),('efg,hi-jk','place')]
    # test_data = [('abdc','what'),('efrs','what'),('pqrs opi mn','what')]
    # word_bank = get_wordbank(input_data,2,'bow')
    # print(word_bank)
    # print(propername_featurize(test_data, 2, 'bow',word_bank))
    # input_data = [("mama mia hey",'place'),("isn't it good","place"),("mama mia hey",'place') ]
    # print(get_wordbank(input_data, 3, 1000, "mixed", 2))
    # print(propername_featurize(input_data, 3, "mixed", word_bank, 2))
    # get_wordbank(train, 3, 100, "char", 0)