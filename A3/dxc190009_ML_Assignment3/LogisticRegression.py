import os
import re
import sys

from numpy import *


# function that returns the list of stopwords in a file
def extract_stopwords(stopwords_file):
    stopWords = []
    fil = open(stopwords_file)
    stopWords = fil.read().strip().split()
    return stopWords


# Function to read data files without removing stopWords
def read_withStopWords(folder):
    files = os.listdir(folder)
    dict = {}
    vocab = []
    for f in files:
        fil = open(folder + "/" + f, encoding="ISO-8859-1")
        words = fil.read()
        all_words = words.strip().split()
        dict[f] = all_words
        vocab.extend(all_words)
    return vocab, dict


# Function to read data files by removing stopWords
def read_withoutStopWords(folder, stopWordsF):
    files = os.listdir(folder)
    dict = {}
    vocab = []
    stopWords = extract_stopwords(stopWordsF)
    for f in files:
        fil = open(folder + "/" + f, encoding="ISO-8859-1")
        words = fil.read()
        words = re.sub('[^a-zA-Z]', ' ', words)
        words_all = words.strip().split()
        required_words = []
        for word in words_all:
            if (word not in stopWords):
                required_words.append(word)
        dict[f] = required_words
        vocab.extend(required_words)
    return vocab, dict


# Function to train for MCAP Logistic Regression & returns the weight vector
def LR_train(train_features, labelList, lmbda):
    featureMatrix = mat(train_features)
    p, q = shape(featureMatrix)
    labelMatrix = mat(labelList).transpose()
    eeta = 0.1
    weight = zeros((q, 1))
    number_of_iterations = 100
    for i in range(number_of_iterations):
        predict_condProb = 1.0 / (1 + exp(-featureMatrix * weight))
        error = labelMatrix - predict_condProb
        weight = weight + eeta * featureMatrix.transpose() * error - eeta * lmbda * weight
    return weight


# Function to apply MCAP Logistic Regression on given test set and returns the accuracy
def LR_apply(weight, test_features, length_test_spamDict, length_test_hamDict):
    featureMatrix = mat(test_features)
    res = featureMatrix * weight
    val = 0
    len_allDict = length_test_spamDict + length_test_hamDict
    for i in range(length_test_spamDict):
        if (res[i][0] < 0.0):
            val += 1
    i = 0
    for i in range(length_test_spamDict + 1, len_allDict):
        if (res[i][0] > 0.0):
            val += 1
    return (float)(val / len_allDict) * 100


def feature_vector(all_distinctWords, dict):
    feature = []
    for f in dict:
        row = [0] * (len(all_distinctWords))
        for i in all_distinctWords:
            if (i in dict[f]):
                row[all_distinctWords.index(i)] = 1
        row.insert(0, 1)
        feature.append(row)
    return feature


if __name__ == "__main__":
    if (len(sys.argv) != 7):
        # format for passing the arguments in cmd
        print("Wrong arguments passed ")

        print(
            "Enter in this format: python LogisticRegression.py <Path_to_train\spam> <Path_to_train\ham> <ath_to_test\spam <Path_to_test\ham>",
            "<Path_to_stopWords.txt> <Enter yes/no to add/remove stopWords>")
        sys.exit()

    spam_train = sys.argv[1]
    ham_train = sys.argv[2]
    spam_test = sys.argv[3]
    ham_test = sys.argv[4]
    stopWords = sys.argv[5]
    stopWords_throw = sys.argv[6]
    lmbda = 0.1

    # checking to add/remove stopWords
    if (stopWords_throw == "yes"):
        spamVocab_train, spamDict_train = read_withoutStopWords(spam_train, stopWords)
        hamVocab_train, hamDict_train = read_withoutStopWords(ham_train, stopWords)
    else:
        spamVocab_train, spamDict_train = read_withStopWords(spam_train)
        hamVocab_train, hamDict_train = read_withStopWords(ham_train)

    spamVocab_test, test_spamDict = read_withStopWords(spam_test)
    hamVocab_test, test_hamDict = read_withStopWords(ham_test)

    all_distinctWords = list(set(spamVocab_train) | set(hamVocab_train))
    all_trainDict = spamDict_train.copy()
    all_trainDict.update(hamDict_train)

    all_testDict = test_spamDict.copy()
    all_testDict.update(test_hamDict)

    labelList = []
    for i in range(len(spamDict_train)):
        labelList.append(0)
    i = 0
    for i in range(len(hamDict_train)):
        labelList.append(1)

    train_features = feature_vector(all_distinctWords, all_trainDict)
    test_features = feature_vector(all_distinctWords, all_testDict)

    weight = LR_train(train_features, labelList, lmbda)
    accuracy = LR_apply(weight, test_features, len(test_spamDict), len(test_hamDict))
    print("The Accuracy of Logistic Regression is: ", accuracy)
