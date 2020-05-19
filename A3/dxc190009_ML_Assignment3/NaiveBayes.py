import os
import re
import sys
from collections import Counter
from math import log

all_distinctWords = []


def NB_train(total_spamDocs, total_hamDocs, spamVocab_train, hamVocab_train):
    global all_distinctWords
    prior_spam = (float)(total_spamDocs / (total_spamDocs + total_hamDocs))
    prior_ham = (float)(total_hamDocs / (total_spamDocs + total_hamDocs))

    all_spam_dict = Counter(spamVocab_train)
    all_ham_dict = Counter(hamVocab_train)

    spam_totWords = len(spamVocab_train)
    ham_totWords = len(hamVocab_train)

    all_distinctWords = list(set(all_spam_dict) | set(all_ham_dict))
    num_distinctWords = len(all_distinctWords)

    condprob_spam = {}
    condprob_ham = {}

    for term in all_distinctWords:
        count = 0
        if term in all_spam_dict:
            count = all_spam_dict[term]
        cond_probS = (float)((count + 1) / (spam_totWords + num_distinctWords))
        condprob_spam[term] = cond_probS

    for term in all_distinctWords:
        count = 0
        if term in all_ham_dict:
            count = all_ham_dict[term]
        cond_probH = (float)((count + 1) / (ham_totWords + num_distinctWords))
        condprob_ham[term] = cond_probH

    return prior_spam, prior_ham, condprob_spam, condprob_ham


# apply Naive Bayes on test sets
def NB_apply(prior_spam, prior_ham, condprob_spam, condprob_ham, spamDict_test, hamDict_test):
    global all_distinctWords
    spam_hamDict = [spamDict_test, hamDict_test]
    val = 0
    for i in range(len(spam_hamDict)):
        for j in spam_hamDict[i]:
            spam_p = log(prior_spam)
            ham_p = log(prior_ham)
            for term in spam_hamDict[i][j]:
                if term in all_distinctWords:
                    spam_p = spam_p + log(condprob_spam[term])
                    ham_p = ham_p + log(condprob_ham[term])
            if (spam_p >= ham_p and i == 0):
                val += 1
            elif (spam_p <= ham_p and i == 1):
                val += 1
    return (float)(val / (len(spamDict_test) + len(hamDict_test))) * 100


# function that returns the list of stopwords in a file
def extract_stopWords(stopwords_file):
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
    stopWords = extract_stopWords(stopWordsF)
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


if __name__ == "__main__":
    if (len(sys.argv) != 7):
        # format for passing the arguments in cmd
        print("Wrong arguments passed ")

        print(
            "Enter in this format: python NaiveBayes.py <Path_to_train\spam> <Path_to_train\ham> <ath_to_test\spam <Path_to_test\ham>",
            "<Path_to_stopWords.txt> <Enter yes/no to add/remove stopWords>")
        sys.exit()

    spam_train = sys.argv[1]
    ham_train = sys.argv[2]
    spam_test = sys.argv[3]
    ham_test = sys.argv[4]
    stopWords = sys.argv[5]
    stopWords_throw = sys.argv[6]

    # checking to add/remove stopWords
    if (stopWords_throw == "yes"):
        spamVocab_train, spamDict_train = read_withoutStopWords(spam_train, stopWords)
        hamVocab_train, hamDict_train = read_withoutStopWords(ham_train, stopWords)
    else:
        spamVocab_train, spamDict_train = read_withStopWords(spam_train)
        hamVocab_train, hamDict_train = read_withStopWords(ham_train)

    spamVocab_test, spamDict_test = read_withStopWords(spam_test)
    hamVocab_test, hamDict_test = read_withStopWords(ham_test)

    # total spam docs
    total_spamDocs = len(spamDict_train)
    total_hamDocs = len(hamDict_train)

    prior_spam, prior_ham, condprob_spam, condprob_ham = NB_train(total_spamDocs, total_hamDocs, spamVocab_train,
                                                                  hamVocab_train)

    accuracy = NB_apply(prior_spam, prior_ham, condprob_spam, condprob_ham, spamDict_test, hamDict_test)
    print("The Accuracy for Naive bayes = ", accuracy)
