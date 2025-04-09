from sys import argv
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pickle import dump
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from random import shuffle


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def createTFIDFVectorsFromTrainData(trainData, analyzer='word', ngram_range=(1, 1)):
    tfIDFVect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    trainTFIdf = tfIDFVect.fit_transform(trainData)
    return trainTFIdf, tfIDFVect


def dumpObjectIntoFile(filePath, dataObject):
    with open(filePath, 'wb') as fileDump:
        dump(dataObject, fileDump)


def fitTrainDataWithClassifier(clf, trainData, trainLabels):
    clf.fit(trainData, trainLabels)
    return clf


def SVMClassifier(trainData, trainLabels):
    svm = LinearSVC()
    svm = fitTrainDataWithClassifier(svm, trainData, trainLabels)
    return svm


def logisticClassifier(trainData, trainLabels):
    logit = LogisticRegression()
    logit = fitTrainDataWithClassifier(logit, trainData, trainLabels)
    return logit


def gradientDescent(trainData, trainLabels):
    sgd = SGDClassifier(loss='perceptron')
    sgd = fitTrainDataWithClassifier(sgd, trainData, trainLabels)
    return sgd


def gradientBoost(trainData, trainLabels):
    gradBoost = GradientBoostingClassifier()
    gradBoost = fitTrainDataWithClassifier(gradBoost, trainData, trainLabels)
    return gradBoost


def multinomialNBClassifier(trainData, trainLabels):
    mNB = MultinomialNB(alpha=0.1)
    mNB = fitTrainDataWithClassifier(mNB, trainData, trainLabels)
    return mNB


def main():
    dataFilePath = argv[1]
    labelFilePath = argv[2]
    classifier = argv[3]
    char_analyzer = 'char'
    char_ngram_range = (2, 5)
    word_analyzer = 'word'
    word_ngram_range = (1, 1)
    trainData = readLinesFromFile(dataFilePath)
    trainLabels = readLinesFromFile(labelFilePath)
    indexes = list(range(len(trainLabels)))
    assert len(trainLabels) == len(trainData)
    shuffle(indexes)
    trainData = [trainData[i] for i in indexes]
    trainLabels = [trainLabels[i] for i in indexes]
    wordTrainTFIdf, wordTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, word_analyzer, word_ngram_range)
    charTrainTFIdf, charTfIdfVect = createTFIDFVectorsFromTrainData(
        trainData, char_analyzer, char_ngram_range)
    combinedTrainTfIdf = hstack([wordTrainTFIdf, charTrainTFIdf])
    if re.search('svm', classifier, re.I):
        classifierToSelect = SVMClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('logistic', classifier, re.I):
        classifierToSelect = logisticClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('multi-nb', classifier, re.I):
        classifierToSelect = multinomialNBClassifier(combinedTrainTfIdf, trainLabels)
    elif re.search('sgd', classifier, re.I):
        classifierToSelect = gradientDescent(combinedTrainTfIdf, trainLabels)
    elif re.search('gradient-boosting', classifier, re.I):
        classifierToSelect = gradientBoost(combinedTrainTfIdf, trainLabels)
    dumpObjectIntoFile('train-vect-pandas-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + classifier + '.pkl', wordTfIdfVect)
    dumpObjectIntoFile('train-vect-pandas-' + char_analyzer + '-' +
                       '-'.join(map(str, char_ngram_range)) + '-' + classifier + '.pkl', charTfIdfVect)
    dumpObjectIntoFile(classifier + '-' + word_analyzer + '-' +
                       '-'.join(map(str, word_ngram_range)) + '-' + char_analyzer + '-' +
                       '-'.join(map(str, char_ngram_range)) + '-pandas.pkl', classifierToSelect)


if __name__ == '__main__':
    main()
