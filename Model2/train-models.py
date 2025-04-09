from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pickle import dump
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
    return fitTrainDataWithClassifier(svm, trainData, trainLabels)


def logisticClassifier(trainData, trainLabels):
    logit = LogisticRegression()
    return fitTrainDataWithClassifier(logit, trainData, trainLabels)


def gradientDescent(trainData, trainLabels):
    sgd = SGDClassifier(loss='perceptron')
    return fitTrainDataWithClassifier(sgd, trainData, trainLabels)


def gradientBoost(trainData, trainLabels):
    gradBoost = GradientBoostingClassifier()
    return fitTrainDataWithClassifier(gradBoost, trainData, trainLabels)


def multinomialNBClassifier(trainData, trainLabels):
    mNB = MultinomialNB(alpha=0.1)
    return fitTrainDataWithClassifier(mNB, trainData, trainLabels)


def main():
    # ✅ Direct file paths (Kaggle)
    dataFilePath = '/kaggle/input/combine-datasets/Combine-training-data.txt'
    labelFilePath = '/kaggle/working/Combine-training-data-labels.txt'
    classifier = 'svm'  # 👈 You can change this to 'logistic', 'multi-nb', 'sgd', etc.

    char_analyzer = 'char'
    char_ngram_range = (2, 5)
    word_analyzer = 'word'
    word_ngram_range = (1, 1)

    trainData = readLinesFromFile(dataFilePath)
    trainLabels = readLinesFromFile(labelFilePath)

    assert len(trainLabels) == len(trainData)

    indexes = list(range(len(trainLabels)))
    shuffle(indexes)
    trainData = [trainData[i] for i in indexes]
    trainLabels = [trainLabels[i] for i in indexes]

    wordTrainTFIdf, wordTfIdfVect = createTFIDFVectorsFromTrainData(trainData, word_analyzer, word_ngram_range)
    charTrainTFIdf, charTfIdfVect = createTFIDFVectorsFromTrainData(trainData, char_analyzer, char_ngram_range)
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
    else:
        raise ValueError("Unsupported classifier selected.")

    # ✅ Save outputs to /kaggle/working/
    dumpObjectIntoFile(f'/kaggle/working/train-vect-word-{classifier}.pkl', wordTfIdfVect)
    dumpObjectIntoFile(f'/kaggle/working/train-vect-char-{classifier}.pkl', charTfIdfVect)
    dumpObjectIntoFile(f'/kaggle/working/classifier-{classifier}.pkl', classifierToSelect)

    print("✅ Model and vectorizers saved to /kaggle/working/")


# ✅ Run it
main()
