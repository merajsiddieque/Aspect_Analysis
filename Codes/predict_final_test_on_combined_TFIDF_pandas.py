from sys import argv
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load


def createTFIDFVectorsFromTrainData(trainData, analyzer='word', ngram_range=(1, 1)):
    tfIDFVect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    trainTFIdf = tfIDFVect.fit_transform(trainData)
    return trainTFIdf, tfIDFVect


def createTestTfIdf(testData, tfIdfVect):
    return tfIdfVect.transform(testData)


def loadObjectFromFile(filePath):
    with open(filePath, 'rb') as fileLoad:
        return load(fileLoad)


def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(dataList))


def predictOnFeatures(classifier, features):
    return classifier.predict(features)


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def main():
    testFile = argv[1]
    classifier = loadObjectFromFile(argv[2])
    wordTfIdfVect = loadObjectFromFile(argv[3])
    charTfIdfVect = loadObjectFromFile(argv[4])
    predFile = argv[5]
    testData = readLinesFromFile(testFile)
    wordTestTfIdf = createTestTfIdf(
        testData, wordTfIdfVect)
    charTestTfIdf = createTestTfIdf(
        testData, charTfIdfVect)
    combinedTestTfIdf = hstack(
        [wordTestTfIdf, charTestTfIdf])
    testPredictions = predictOnFeatures(classifier, combinedTestTfIdf)
    writeListToFile(predFile, testPredictions)


if __name__ == '__main__':
    main()
