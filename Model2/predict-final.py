from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load

def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]

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

# 🔽 Paths
testFilePath = '/kaggle/input/translated-datasets/training-data-telugu.txt'
wordVectPath = '/kaggle/working/train-vect-word-svm.pkl'
charVectPath = '/kaggle/working/train-vect-char-svm.pkl'
classifierPath = '/kaggle/working/classifier-svm.pkl'
outputPredPath = '/kaggle/working/test-predictions-telugu-2.txt'

# 🔄 Prediction steps
testData = readLinesFromFile(testFilePath)
wordVect = loadObjectFromFile(wordVectPath)
charVect = loadObjectFromFile(charVectPath)
clf = loadObjectFromFile(classifierPath)

wordTestTfIdf = createTestTfIdf(testData, wordVect)
charTestTfIdf = createTestTfIdf(testData, charVect)
combinedTestTfIdf = hstack([wordTestTfIdf, charTestTfIdf])

predictions = predictOnFeatures(clf, combinedTestTfIdf)

writeListToFile(outputPredPath, predictions)

print("✅ Predictions saved to:", outputPredPath)
