import sys
import getopt
import pickle
from nltk.tokenize import word_tokenize
from TokenAnalysis import RemoveNoise

tweet = "I wanna go to Brazil"
language = "en"

print("Tweet: " + tweet)
print("Language: " + language)

tokens = RemoveNoise(word_tokenize(tweet))

if language == "en":
    modelFile = open('models/en_leadClassifier.pickle', 'rb')
    print("English model loaded")
elif language == "es":
    modelFile = open('models/es_leadClassifier.pickle', 'rb')
    print("Spanish model loaded")
elif language == "pt":
    modelFile = open('models/pt_leadClassifier.pickle', 'rb')
    print("Portuguese model loaded")

classifier = pickle.load(modelFile)
modelFile.close(),

result = classifier.classify(dict([token, True] for token in tokens))
print("Result: " + result)