# !pip install --upgrade gspread
# !pip install langid fasttext textblob
import sys
import os 
import getopt
import pandas as pd
import random
import nltk
import pickle
from nltk import classify, NaiveBayesClassifier
from TokenAnalysis import LemmatizeSentence, RemoveNoise, GetAllWords, GetTweetsForModel
from TokenAnalysis import DefineStopWords, GetData, GetCleanedTokenList, GetTokensForModel
from PreProcessing import PreProcessData
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptronTagger')

# python Train.py data/ en 10295
argv = sys.argv[1:]
try:
    # Define the parameters
    opts, args = getopt.getopt(argv, 'data:language:limit', ['foperand', 'soperand'])
    if len(args) != 3:
      print ('usage: Train.py -data <data_root> -language <en_or_es_or_pt> -limit <data_limit>')
    else:
        dataRoot = str(args[0]) # data/
        language = str(args[1]) # en/es/pt
        limit = int(args[2]) # 10295
        dataSplit = 0.7 # 0.7 train - 0.3 test
        print("Language: " + language)

        stopwords = DefineStopWords(language)

        # Create data/en_ or es_ or pt_ files
        PreProcessData(dataRoot, language, limit)        

        # Get the data in data/en_p or es_ or pt_ files
        positiveTweets, negativeTweets = GetData(dataRoot, language)
        positiveTokensForModel, negativeTokensForModel = GetTokensForModel(positiveTweets, negativeTweets, stopwords)
        
        # Get the positive and negative datasets 
        positiveDataset = [(tweetDict, "Positive") for tweetDict in positiveTokensForModel]
        random.shuffle(positiveDataset)

        negativeDataset = [(tweetDict, "Negative") for tweetDict in negativeTokensForModel]
        random.shuffle(negativeDataset)
        
        dataset = positiveDataset + negativeDataset
        random.shuffle(dataset)

        datasetSplit = int((dataSplit)*len(dataset))
        trainData = dataset[:datasetSplit]
        testData = dataset[datasetSplit+1:]

        # Train the classifier
        print("Training the model...")
        classifier = NaiveBayesClassifier.train(trainData)

        # Save the classifier
        if language == "en":
            modelFile = open('models/en_leadClassifier.pickle', 'wb')
            print("English classifier saved!")
        elif language == "es":
            modelFile = open('models/es_leadClassifier.pickle', 'wb')
            print("Spanish classifier saved!")
        elif language == "pt":
            modelFile = open('models/pt_leadClassifier.pickle', 'wb')
            print("Portuguese classifier saved!")
        pickle.dump(classifier, modelFile)
        modelFile.close()

        print("Accuracy is:", classify.accuracy(classifier, testData))
        # print(classifier.show_most_informative_features(10))     

except getopt.GetoptError:
    print ('usage: Train.py -data <data_root> -language <en_or_es_or_pt> -limit <dataLimit>')
    sys.exit(2)