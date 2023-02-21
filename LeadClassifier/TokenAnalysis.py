import os 
import pandas as pd
import nltk
import re, string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag

def LemmatizeSentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def RemoveNoise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        token = re.sub('\\\\n'," ", token)
        token = re.sub(" URL","", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def GetAllWords(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def GetTweetsForModel(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def DefineStopWords(language):
    if language.lower() == "en":
        return stopwords.words('english')
    elif language.lower() == "es":
        return stopwords.words('spanish')
    elif language.lower() == "pt":
        return stopwords.words('portuguese')
    else:
        print('The language must be defined between en, es or pt')
        exit()

def GetData(dataRoot, language):
    positiveDataPath = os.path.abspath(dataRoot + language + '_p.xlsx')
    negativeDataPath = os.path.abspath(dataRoot + language + '_n.xlsx')
    positiveTweets = pd.read_excel(positiveDataPath)["descricao"].values.tolist()
    negativeTweets = pd.read_excel(negativeDataPath)["descricao"].values.tolist()

    return positiveTweets, negativeTweets

def GetCleanedTokenList(positiveTweets, negativeTweets, stopwords):
    positiveTweetTokens = []
    negativeTweetTokens = []
    positiveCleanedTokensList = []
    negativeCleanedTokensList = []

    for i in positiveTweets:
        positiveTweetTokens.append(i.split())

    for i in negativeTweets:
        negativeTweetTokens.append(i.split())

    for tokens in positiveTweetTokens:
        positiveCleanedTokensList.append(RemoveNoise(tokens, stopwords))

    for tokens in negativeTweetTokens:
        negativeCleanedTokensList.append(RemoveNoise(tokens, stopwords))

    return positiveCleanedTokensList, negativeCleanedTokensList

def GetTokensForModel(positiveTweets, negativeTweets, stopwords):
    positiveCleanedTokensList, negativeCleanedTokensList = GetCleanedTokenList(positiveTweets, negativeTweets, stopwords)
    positiveTokensForModel = GetTweetsForModel(positiveCleanedTokensList)
    negativeTokensForModel = GetTweetsForModel(negativeCleanedTokensList)

    return positiveTokensForModel, negativeTokensForModel
