import os
import pandas as pd
import langid as lg
import fasttext
from textblob import TextBlob

def PreProcessData(dataRoot, language, limit):
    print("Preprocessing started!")

    dataPath = pd.ExcelFile(dataRoot+'DataTweetsSystemPicked.xlsx')    
    data = pd.read_excel(dataPath)
    trunkData = data[0:int(limit)]

    # Leads analysis
    leads = trunkData[trunkData['lead'] == 'lead']
    notLeads = trunkData[trunkData['lead'] != 'lead'] 

    # Language analysis
    if language == "en":
        en_p = leads[leads['idioma'] == 'en']
        en_n = notLeads[notLeads['idioma'] == 'en']
        print("English - leads: "+str(len(en_p))+" not leads: "+str(len(en_n)))
        print("Saving english processed data...")
        en_n.to_excel(dataRoot + "en_n.xlsx")
        en_p.to_excel(dataRoot + "en_p.xlsx")
        print("English data saved!")

    elif language == "es":
        es_p = leads[leads['idioma'] == 'es']
        es_n = notLeads[notLeads['idioma'] == 'es']
        print("Spanish - leads: "+str(len(es_p))+" not leads: "+str(len(es_n)))
        print("Saving spanish processed data...")
        es_n.to_excel(dataRoot + "es_n.xlsx")
        es_p.to_excel(dataRoot + "es_p.xlsx")
        print("Spanish data saved!")

    elif language == "pt":
        pt_p = leads[leads['idioma'] == 'pt']
        pt_n = notLeads[notLeads['idioma'] == 'pt']
        print("Portuguese - leads: "+str(len(pt_p))+" not leads: "+str(len(pt_n)))
        print("Saving portuguese processed data...")
        pt_n.to_excel(dataRoot + "pt_n.xlsx")
        pt_p.to_excel(dataRoot + "pt_p.xlsx")
        print("Portuguese data saved!")





    
    
    

    
    