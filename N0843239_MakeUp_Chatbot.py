# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:49:57 2023

@author: N0843239
"""

#!pip install aiml
#!pip install scikit-learn
#!pip install nltk
#!pip install azure-cognitiveservices-vision-computervision 

import aiml
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import json, requests
import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import requests, uuid, json



# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="MakeUpBot.xml")
lemmatizer = WordNetLemmatizer()

#set up Azure Cognitive Service
key = 'f3deb28895834d398c42e0ba2cb47ed0'
endpoint = 'https://section-d.cognitiveservices.azure.com/'
region = 'uksouth'

#global variable for the lines found in the image.
imageLines= ''

# Get client for computer vision service
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))


#This function has been derived from #https://github.com/MicrosoftDocs/ai-fundamentals/blob/master/02c%20-%20Translation.ipynb

def azure_Translator(region, key, text, target_Lang='fr'):

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&to={}'.format(target_Lang)
    constructed_url = path + params

    # Prepare the request headers
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region':region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    translate_Request = requests.post(constructed_url, headers=headers, json=body)
    translate_Response = translate_Request.json()
    return translate_Response[0]["translations"][0]["text"]


print("\nWelcome to this Makeup chatbot. Please feel free to ask questions from me!\n")

while True:
    #get user input
    try:
        userInput = input("> ")

    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        xmlAnswer = kern.respond(userInput)
        
        if xmlAnswer[0] == '#':
            params = xmlAnswer[1:].split('$')
            cmd = int(params[0])
            
            if cmd == 0:
                print(params[1])
                break
            
            elif cmd == 1:
                #User is able to ask for a branded product type. 
                #The question must begin with 'what is' 
                try:
                    responseSuccess = False
                    URL_makeupAPI = r"https://makeup-api.herokuapp.com/api/v1/products.json?"
                    inputBrandProd = params[1]
                    inputBrandProd = inputBrandProd.split(" ")
                    response = requests.get(URL_makeupAPI + r"brand=" + inputBrandProd[0] + r"&product_type=" + inputBrandProd[1])
                    if response.status_code == 200:
                        response_json = json.loads(response.content)
                        if response_json:
                            name = response_json[0]['name']
                            prod_type = response_json[0]['product_type']
                            description = response_json[0]['description']
                            print('\n\bName:\t' , name,'\n\n\bDescription:\n\t', description, "\n")
                            responseSuccess = True
                    if not responseSuccess:
                        print("Sorry, I could not find an example for the brand and product you gave me")
                except:
                    print("I did not get that, please try again.")
            elif cmd == 99:
                if (userInput.startswith("Show me text from")):
                    print ('testing elif - section d')
                    input_Array=userInput.split(" ")
                    image_Path = input_Array[3]
                    
                    #ADD FUNCTIONALITY TO CHOOSE LANGUAGE - COMAPRE ON LIST?
                    chosen_Lang = input_Array[5]
                    
                    
                    # Read the image file
                    #image_path = 'test-quote.jpg'
                    found_Image = open(image_Path, "rb")

                    # Use Computer Vision to find text in image
                    image_Results = computervision_client.recognize_printed_text_in_stream(found_Image)

                    #reads the lines in the image one by one
                    for a_Region in image_Results.regions:
                        for a_Line in a_Region.lines:
                            # Read the words in the line of text
                            a_Line_text = ''
                            for a_Word in a_Line.words:
                                a_Line_text += a_Word.text + ' '
                                imageLines = a_Line_text
                           # print(line_text.rstrip())
                           
                    translated_Image = azure_Translator(region, key, imageLines, to_lang='it-IT')
                    print('{} -> {}'.format(imageLines,translated_Image))
               
                else:
            
                    #if no other options fit, the user is directed towards the CSV file.
                    try:
                        df = pd.read_csv('QA.csv').dropna()
                        lemmatizer.lemmatize(userInput)
                        inputArray = [userInput]
                        counter = 0
                        #like 91 has derived from: https://goodboychan.github.io/python/datacamp/natural_language_processing/2020/07/17/04-TF-IDF-and-similarity-scores.html
                        # Create TfidfVectorizer object
                        vectorizer = TfidfVectorizer()
                        
                        #line 93 has derived from: https://stackoverflow.com/questions/58240401/matching-phrase-using-tf-idf-and-cosine-similarity
                        similarity_index_list = cosine_similarity(vectorizer.fit_transform(df["Question"]), vectorizer.transform(inputArray)).flatten()
                       
                        #stores the value of the answer at the index position of the best match
                        csvAnswerOuput = df.loc[similarity_index_list.argmax(), "Answer"]
                        
                        #Checks if every index returns a 0 similarity against the CSV questions. 
                        #The potential answer is then disregarded
                        for x in similarity_index_list:
                            if x <= 0.1:
                                counter = counter + 1
    
                        #only printing the answer if there is a suitable similarity level       
                        if counter == len(similarity_index_list):
                                print ("I'm sorry, I don't have an answer for that.")
                        else:
                                print (csvAnswerOuput)
                    except:    
                        print("I did not get that, please try again.")                
        else:
            print(xmlAnswer)
    else:
        print ("Error")
