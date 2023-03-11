# -*- coding: utf-8 -*-

#!pip install aiml
#!pip install scikit-learn
#!pip install nltk
#!pip install azure-cognitiveservices-vision-computervision 
#!pip install azure-ai-vision
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import azure.ai.vision as aiv
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="MakeUpBot.xml")
lemmatizer = WordNetLemmatizer()

#set up Azure Cognitive Service
key = 'f3deb28895834d398c42e0ba2cb47ed0'
endpoint = 'https://section-d.cognitiveservices.azure.com/'
region = 'uksouth'

#

# Get client for computer vision service
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))


def azure_Translator(region, key, text, target_Lang='fr'):
#This function has been derived from #https://github.com/MicrosoftDocs/ai-fundamentals/blob/master/02c%20-%20Translation.ipynb
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

def image_Checker(imagepath):
    #function has been derived from https://www.analyticsvidhya.com/blog/2021/06/beginner-friendly-project-cat-and-dog-classification-using-cnn/#:~:text=Cat%20and%20dog%20classification%20using%20CNN,-Convolutional%20Neural%20Network&text=Neural%20networks%20can%20be%20trained,tenths%20to%20hundreds%20of%20images.
    predict_Image = load_img(imagepath, target_size= (80,80))
    predict_Image_Modified = img_to_array(predict_Image)
    predict_Image_Modified = predict_Image_Modified / 255
    predict_Image_Modified = np.expand_dims(predict_Image_Modified, axis=0)
    
    image_Classifier = keras.models.load_model("CNN_Image_Classification_Model.h5")
    
    result = image_Classifier.predict(predict_Image_Modified)
    
    if result[0][0] >= 0.5:
        result_Prediction = 'Make up is worn'
    else:
        result_Prediction = 'Make up is not worn'
    
    print("\nPrediction -> " + result_Prediction+"\n")

language_Code_Dict = {
"Afrikaans":"af",
"Albanian":"sq",
"Amharic":"am",
"Arabic":"ar",
"Armenian":"hy",
"Assamese":"as",
"Azerbaijani (Latin)":"az",
"Bangla":"bn",
"Bashkir":"ba",
"Basque":"eu",
"Bosnian (Latin)":"bs",
"Bulgarian":"bg",
"Cantonese (Traditional)":"yue",
"Catalan":"ca",
"Chinese (Literary)":"lzh",
"Chinese Simplified":"zh-Hans",
"Chinese Traditional":"zh-Hant",
"Croatian":"hr",
"Czech":"cs",
"Danish":"da",
"Dari":"prs",
"Divehi":"dv",
"Dutch":"nl",
"English":"en",
"Estonian":"et",
"Faroese":"fo",
"Fijian":"fj",
"Filipino":"fil",
"Finnish":"fi",
"French":"fr",
"French (Canada)":"fr-ca",
"Galician":"gl",
"Georgian":"ka",
"German":"de",
"Greek":"el",
"Gujarati":"gu",
"Haitian Creole":"ht",
"Hebrew":"he",
"Hindi":"hi",
"Hmong Daw (Latin)":"mww",
"Hungarian":"hu",
"Icelandic":"is",
"Indonesian":"id",
"Inuinnaqtun":"ikt",
"Inuktitut":"iu",
"Inuktitut (Latin)":"iu-Latn",
"Irish":"ga",
"Italian":"it",
"Japanese":"ja",
"Kannada":"kn",
"Kazakh":"kk",
"Khmer":"km",
"Klingon":"tlh-Latn",
"Klingon (plqaD)":"tlh-Piqd",
"Korean":"ko",
"Kurdish (Central)":"ku",
"Kurdish (Northern)":"kmr",
"Kyrgyz (Cyrillic)":"ky",
"Lao":"lo",
"Latvian":"lv",
"Lithuanian":"lt",
"Macedonian":"mk",
"Malagasy":"mg",
"Malay (Latin)":"ms",
"Malayalam":"ml",
"Maltese":"mt",
"Maori":"mi",
"Marathi":"mr",
"Mongolian (Cyrillic)":"mn-Cyrl",
"Mongolian (Traditional)":"mn-Mong",
"Myanmar":"my",
"Nepali":"ne",
"Norwegian":"nb",
"Odia":"or",
"Pashto":"ps",
"Persian":"fa",
"Polish":"pl",
"Portuguese (Brazil)":"pt",
"Portuguese (Portugal)":"pt-pt",
"Punjabi":"pa",
"Queretaro Otomi":"otq",
"Romanian":"ro",
"Russian":"ru",
"Samoan (Latin)":"sm",
"Serbian (Cyrillic)":"sr-Cyrl",
"Serbian (Latin)":"sr-Latn",
"Slovak":"sk",
"Slovenian":"sl",
"Somali (Arabic)":"so",
"Spanish":"es",
"Swahili (Latin)":"sw",
"Swedish":"sv",
"Tahitian":"ty",
"Tamil":"ta",
"Tatar (Latin)":"tt",
"Telugu":"te",
"Thai":"th",
"Tibetan":"bo",
"Tigrinya":"ti",
"Tongan":"to",
"Turkish":"tr",
"Turkmen (Latin)":"tk",
"Ukrainian":"uk",
"Upper Sorbian":"hsb",
"Urdu":"ur",
"Uyghur (Arabic)":"ug",
"Uzbek (Latin)":"uz",
"Vietnamese":"vi",
"Welsh":"cy",
"Yucatec Maya":"yua"
}



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
                        try:
                            input_Array=userInput.split(" ")
                            image_Path = input_Array[4]
                            
                            #compares the language input by the user with dictionary
                            chosen_Lang = input_Array[6]
                            chosen_Lang_code = language_Code_Dict[chosen_Lang]
                            
                            # Read the image file
                            found_Image = open(image_Path, "rb")
        
                            # Use Computer Vision to find text in image
                            image_Results = computervision_client.recognize_printed_text_in_stream(found_Image)
        
                            # Read the words in the line of text
                            a_Line_text = ''
                            #reads the lines in the image one by one
                            for a_Region in image_Results.regions:
                                for a_Line in a_Region.lines:
                                    for a_Word in a_Line.words:
                                        a_Line_text += a_Word.text + ' '
                                   
                            translated_Image = azure_Translator(region, key, a_Line_text, target_Lang=chosen_Lang_code)
                            print('{} -> {}'.format(a_Line_text,translated_Image))
                            #print("\n Should I further analyse this image? Y/N")
                            #userInput = input("> ")
                                                          
                        except:
                            print("Sorry, I could not find the image and complete translation")
                if (userInput.startswith("Does she wear makeup in")):
                    try:
                        input_Array = userInput.split(" ")
                        input_Array[5].replace('?', '')
                        image_Path = input_Array[5];
                        image_Checker(image_Path)
                        #code has been derived from https://youtu.be/2gW-JzY4JgU
                        with open (image_Path, 'rb') as chosen_Image:
                            result_Caption = computervision_client.describe_image_in_stream(chosen_Image)
                            print("Content found using Cloud -> " + result_Caption.captions[0].text)
                    except:
                        print("Image could not be found")
                else:
            
                    #if no other options fit, the user is directed towards the CSV file.
                    try:
                        df = pd.read_csv('QA.csv').dropna()
                        lemmatizer.lemmatize(userInput)
                        inputArray = [userInput]
                        counter = 0
                        #derived from: https://goodboychan.github.io/python/datacamp/natural_language_processing/2020/07/17/04-TF-IDF-and-similarity-scores.html
                        # Create TfidfVectorizer object
                        vectorizer = TfidfVectorizer()
                        
                        #derived from: https://stackoverflow.com/questions/58240401/matching-phrase-using-tf-idf-and-cosine-similarity
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
