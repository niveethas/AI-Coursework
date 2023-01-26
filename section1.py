# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:31:51 2023

@author: nivee
"""
# !pip install aiml
# !pip install wikipedia
# !pip install scikit-learn
import wikipedia
import aiml
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import json, requests


# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="MakeUpBot.xml")


print("Welcome to this Makeup chatbot. Please feel free to ask questions from me!")
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
        if answer[0] == '#':
            params = answer[1:].split('$')
            cmd = int(params[0])
            if cmd == 0:
                print(params[1])
                break
            #else:
            elif cmd == 1:
                #what is maybelline, may be cmd 3 not 2 -> 2 is how is weather i think
                    succeeded = False
                    api_url = r"https://makeup-api.herokuapp.com/api/v1/products.json?"
                    inputBrandProd = params[1]
                    inputBrandProd = inputBrandProd.split(" ")
                    response = requests.get(api_url + r"brand=" + inputBrandProd[0] + r"&product_type=" + inputBrandProd[1])
                    if response.status_code == 200:
                        response_json = json.loads(response.content)
                        if response_json:
                            name = response_json[0]['name']
                            prod_type = response_json[0]['product_type']
                            description = response_json[0]['description']
                            print('\n\bName:\t' , name,'\n\n\bDescription:\n\t', description, "\n")
                            succeeded = True
                    if not succeeded:
                        print("Sorry, I could not find an example for the brand and product you gave me")
            elif cmd == 99:
                
                try:
                    # --------- MAKE SURE TO CHANGE THE CODE IN HERE TO NOT PLAGARISE ---------'
                    df = pd.read_csv('QA.csv').dropna()
                    #inputArray = userInput.split()
                    inputArray = [userInput]
                    counter = 0
                    #https://goodboychan.github.io/python/datacamp/natural_language_processing/2020/07/17/04-TF-IDF-and-similarity-scores.html
                    # Create TfidfVectorizer object
                    vectorizer = TfidfVectorizer()

                   
                    #https://stackoverflow.com/questions/58240401/matching-phrase-using-tf-idf-and-cosine-similarity
                    #for x in range(len(inputArray)):
                     #   temp = [inputArray[x]]
                      #  print (temp, " = temp")
                    
                    #LEMATISE THE QUESTIONS ARRAY FIRST    
                
                    similarity_index_list = cosine_similarity(vectorizer.fit_transform(df["Question"]), vectorizer.transform(inputArray)).flatten()
                    output = df.loc[similarity_index_list.argmax(), "Answer"]
                    
                    for x in similarity_index_list:
                        if x <= 0.1:
                            counter = counter + 1

                    #only printing the answer if there is a suitable similarity level       
                    if counter == len(similarity_index_list):
                            print ("I'm sorry, I don't have an answer for that.")
                    else:
                            print (output)
                    
                    # --------- MAKE SURE TO CHANGE THE CODE IN HERE TO NOT PLAGARISE ---------
                except:    
                    print("I did not get that, please try again.")
        else:
            print(answer)
    else:
        print ("aiml not needed")