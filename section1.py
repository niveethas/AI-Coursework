# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:31:51 2023

@author: nivee
"""
#!pip install scikit-learn
import wikipedia
import aiml
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

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
        # --------- MAKE SURE TO CHANGE THE CODE IN HERE TO NOT PLAGARISE ---------'

        df = pd.read_csv('QA.csv').dropna()
       # makeupBot = df['Question']
       # print (type(makeupBot))
        r = [userInput]
       # temp = pd.Series(r,copy=False)
        #makeupBot = makeupBot.append(temp, ignore_index = True)
        #print (makeupBot)
        
        #to find data type use type(variable-name)
        

        #https://goodboychan.github.io/python/datacamp/natural_language_processing/2020/07/17/04-TF-IDF-and-similarity-scores.html
        # Create TfidfVectorizer object
        vectorizer = TfidfVectorizer()

        # Generate matrix of word vectors
        #tfidf_matrix = vectorizer.fit_transform(makeupBot)

        #https://stackoverflow.com/questions/58240401/matching-phrase-using-tf-idf-and-cosine-similarity
        #this works
        similarity_index_list = cosine_similarity(vectorizer.fit_transform(df["Question"]), vectorizer.transform(r)).flatten()
        output = df.loc[similarity_index_list.argmax(), "Answer"]
        print ("output",output)
       
        #consine_similarities = linear_kernel(tfidf_matrix[0:10],tfidf_matrix).flatten()
        
        #indices =  consine_similarities.argsort()[:-5:-1]
       # print (indices)
        #print(consine_similarities[indices])
        #print (makeupBot)

        # --------- MAKE SURE TO CHANGE THE CODE IN HERE TO NOT PLAGARISE ---------
        if answer[0] == '#':
            params = answer[1:].split('$')
            cmd = int(params[0])
            if cmd == 0:
                print(params[1])
                break
            elif cmd == 99:
                print("I did not get that, please try again.")
        else:
            print(answer)