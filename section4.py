# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:36:50 2023

@author: nivee
"""
#!pip install azure-cognitiveservices-vision-computervision 


from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
#%matplotlib inline


#THIS CODE IS DIRECT COPY OF
#https://github.com/MicrosoftDocs/ai-fundamentals/blob/master/01e%20-%20Optical%20Character%20Recognition.ipynb


cog_key = 'f3deb28895834d398c42e0ba2cb47ed0'
cog_endpoint = 'https://section-d.cognitiveservices.azure.com/'
cog_region = 'uksouth'

#global variable for the lines found in the image.
imageLines= ''


# Get a client for the computer vision service
computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

# Read the image file
image_path = 'test-quote.jpg'
#os.path.join('data', 'ocr', 'advert.jpg')

image_stream = open(image_path, "rb")

# Use the Computer Vision service to find text in the image
read_results = computervision_client.recognize_printed_text_in_stream(image_stream)

# Process the text line by line
for region in read_results.regions:
    for line in region.lines:

        # Read the words in the line of text
        line_text = ''
        for word in line.words:
            line_text += word.text + ' '
            imageLines = line_text
        print(line_text.rstrip())

# Open image to display it.
fig = plt.figure(figsize=(7, 7))
img = Image.open(image_path)
draw = ImageDraw.Draw(img)
plt.axis('off')
plt.imshow(img)


#THIS IS A DIRECT COPY OF
#https://github.com/MicrosoftDocs/ai-fundamentals/blob/master/02c%20-%20Translation.ipynb

# Create a function that makes a REST request to the Text Translation service
def translate_text(cog_region, cog_key, text, to_lang='fr', from_lang='en'):
    import requests, uuid, json

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region':cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]


# Test the function
text_to_translate = imageLines

translation = translate_text(cog_region, cog_key, text_to_translate, to_lang='it-IT', from_lang='en')
print('{} -> {}'.format(text_to_translate,translation))

