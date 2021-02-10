from keras.models import load_model
import json 
import re
from keras.preprocessing import sequence

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


model = load_model('getEmotions_1LSTM.h5')

with open('top_words_dict.txt') as f:
  top_words_dict = json.load(f)


stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
#function to clean the word of any punctuation or special characters
def cleanpunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

def analyzeSentence(sent):
	s=''
	filtered_sentence=[]
	sent=cleanhtml(sent) # remove HTMl tags
	for w in sent.split():
	    for cleaned_words in cleanpunc(w).split():
	        if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
	            if(cleaned_words.lower() not in stop):
	                s=(sno.stem(cleaned_words.lower()))
	                filtered_sentence.append(s)
	            else:
	                continue
	        else:
	            continue 

	indexedReview = []
	wordIndices = []
	for word in filtered_sentence:
	    wordIndex = top_words_dict.get(word, -1)
	    if (wordIndex > 0):
	        wordIndices.append(wordIndex)
	    # print(word)
	indexedReview.append(wordIndices)


	# Truncate and/or pad input sequences
	max_review_length = 600
	indexedText = sequence.pad_sequences(indexedReview, maxlen=max_review_length)

	predicted_output = model.predict([indexedText], batch_size=512) 
	# print(predicted_output)
	return predicted_output[0][0]



score = 0
question = input('\n\nSmart Bot:> ')
while (question != 'q'):
	score = analyzeSentence(question)
	print(score)
	if (score < 0.1):
		print("Sentiment: Very disappointed. Activating Human Handoff!")
	elif (score < 0.2):
		print("Sentiment: Negative Mood. User has concerns")
	elif (score < 0.6):
		print("Sentiment: Neutral Mood.")
	elif (score < 0.8):
		print("Sentiment: Positive Mood.")
	else:
		print("Sentiment: Exhuberant Mood.")
	question = input('\nSmart Bot:> ')



# sent = 'that was great. you have done a great job'
# sent = 'the service is ok. how are you?'
# sent = that information is wrong. so disappointing
# sent = 'i am terribly angry. how can you give such a bad service?'


# 5-class classification
# sent = 'the product is great. I want to buy'
# sent = 'product is average. worth a buy'
# sent = 'the service is ok. what do you think?''
# sent = that information is wrong. so disappointing
# sent = 'i am terribly angry. how can you give such a bad service?'
