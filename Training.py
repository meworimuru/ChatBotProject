import json
from nltk.stem import PorterStemmer
# import torch
# from torch.utils.data import Dataset
# torch.manual_seed(42)
import tensorflow as tf
from tensorflow.keras import models, layers
# lowercase_tokens = [token.lower for token in tokened]
# print(lowercase_tokens)

with open("/Users/aidenchang/Desktop/ChatBotProject/chatterbot-corpus-master/file.json", "r") as f:
    loaded_data = json.load(f)


def tokenize(sentence):
    tokenized = sentence.split()
    for i in range(len(tokenized)):
        if "?" in tokenized[i] or "." in tokenized[i] or "'" in tokenized[i] or "," in tokenized[i] or "!" in tokenized[i]:
            without_punct = tokenized[i][:-1]
            tokenized[i] = without_punct
    return tokenized

words = []
documents = []
tags_list = []


for intent_name in loaded_data:
    intent_data = loaded_data[intent_name]
    for d in intent_data:
        tag = d['tags']
        patterns = d['patterns']
        responses = d['responses'] #these are dictionaries 
        for pattern in patterns: #this is the actual patterns
            stemmer = PorterStemmer() 
            tokenized = tokenize(pattern)
            stemmed_words = [stemmer.stem(w) for w in tokenized] #this stems all the tokenized words
            for w in stemmed_words: 
                words.append(w) #this appends all stemmed words into [word]
            #tuple_result = tuple((tokenized,tag)) <--- not worked version this is because each tokenized and tag is a list, each of those need to be turned into tuple, as the () just becomes one tuple if this is done. 
            tuple_result2 = (tuple(tokenized),tuple(tag))
            documents.append(tuple_result2)
        for num in tag:
            tags_list.append(num) #this appends all class names to [class]
            #print(classes)
set_words = set(words)       
list_words = list(set_words) 
set_documents = set(documents) 
print(tags_list)
# set_words = [i, like, subject, secret, how, you]
# (sentence, tag) = (i like you, emotion)
# back of words -> return [1, 1, 0, 0, 0, 1]
# make a temp list that has length of set_words
# check if set_words[index] is in sentence
# temp_list[index]=1

def bagofwords(pattern):
    bow = [0] * len(set_words)
    for i, word in enumerate(list_words):
        if word in patterns:
            bow[i] = 1
    return bow
def tagbow(tag):
    tbow = [0] * len(tags_list)
    location = tags_list.index(tag)
    tbow[location] = 1
    return tbow
#print(bow)
X = []
Y = []
for patterns, tag in set_documents:
    entire_bow = bagofwords(patterns)
    entire_tbow = tagbow(list(tag)[0])
    X.append(entire_bow)
    Y.append(entire_tbow)


print(X[0])
print(Y[0])
print(len(X))
# print(len(Y))
# for i in range (0,581):
#     print(len(X[0]))


input_size = len(X[0])
# Create a `Sequential` model and add a Dense layer as the first layer.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape = (input_size,), activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(20, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

#Adam is supposedly better than SGD(Stochastic Gradient Descent) but idk what these are in the first place 
#binary_crossentropy is not a good loss function as that handles only 2 classfications and we need a multiclass classification function 
# Multi-Class Cross-Entropy Loss: Used for when there are more than two categories 
# Sparse Multiclass Cross-Entropy Loss: similar to cross-entropy but uses integers as class labels
# Kullback Leibler Divergence Loss: More used for probability distributions? Which isn't the case here
# From these 3 Loss Functions(All used for multiclass) it seems multi-class cross entropy is best.

#The "Metric" segment is used to specify which evalution metrics you want to use. In this case, it calculates the classification accuracy 
model.fit(X,Y, epochs = 100, verbose=1)
#hyperparameter: parameters that ppl set 
X_test = input()
prediction = model.predict()



for intent_name in loaded_data:
    intent_data = loaded_data[intent_name]
    for d in intent_data:
        tag = d['new_tags']
        patterns = d['new_patterns']
        responses = d['new_responses']
        for pattern in patterns: 
            stemmer = PorterStemmer() 
            tokenized = tokenize(pattern)
            stemmed_words = [stemmer.stem(w) for w in tokenized] #this stems all the tokenized words
            for w in stemmed_words: 
                words.append(w) #this appends all stemmed words into [word]
            #tuple_result = tuple((tokenized,tag)) <--- not worked version this is because each tokenized and tag is a list, each of those need to be turned into tuple, as the () just becomes one tuple if this is done. 
            tuple_result2 = (tuple(tokenized),tuple(tag))
            documents.append(tuple_result2)
        for num in tag:
            tags_list.append(num) #this appends all class names to [class]
            #print(classes)
set_words = set(words)       
list_words = list(set_words) 
set_documents = set(documents) 
print(tags_list)

user_input = input("Begin with your sentence: ")
print("You said: "+ user_input)
tokenized_ui = tokenize(user_input)
print(tokenized_ui)


# class SimpleDataset(Dataset):
#     # defining values in the constructor
#     def __init__(self, data_length = 20, transform = None):
#         self.x = 3 * torch.eye(data_length, 2)
#         self.y = torch.eye(data_length, 4)
#         self.transform = transform
#         self.len = data_length
     
#     # Getting the data samples
#     def __getitem__(self, idx):
#         sample = self.x[idx], self.y[idx]
#         if self.transform:
#             sample = self.transform(sample)     
#         return sample
    
#     # Getting data size/length
#     def __len__(self):
#         return self.len
        
        
    # print(set_documents)
    # print(entire_bow)
    # print(entire_tbow)
    #print(entire_tbow)

    # def create_bow(set_documents):
#     word_count = {}
#     bag_of_words = []
#     print(word_count)
#     for tuple in set_documents:
#         for pattern in tuple:
#             for word in pattern:
#                 if word not in word_count:
#                     word_count[word] = 0
#                 else: word_count[word] += 1
#     print(word_count)
#     for patterns, _ in set_documents:
#         document_vector = []
#         for word in patterns:
#             # Count the occurrences of each word in the document
#             count = patterns.count(word)
#             document_vector.append(count)
#         bag_of_words.append(document_vector)
#     return bag_of_words

# # Example usage
# bag_of_words = create_bow(set_documents)

# print(set_documents)
# print(set_words)

# pattern_count = pattern.count(bag_of_words)
# print(pattern_count)