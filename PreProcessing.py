
import yaml #this import is used for with open function and read from a file type yml(aka yaml)
import glob #this import is used to basically get all the files within the specified directory
#import nltk #this import is used for tokenization of the string and lowercase function of the tokenized txt
import json
from nltk.stem import PorterStemmer


print("hello")
#print(glob.glob("/Users/aidenchang/Desktop/ChatBotProject/chatterbot-corpus-master/chatterbot_corpus/data/english/*.yml"))

outputpath = "/Users/aidenchang/Desktop/ChatBotProject/chatterbot-corpus-master/"

intents = dict() #this will be the final dictionary with tags, patterns, and responses. 
intents["intents"] = []
for path in (glob.glob("/Users/aidenchang/Desktop/ChatBotProject/chatterbot-corpus-master/chatterbot_corpus/data/english/*.yml")):
    with open(path) as f:
        tagpatres = dict()
        tagpatres["tags"] = []
        tagpatres["patterns"] = []
        tagpatres["responses"] = []

        Test_List = yaml.load(f, Loader=yaml.FullLoader)
        #This Code will read the entire files in the folder
        for categories in Test_List["categories"]:
            tagpatres["tags"].append(categories)
            #print(categories)
        
        for list in Test_List["conversations"]:
            for i in range(len(list)):
                if i % 2 == 0:
                    tagpatres["patterns"].append(list[i])
                else: 
                    tagpatres["responses"].append(list[i])

        
        intents["intents"].append(tagpatres) #preprocessing

        #This Code will Read the values in the first "conversations", but we have to assign them to a tag
#print(intents)
intents = json.dumps(intents)

f = open(outputpath+ "file.json", "w")  #x will create a file, while w will write on the created file, the +"" part will create a file 
f.write(intents)
f.close()

with open("/Users/aidenchang/Desktop/ChatBotProject/chatterbot-corpus-master/file.json") as f:
    d = json.load(f)
#print(d["intents"])
# tokened = nltk.word_tokenize(intents)
        
#example of dict?
# Greetings = dict()
# Greetings["tag"] = "greeting"
# while True: