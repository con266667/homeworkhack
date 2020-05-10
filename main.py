from sklearn.externals import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier

import sys
import os, os.path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from googletrans import Translator
import pyperclip



f = open("chapter.txt", "r")
chapter = f.read()
res = chapter.split()

languages_data_folder = sys.argv[1]
dataset = load_files(languages_data_folder, encoding='latin-1')

docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.01)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',
                             use_idf=False)

clf = Pipeline([
    ('vec', vectorizer),
    ('clf', Perceptron()),
])

clf.fit(docs_train, y_train)

sentences = res
predicted = clf.predict(sentences)

path, dirs, files = next(os.walk(languages_data_folder + "/known"))
known_count = len(files)
path, dirs, files = next(os.walk(languages_data_folder + "/unknown"))
unknown_count = len(files)

translator = Translator()

for s, p in zip(sentences, predicted):
    if dataset.target_names[p] == "unknown" and s != "" and s != None:
        print(s)
        known = input("Do you know this word(Y or N):")  
        if known == "Y":
            file1 = open("data/known/" + str(known_count) + ".txt" ,"w") 
            file1.writelines(s) 
            file1.close()
            known_count += 1
        else :
            translation = translator.translate(s)
            print(s + " - " + translation.text)
            pyperclip.copy(s + " - " + translation.text + "\n")
            file1 = open("data/unknown/" + str(unknown_count) + ".txt" ,"w") 
            file1.writelines(s) 
            file1.close()
            unknown_count += 1
            
