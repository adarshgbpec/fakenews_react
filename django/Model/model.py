import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df_fake=pd.read_csv('Fake.csv')
df_true=pd.read_csv('True.csv')

df_fake['class']=0
df_true['class']=1

df_merge=pd.concat([df_fake,df_true],axis=0)

df=df_merge.drop(['title','subject','date'],axis=1)

df=df.sample(frac=1)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


df["text"] = df["text"].apply(wordopt)

x = df["text"]
y = df["class"]
print(df.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression()
LR.fit(xv_train,y_train)

print(LR.score(xv_test, y_test))

filename='finalized_model.sav'
joblib.dump(LR,filename)
