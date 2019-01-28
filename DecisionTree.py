import pandas as pd
import numpy as np
import time

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

startTime = time.time()

df = pd.read_csv('reviews.csv', sep='|')
df.head()

df.dropna(inplace=True)

df['label'] = np.where(df['label'] == 'positive', 1, 0)


train = df[df.index % 5 != 0]
test = df[df.index % 5 == 0][1:]

print("sliced size: "+str(len(train)))

X_train = train["text"]
y_train = train["label"]

X_test = test["text"]
y_test = test["label"]


# Countervectorizer/tf idf
cntVect = CountVectorizer()

classifier = DecisionTreeClassifier()

# pipeline = Pipeline()
pipel = Pipeline([('vectorizA',cntVect), ('clissifiA', classifier)])

print("Starting to train after: "+str(time.time()-startTime))
pipel.fit(X_train,y_train)

print("Finished training after: "+str(time.time()-startTime))

# picklemain, cPickle.dump, picklemain
file = open('DecisionTree.m', 'wb')
joblib.dump(pipel,file)
file.close()

endTime = time.time()

print(str(endTime-startTime))
