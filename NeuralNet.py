import pandas as pd
import numpy as np
import time

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
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

# Classifier = MLP/ logistic/ ...
classifier = MLPClassifier(activation='relu', solver='adam', early_stopping=True, verbose=True, hidden_layer_sizes=(8,5))

# pipeline = Pipeline()
pipel = Pipeline([('lalalafdsa',cntVect), ('mdfamfdaklsclassifier', classifier)])

print("Starting to train after: "+str(time.time()-startTime))
# pipeline.fit(X,y)
pipel.fit(X_train,y_train)

print("Finished training after: "+str(time.time()-startTime))

# picklemain, cPickle.dump, picklemain
file = open('NeuralNet.m', 'wb')
joblib.dump(pipel,file)
file.close()

endTime = time.time()

print(str(endTime-startTime))
