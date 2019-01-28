import sys
from sklearn.externals import joblib

inputFile = open(sys.argv[2], 'r')
outputFile = open('output.txt', 'w+')

model = joblib.load(open('NeuralNet.m', 'r'))
with open(path, 'rb') as f:
  contents = f.read()

prediction = model.predict(contents)

if len(prediction) > 0:
  outputFile.write(prediction[0])

inputFile.close()
outputFile.close()
