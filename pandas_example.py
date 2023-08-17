import pandas
import numpy

dataframe = pandas.read_csv('text.txt', dtype={'eval':numpy.int16, 'positions':str})

print(dataframe['eval'])
for p in dataframe['positions']:
  print(p)
