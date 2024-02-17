import pandas
import numpy

filename = "values.txt"
dataset = pandas.read_csv(filename, dtype={'eval':numpy.int16, 'positions':str})

def how_many(a, b):
  inbetween = dataset[((dataset['eval'] > a) & (dataset['eval'] < b))]
  print(len(inbetween), 'out of ', len(dataset))

how_many(-100,0)
how_many(-200,0)
how_many(-300,0)
how_many(-400,0)
