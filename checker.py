import pandas
import numpy

filename = "xiangqi_evaluations.txt"
dataset = pandas.read_csv(filename, dtype={'eval':numpy.int16, 'positions':str})

def how_many(margin):
  inbetween = dataset[((dataset['eval'] > -margin) & (dataset['eval'] < margin))]
  print(len(inbetween), 'out of ', len(dataset))

how_many(100)
how_many(200)
how_many(300)
how_many(400)
