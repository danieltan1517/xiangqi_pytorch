import pandas
import numpy

filename = 'xiangqi_evaluations.txt'
dataframe = pandas.read_csv(filename, dtype={'eval':numpy.int16, 'positions':str}, nrows=1000)
print(dataframe)

dataframe = dataframe.loc[(dataframe['eval'] >= -150) & (dataframe['eval'] <= 150)]
print(dataframe)
