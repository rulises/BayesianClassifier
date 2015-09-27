from __future__ import division
import math, os, pickle
from DataReader import *
from collections import defaultdict

class BayesClassifier:

   def __init__(self):
      '''This method initializes the Naive Bayes classifier'''
      self.training_set = defaultdict(lambda: [0,0])
      self.total_positive = 0
      self.total_negative = 0 
      self.k_smooth = 0.5
      
   def train(self, dataFile):   
      '''Trains the Naive Bayes Sentiment Classifier.'''
      count = defaultdict(lambda: [0,0])
      reader = DataReader(dataFile)
      for label, tokens in reader:
          for token in tokens:
              if label == "negative":
                  self.total_negative += 1
                  count[token][0] += 1  
              else:
                  self.total_positive += 1
                  count[token][1] += 1  

      ''' Compute P(X|A) and P(X|~A)  '''
      for token in count.keys():
          p_negative = (self.k_smooth + count[token][0])/(self.total_negative + 2 * self.k_smooth)
          p_positive = (self.k_smooth + count[token][1])/(self.total_positive + 2 * self.k_smooth)
          self.training_set[token] = [p_negative, p_positive]

   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive or negative ).
      '''
      tokens = tokenize(sText)
      p_is_pos = 0.0
      p_is_neg = 0.0
      for token in self.training_set.keys():
          if token in tokens:
              p_is_pos += math.log(self.training_set[token][1])    
              p_is_neg += math.log(self.training_set[token][0]) 
          else:
              p_is_pos += math.log(1.0 - self.training_set[token][1]) 
              p_is_neg += math.log(1.0 - self.training_set[token][0])
      p_is_pos = math.exp(p_is_pos)
      p_is_neg = math.exp(p_is_neg)
      print p_is_pos / ( p_is_pos + p_is_neg )
      return p_is_pos / ( p_is_pos + p_is_neg )
 

   def save(self, sFilename):
      '''Save the learned data during training to a file using pickle.'''

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      # use dump to dump your variables
      #p.dump(self.var1)
      #p.dump(self.var2)
      #p.dump(self.var3)
      f.close()
   
   def load(self, sFilename):
      '''Given a file name of stored data, load and return the object stored in the file.'''

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      # use load to load in previously dumped variables
      #self.var1 = u.load()
      #self.var2 = u.load()
      #self.var3 = u.load()
      
      f.close()

