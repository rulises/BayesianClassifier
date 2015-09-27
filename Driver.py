#!/usr/bin/python

import sys
from BayesClassifier import *




def main():
    data_path = str(sys.argv[1])
    output_path = "output/SimpleData"

    split(data_path, output_path)

    model = BayesClassifier()
    model.train(output_path+".train")

    if model.classify("I hate my AI class") > 0.5:
        print "positive"
    else:
        print "negative"



main()
