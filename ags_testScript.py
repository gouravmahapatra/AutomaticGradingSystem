"""
Created on Sun Oct 21 21:59:01 2018

This is the automatic grader functions test script...

To make this script work, 
- first run the corenlp.py code.
- run the pretrained grader code. 

@author: gouravmahapatr
"""
from __future__ import division
import os
import sys
path = '/Users/gouravmahapatr/Dropbox/EdtechProject(Shrey,Gourav)/AGS (Automated Grading System)/codes/short-answer-grader-master'
os.chdir(path)
sys.path.append(path)

from featureExtraction import *
from config import *
from align import *
from scipy import spatial
import numpy as np

#question = "How are infix expressions evaluated by computers?"
#
#ref_answer = "First they are converted into postfix form." #+ \
##             "followed by an evaluation of the postfix expression."
##student_response = "computers usually convert infix expressions to postfix " +\
##                   "expression and evaluate them using a stack."
#  
#student_response = " The postfix form is evaluated."                 

#sentence1 = ref_answer
#sentence2 = student_response

sentence1 = "There is food at the dining hall."
sentence2 = "The dining hall serves food at 8 PM."

sentence1parsed = parseText(sentence1)
sentence2parsed = parseText(sentence2)
          
''' parse the sentence using NLP code. '''
#parseResult = parseText(sentence)

# accessing parseResult dictionary

''' get all the sentences '''
#sentences = parseResult['sentences']

''' get the whole sentence in text form'''
#text = parseResult['sentences'][0]['text']

''' extract information about all the words '''
#words = parseResult['sentences'][0]['words']
''' now read in each word and its information '''
#word1 = words[0]
#word2 = words[1]


''' this function "align" evaluates two input sentences and then provides the 
 similar words in the two sentences. 
 e.g. similarities = align(sentence1,sentence2,sentence1parsed,sentence2parsed) '''
similarities = align(sentence1,sentence2,sentence1parsed,sentence2parsed)[0]

''' 
Find the similarity score, coverage and parse results using sts_alignment.
'''
sim,cov,parse = sts_alignment(sentence1,sentence2)



 