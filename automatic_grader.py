#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:37:24 2018

@author: gouravmahapatr
"""
from __future__ import division
import os
path = '/Users/gouravmahapatr/Dropbox/EdtechProject(Shrey,Gourav)/AGS (Automated Grading System)/codes/short-answer-grader-master'
os.chdir(path)
from featureExtraction import *
from config import *
from align import *
from scipy import spatial
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

global embeddings
embeddings = {}

def get_lemmas(parsed_sentence):
    lemmatized = lemmatize(parsed_sentence)
    lemmas = [item[3] for item in lemmatized]
    return lemmas

def get_content_lemmas(lemmas):
    content_lemmas = []
    for i in range(len(lemmas)):
        if lemmas[i].lower() not in \
            stop_words+punctuations:
                content_lemmas.append(lemmas[i].lower())
    return content_lemmas

def get_length_ratio(ref_content_lemmas,ans_content_lemmas):
    length_ratio = float(len(ans_content_lemmas))/float(len(ref_content_lemmas))
    return length_ratio

def get_unique_lemmas(lemma_list):
    '''
    Returns a flattened lemma list
    '''
    unique_lemmas = []
    for item in lemma_list:
        for word in item:
            if word not in unique_lemmas:
                unique_lemmas.append(word)
    return unique_lemmas

def get_aligned_words(sentence1,sentence2):
    sentence1parsed = parseText(sentence1)
    sentence2parsed = parseText(sentence2)
    similarity = align(sentence1,sentence2,sentence1parsed,sentence2parsed)[0]

    sim_list = []
    for item in similarity:
        word1 = sentence1parsed['sentences'][0]['words'][item[0]-1][1]['Lemma']
        word2 = sentence2parsed['sentences'][0]['words'][item[1]-1][1]['Lemma']
        if word1 not in punctuations:
            sim_words = list([word1,word2])
            sim_list.append(sim_words)
    return sim_list
    
def vector_sum(vectors):
    '''
    This function inside "featureExtraction.py" performs a simple vector addition 
    of all the vectors for each word corresponding to the word embeddings. 
    '''
    n = len(vectors)
    d = len(vectors[0])

    s = []
    for i in xrange(d):
        s.append(0)
    s = np.array(s)

    for vector in vectors:
        s = s + np.array(vector)

    return list(s)


def cosine_similarity(vector1, vector2):
    '''
    This function computes cosine similarity by first computing cosine distance.
    
    cosine similarity = 1 - cosine distance
    '''
    
    return 1 - spatial.distance.cosine(vector1, vector2)

def load_embeddings(file_name):

    embeddings = {}

    input_file = open(file_name, 'r')
    for line in input_file:
        tokens = line.split('\t')
        tokens[-1] = tokens[-1].strip()
        for i in xrange(1, len(tokens)):
            tokens[i] = float(tokens[i])
        embeddings[tokens[0]] = tokens[1:-1]

    return embeddings

def get_lemma_embeddings(lemma_list):
    lemma_embeddings = []
    for item in lemma_list:
        if item.lower() in stop_words+punctuations:
            continue
        elif item.lower() in embeddings:
            lemma_embeddings.append(embeddings[item.lower()])
    return lemma_embeddings

#if __name__ == '__main__':

# define the embeddings


os.chdir('Sample Train Data')
# first import the data
data = [line.strip() for line in open('SampleAnswerDataPS.txt')]

question = data[1]

refs1 = data[3]
refs2 = data[4]
refs3 = data[5]
refs4 = data[6]
refs5 = data[7]

ans1 = data[9]
ans2 = data[10]
ans3 = data[11]
ans4 = data[12]

# this produces the whole response
refans = refs1+refs2+refs3+refs4+refs5
student_ans = ans1+ans2+ans3+ans4

# make a list containing all the individual sentences
ref_slist = [refs1,refs2,refs3,refs4,refs5]
ans_slist = [ans1,ans2,ans3,ans4]
#ans_slist = ref_slist

ref_slist = ["They are going for food.",]
ans_slist = ["They are going to eat something.",]

# parse the individual sentences and store them as a list
ref_parsed = []
for i in ref_slist: 
    ref_parsed.append(parseText(i))
    
ans_parsed = []
for i in ans_slist:
    ans_parsed.append(parseText(i))
    

# get the lemma words and store them
ref_lemmas = []
for item in ref_parsed:
    ref_lemmas.append(get_lemmas(item))

ans_lemmas = []
for item in ans_parsed:
    ans_lemmas.append(get_lemmas(item))

# filter the stop words and punctuations 
ref_content_lemmas = []
for item in ref_lemmas:
    ref_content_lemmas.append(get_content_lemmas(item))
    
ans_content_lemmas = []
for item in ans_lemmas:
    ans_content_lemmas.append(get_content_lemmas(item))

# get the unique content_lemmas
ref_unique_content = get_unique_lemmas(ref_content_lemmas)
ans_unique_content = get_unique_lemmas(ans_content_lemmas)

# calculate the word-to-word similarity
c=0
for item in ans_unique_content:
    if item in ref_unique_content:
        c+=1
nref_words = float(len(ref_unique_content))
sim_score = (float(c)/nref_words)*1e2
print "Word-to-word similarity score is...",sim_score

# compute the cosine similarity
global embeddings
if embeddings == {}:
    print 'loading embeddings...'
    embeddings = \
       load_embeddings('../Resources/EN-wform.w.5.cbow.neg10.400.subsmpl.txt')
    print 'done'    

ref_embeddings = np.array(get_lemma_embeddings(ref_unique_content))
ans_embeddings = np.array(get_lemma_embeddings(ans_unique_content))
# delete the embeddings after use
if embeddings != {}:
    embeddings = {}

# sum up all the embeddings for each answer set
ref_embeddings_sum = np.sum(ref_embeddings,axis=0)
ans_embeddings_sum = np.sum(ans_embeddings,axis=0)

cos_sim = cosine_similarity(ref_embeddings_sum,ans_embeddings_sum)
print "cosine similarity score is...",cos_sim*1e2," %"

# measures the amount of content words in answer by refernce answer 
length_ratio = get_length_ratio(ref_content_lemmas,ans_content_lemmas)
    
# get the tf-idf of the document set
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([refans,student_ans])
tfidf_sim = (X*X.T).A
print "The tf-idf similarity score is...",tfidf_sim[0,1]*1e2," %"
    

    
    


'''We learn by constantly trying to improve.'''
# store the lemmatized sentences
#ref_lemmatize = []
#for i in ref_parsed:
#    ref_lemmatize.append(lemmatize(i))
#    
#ans_lemmatize = []
#for i in ans_parsed:
#    ans_lemmatize.append(lemmatize(i))
    
    