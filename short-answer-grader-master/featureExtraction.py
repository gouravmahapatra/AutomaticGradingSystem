from __future__ import division

from config import *
from align import *
from scipy import spatial
import numpy as np

embeddings = {}

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


def sts_alignment(sentence1, sentence2,
                  parse_results=None,
                  sentence_for_demoting=None):
    '''
    This function performs semantic texual similarity alignment.
    
    How it works:
        1. It first uses the Stanford NLP python binding to perform 
        parsing of each of the sentences.
        2. Then it uses the "lemmatize" function to get the lemmatized 
        words for each word in a sentence. 
        3. Checks and does sentence demoting if a demoted sentence is 
        provided.
        4. Uses the "align" function to find the semantically/texually similar words.
        The alignment produces output of combination of words from two sentences 
        that have a similar meaning.
        5. Finds only the content words which are lemmatized. Ignores stop words
        , punctuations and lemmas to be demoted.
        6. Calculates the similarity score (sim_score) semantic similarity score.
        7. Finds the coverage. Coverage is the measure of the amount of content words 
        found in students response that are similar to the refrence answer.
        
    '''
                      
    if parse_results == None:
        sentence1_parse_result = parseText(sentence1)
        sentence2_parse_result = parseText(sentence2)
        parse_results = []
        parse_results.append(sentence1_parse_result)
        parse_results.append(sentence2_parse_result)
    else:
        sentence1_parse_result = parse_results[0]
        sentence2_parse_result = parse_results[1]
        

    sentence1_lemmatized = lemmatize(sentence1_parse_result)
    sentence2_lemmatized = lemmatize(sentence2_parse_result)

    lemmas_to_be_demoted = []
    if sentence_for_demoting != None:
        if len(parse_results) == 2:
            sentence_for_demoting_parse_result = \
                                parseText(sentence_for_demoting)
            parse_results.append(sentence_for_demoting_parse_result)
        else:
            sentence_for_demoting_parse_result = parse_results[2]


        sentence_for_demoting_lemmatized = \
                            lemmatize(sentence_for_demoting_parse_result)
    
        sentence_for_demoting_lemmas = \
                        [item[3] for item in sentence_for_demoting_lemmatized]
    
        lemmas_to_be_demoted = \
    			[item.lower() for item in sentence_for_demoting_lemmas \
        					if item.lower() not in stop_words+punctuations]
    
    alignments = align(sentence1, sentence2, 
                       sentence1_parse_result, sentence2_parse_result)[0]
    
    # store the lemmatized words from each sentence
    sentence1_lemmas = [item[3] for item in sentence1_lemmatized]
    sentence2_lemmas = [item[3] for item in sentence2_lemmatized]

    # checks for the words except stop_words and punctuations and 
    # lemmas to be demoted.
    sentence1_content_lemmas = \
            [item for item in sentence1_lemmas \
                      if item.lower() not in \
                            stop_words+punctuations+lemmas_to_be_demoted]

    sentence2_content_lemmas = \
            [item for item in sentence2_lemmas \
					if item.lower() not in \
                             stop_words+punctuations+lemmas_to_be_demoted]
            
    # this returns empty sentences I guess.
    if sentence1_content_lemmas == [] or sentence2_content_lemmas == []:
        return (0, 0, parse_results)
    
    # finds only the content words which are lemmatized. Ignores stop words
    # , punctuations and lemmas to be demoted. 
    sentence1_aligned_content_word_indexes = \
		[item[0] for item in alignments if \
				sentence1_lemmas[item[0]-1].lower() not in \
                                stop_words+punctuations+lemmas_to_be_demoted]

    sentence2_aligned_content_word_indexes = \
		[item[1] for item in alignments if \
				sentence2_lemmas[item[1]-1].lower() not in \
                                stop_words+punctuations+lemmas_to_be_demoted]
    
    # calculate the similarity score
    # semantic similarity score = no. of aligned content in both/total lemmatized content
    sim_score = (len(sentence1_aligned_content_word_indexes) + \
	             len(sentence2_aligned_content_word_indexes)) / \
                        				(len(sentence1_content_lemmas) + \
                        	              len(sentence2_content_lemmas))

    # coverage is the measure of the amount of content words found in students 
    # response that are similar to the refrence answer.
    coverage = len(sentence1_aligned_content_word_indexes) / \
                                           len(sentence1_content_lemmas) 

    return (sim_score, coverage, parse_results)


def sts_cvm(sentence1, sentence2,
            parse_results,
            sentence_for_demoting=None,):
    '''
    This function does cosine vector matching (cvm)? :
        1. Loads the word embeddings from Baroni et al., 2014.
        2. Parses the sentences to be processed using SNLP.
        3. Lemmatizes the sentences to be processed SNLP.
        4. Finds the word vectors from the embeddings corresponding 
        to the lemmas.
        5. Sums the embeddings from each sentence into one single sentence 
        vector.
    '''
    # Loads the word embeddings from Baroni et al., 2014.
    global embeddings
    
    if embeddings == {}:
        print 'loading embeddings...'
        embeddings = \
           load_embeddings('/Users/gouravmahapatr/Dropbox/EdtechProject(Shrey,Gourav)/AGS (Automated Grading System)/codes/short-answer-grader-master/Resources/EN-wform.w.5.cbow.neg10.400.subsmpl.txt')
        print 'done'    

    sentence1_parse_result = parse_results[0]
    sentence2_parse_result = parse_results[1]
    
    sentence1_lemmatized = lemmatize(sentence1_parse_result)
    sentence2_lemmatized = lemmatize(sentence2_parse_result)

    lemmas_to_be_demoted = []
    if sentence_for_demoting != None:
        sentence_for_demoting_parse_result = parse_results[2]

        sentence_for_demoting_lemmatized = \
                            lemmatize(sentence_for_demoting_parse_result)
    
        sentence_for_demoting_lemmas = \
                        [item[3] for item in sentence_for_demoting_lemmatized]
    
        lemmas_to_be_demoted = \
    			[item.lower() for item in sentence_for_demoting_lemmas \
        					if item.lower() not in stop_words+punctuations]

    sentence1_lemmas = [item[3].lower() for item in sentence1_lemmatized]
    sentence2_lemmas = [item[3].lower() for item in sentence2_lemmatized]

    #sentence1_lemmas[:] = sorted(sentence1_lemmas)
    #sentence2_lemmas[:] = sorted(sentence2_lemmas)
    
    if sentence1_lemmas == sentence2_lemmas:
        return 1

    # load the embeddings corresponding to the lemma in sentence1
    sentence1_content_lemma_embeddings = []
    for lemma in sentence1_lemmas:
        if lemma.lower() in stop_words+punctuations+lemmas_to_be_demoted:
            continue
        if lemma.lower() in embeddings:
            sentence1_content_lemma_embeddings.append(
                                            embeddings[lemma.lower()])

    # load the embeddings corresponding to the lemma in sentence2
    sentence2_content_lemma_embeddings = []
    for lemma in sentence2_lemmas:
        if lemma.lower() in stop_words+punctuations+lemmas_to_be_demoted:
            continue
        if lemma.lower() in embeddings:
            sentence2_content_lemma_embeddings.append(
                                            embeddings[lemma.lower()])

    if sentence1_content_lemma_embeddings == \
                       sentence2_content_lemma_embeddings:
        return 1
    elif sentence1_content_lemma_embeddings == [] or \
         sentence2_content_lemma_embeddings == []:
        return 0
    
    # perform vector summing of each word embeddings to form a sentence vector.
    sentence1_embedding = vector_sum(sentence1_content_lemma_embeddings)
    sentence2_embedding = vector_sum(sentence2_content_lemma_embeddings)
    
    return cosine_similarity(sentence1_embedding, sentence2_embedding)
    
    
def length_ratio(sentence1, sentence2, parse_results):
    '''
    This function:
        1. Parses the data using SNLP.
        2. Lemmatizes the data using SNLP.
        3. Finds the content lemmas.
        4. Returns a simple length ratio where 
        length ratio = len(sentence1_content_lemmas) / 
                                         len(sentence2_content_lemmas)
    '''
    
    sentence1_parse_result = parse_results[0]
    sentence2_parse_result = parse_results[1]
        
    sentence1_lemmatized = lemmatize(sentence1_parse_result)
    sentence2_lemmatized = lemmatize(sentence2_parse_result)
    
    sentence1_lemmas = [item[3] for item in sentence1_lemmatized]
    sentence2_lemmas = [item[3] for item in sentence2_lemmatized]

    sentence1_content_lemmas = \
            [item for item in sentence1_lemmas \
                      if item.lower() not in \
                            stop_words+punctuations]

    sentence2_content_lemmas = \
            [item for item in sentence2_lemmas \
					if item.lower() not in \
                             stop_words+punctuations]
    
    if sentence2_content_lemmas == []:
        return len(sentence1_lemmas) / len(sentence2_lemmas)

    return len(sentence1_content_lemmas) / len(sentence2_content_lemmas)
