''' Given a question, a reference answer, and a student response, this 
    pretrained grader computes a score in % for the student answer based on
    how semantically similar it is to the reference answer. '''

from featureExtraction import *
import numpy as np


def extract_features(question, ref_answer, student_response):
    
    # Semantic texual similarity alignment. Returns:
    # sim_alignment = similarity measure
    # cov_alignment = coverage measure
    # parse results
    sim_alignment, cov_alignment, parse_results = \
                            sts_alignment(ref_answer, student_response)
    
    # this is the question demotion feature
    q_demoted_sim_alignment, q_demoted_cov_alignment, _ = \
                            sts_alignment(ref_answer, student_response,
                                          parse_results,
                                          question)
    
    # cosine vector matching. Returns the cosine similarity.                        
    sim_cvm = sts_cvm(ref_answer, student_response, parse_results)
    
    # question demotion in cvm
    q_demoted_sim_cvm = sts_cvm(ref_answer, student_response,
                                parse_results,
                                question)
    
    # finds the length ratio.
    lr = length_ratio(ref_answer, student_response, parse_results)    

    # assembles the whole feature vector here. 
    feature_vector = (sim_alignment, cov_alignment,
                      q_demoted_sim_alignment, q_demoted_cov_alignment,
                      sim_cvm,
                      q_demoted_sim_cvm,
                      lr)
    
    return feature_vector
    

def grade(question, ref_answer, student_response):
    
    feature_vector = extract_features(question, ref_answer, student_response)


    ''' The following weights are learned from the dataset (all examples) 
        reported by Mohler, Bunescu, and Mihalcea in their 2011 paper "Learning
        to Grade Short Answer Questions using Semantic Similarity Measures and 
        Dependency Graph Alignments" '''
    w = [.572813656856, .06392481, .04797512, .00484819, .19755009,
         .07454364, .13800417, .00385674]
    
    score = np.dot(w, [1]+list(feature_vector))
            
    score = min(1, score)
    score = max(0, score)
    score *= 100
            
    return int((score * 100) + 0.5) / 100.0
     


''' below is an example of grading an answer '''
question = "How are infix expressions evaluated by computers?"

ref_answer = "First, they are converted into postfix form, " + \
             "followed by an evaluation of the postfix expression."
student_response = "computers usually convert infix expressions to postfix " +\
                   "expression and evaluate them using a stack."
  
#student_response = " The postfix form is evaluated after the infix expressions are converted into postfix form. "                 
                   
score = grade(question, ref_answer, student_response)
print
print 'score for this student response = '+str(score)+ '%'