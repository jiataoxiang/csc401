#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json

# self import lib
import string
import csv
import numpy as np
import os

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

# get global variable for norms
BGNorm = {}
with open("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv", mode="r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    for row in csv_reader:
        if "WORD" in row:
            BGNorm[row["WORD"]] = [row['AoA (100-700)'], row['IMG'], row['FAM']]

Warringer = {}
with open("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv", mode="r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    for row in csv_reader:
        if "Word" in row:
            Warringer[row["Word"]] = [row['V.Mean.Sum'], row['A.Mean.Sum'], row['D.Mean.Sum']]

classes = ["Left", "Center", "Right", "Alt"]
LIWC = []
ids = [{}, {}, {}, {}]
for i, subroute in enumerate(classes):
    id_file_name = os.path.join("/u/cs401/A1/feats/", subroute + "_IDs.txt")
    with open(id_file_name, mode="r") as id_file:
        for order, line in enumerate(id_file):
            comment_id = line.rstrip("\n")
            ids[i][comment_id] = order # store comment id
    feats_file_name = os.path.join("/u/cs401/A1/feats/", subroute + "_feats.dat.npy")
    LIWC.append(np.load(feats_file_name))


# helper
def is_multi_punctation(word):
    return len(word) > 1 and all([char in string.punctuation for char in word])

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''   
    # print('TODO')
    feats = np.zeros(174)
    sentences = comment.split("\n")
    if sentences[-1] == "\n": # if it ends with \n, remove empty string
        sentences.pop()
    # TODO: Extract features that rely on capitalization.
    total_num_sentences = len(sentences)
    total_num_tokens = 0
    AoA_list, IMG_list, FAM_list = [0], [0], [0]
    V_list, A_list, D_list = [0], [0], [0]
    for sent in sentences:
        tokens = sent.split(" ")
        total_num_tokens += len(tokens)
        for i, token in enumerate(tokens):
            # if token.find("/") == -1: # empty string or a space
            #     continue
            word_tag = token.split("/")
            # print(word_tag)
            if len(word_tag) < 2:
                continue
            word = word_tag[0]
            tag = word_tag[1]
            # print(word, tag)
            # feature 1: Number of tokens in uppercase (≥ 3 letters long)
            if word.isupper() and len(word) >= 3: feats[0] += 1
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
            word = word.lower()
    # TODO: Extract features that do not rely on capitalization.
            # feature 2: Number of first-person pronouns
            if word in FIRST_PERSON_PRONOUNS: feats[1] += 1
            # feature 3: Number of second-person pronouns
            if word in SECOND_PERSON_PRONOUNS: feats[2] += 1
            # feature 4: Number of third-person pronouns
            if word in THIRD_PERSON_PRONOUNS: feats[3] += 1
            # feature 5: Number of coordinating conjunctions
            if tag == "CC": feats[4] += 1
            # feature 6: Number of past-tense verbs
            if tag == "VBD": feats[5] += 1
            # feature 7: Number of future-tense verbs
            if word.endswith("'ll") or word in {"will", "gonna"}: feats[6] += 1
            if tag == "VB" and i > 1 and tokens[i - 1].startswith("to/") and tokens[i - 2].startswith("going/"): feats[6] += 1
            # feature 8: Number of commas
            if word == ",": feats[7] += 1
            # feature 9: Number of multi-character punctuation tokens
            if is_multi_punctation(word): feats[8] += 1
            # feature 10: Number of common nouns
            if tag == "NN" or tag == "NNS": feats[9] += 1
            # feature 11: Number of proper nouns
            if tag == "NNP" or tag == "NNPS": feats[10] += 1
            # feature 12: Number of adverbs
            if tag in {"RB", "RBR", "RBS"}: feats[11] += 1
            # feature 13: Number of wh- words
            if tag in {"WDT", "WP", "WP$", "WRB"}: feats[12] += 1
            # feature 14: Number of slang acronyms
            if word in SLANG: feats[13] += 1  
            # feature 16: Average length of tokens, excluding punctuation-only tokens, in characters
            if is_multi_punctation(word) or (len(word) == 0 and word in string.punctuation): feats[15] += 1
            # feature 18: Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
            # feature 19: Average of IMG from Bristol, Gilhooly, and Logie norms
            # feature 20: Average of FAM from Bristol, Gilhooly, and Logie norms
            # feature 21: Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
            # feature 22: Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
            # feature 23: Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
            if word in BGNorm:
                # print(BGNorm[word])
                # break
                AoA, IMG, FAM = [float((val if val != "" else 0)) for val in BGNorm[word]]
                AoA_list.append(AoA)
                IMG_list.append(IMG)
                FAM_list.append(FAM)
            # feature 24: Average of V.Mean.Sum from Warringer norms
            # feature 25: Average of A.Mean.Sum from Warringer norms
            # feature 26: Average of D.Mean.Sum from Warringer norms
            # feature 27: Standard deviation of V.Mean.Sum from Warringer norms
            # feature 28: Standard deviation of A.Mean.Sum from Warringer norms
            # feature 29: Standard deviation of D.Mean.Sum from Warringer norms
            if word in Warringer:
                # print(Warringer[word])
                V, A, D = [float((val if val != "" else 0)) for val in Warringer[word]]
                V_list.append(V)
                A_list.append(A)
                D_list.append(D)
    # feature 15: Average length of sentences, in tokens
    feats[14] =  total_num_tokens/total_num_sentences
    # feature 16 updates
    feats[15] /= total_num_tokens
    # feature 17: Number of sentences.
    feats[16] = total_num_sentences
    # convert to numpy array
    # if len(AoA_list) == 0 or len(IMG_list) == 0 or len(FAM_list) == 0 or len(V_list) == 0 or len(A_list) == 0 or len(D_list) == 0:
    #     print(comment)
    #     print(AoA_list, IMG_list, FAM_list, V_list, A_list, D_list)
    AoA_list = np.array(AoA_list)
    IMG_list = np.array(IMG_list)
    FAM_list = np.array(FAM_list)
    V_list = np.array(V_list)
    A_list = np.array(A_list)
    D_list = np.array(D_list)

    # feature 18-23
    # print(AoA_list)
    feats[17] = np.mean(AoA_list)
    feats[18] = np.mean(IMG_list)
    feats[19] = np.mean(FAM_list)
    feats[20] = np.std(AoA_list)
    feats[21] = np.std(IMG_list)
    feats[22] = np.std(FAM_list)
    # feature 24-29
    feats[23] = np.mean(V_list)
    feats[24] = np.mean(A_list)
    feats[25] = np.mean(D_list)
    feats[26] = np.std(V_list)
    feats[27] = np.std(A_list)
    feats[28] = np.std(D_list)
    # print("done")
    return feats

          
    
    
def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    # print('TODO')
    comment_class = classes.index(comment_class)
    comment_id_index = ids[comment_class][comment_id]
    feat[29:173] = LIWC[comment_class][comment_id_index]
    return feat




def main(args):
    #Declare necessary global variables here. 

    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Call extract1 for each datatpoint to find the first 29 features. 
    # Add these to feats.
    for i, comment in enumerate(data):
        comment_class = comment["cat"]
        feat = extract1(comment["body"])
        feats[i][:29] = feat[:29]
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
        feats[i][29:173] = extract2(feat, comment_class, comment["id"])[29:173]
        feats[i][173] = classes.index(comment_class)
    
    # print('TODO')

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

