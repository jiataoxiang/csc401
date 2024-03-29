#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
 

import sys
import argparse
import os
import json
import re
import spacy

# self import lib
import html


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  
        #modify this to handle other whitespace chars.
        #replace newlines with spaces
        modComm = re.sub(r"[\n\t\r]+", " ", modComm)

    if 2 in steps:  # unescape html
        # print("TODO")
        modComm = html.unescape(modComm)

    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
        
    if 4 in steps: #remove duplicate spaces.
        # print("TODO")
        modComm = re.sub(r" +", " ", modComm)

    if 5 in steps:
        # print("TODO")
        # TODO: get Spacy document for modComm
        sentences = nlp(modComm)
        modComm = []
        # TODO: use Spacy document for modComm to create a string.
        for sent in sentences.sents:
            sentence = []
            for token in sent:
                # lemmatization
                if token.lemma_.startswith("-") and not token.text.startswith("-"):
                    if token.text.isupper():
                        text = token.text
                    else:
                        text = token.text.lower()
                else:
                    if token.text.isupper():
                        text = token.lemma_.upper()
                    else:
                        text = token.lemma_.lower()
                with_tag = text + "/" + token.tag_ #tagging
                sentence.append(with_tag)
            modComm.append(" ".join(sentence) + "\n")
        modComm = "".join(modComm)
        # Make sure to:
        #    * Insert "\n" between sentences.
        #    * Split tokens with spaces.
        #    * Write "/POS" after each token.
            
    
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            if len(data) < args.max:
                samples = data
            else:
            # TODO: select appropriate args.max lines
                start_line = args.ID[0] % len(data)
                end = start_line + args.max
                end_line = end % len(data)
                samples = data[start_line:end_line] if end_line < len(data) else data[:end_line] + data[start_line:]
            for line in samples:
                result = {}
            # TODO: read those lines with something like `j = json.loads(line)`
                line_js_format = json.loads(line)
            # TODO: choose to retain fields from those lines that are relevant to you
                result["id"] = line_js_format["id"]
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
                result["body"] = preproc1(line_js_format["body"])
                result["cat"] = file
            # TODO: append the result to 'allOutput'
                allOutput.append(result)
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput, indent=2))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
