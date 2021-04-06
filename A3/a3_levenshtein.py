import os
import numpy as np
import string
import fnmatch
import sys

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    n, m = len(r), len(h)
    # basecase
    if m == 0:
        return 1, 0, 0, n
    elif n == 0:
        return float("inf"), 0, m, 0
    # cost for r[i] and h[j]
    C = np.zeros((n, m))

    for i in range(n):
        C[i, 0] = i

    for j in range(m):
        C[0, j] = j

    for i in range(1, n):
        for j in range(1, m):
            if r[i] == h[j]:
                C[i, j] = C[i - 1, j - 1]
            else:
                min_C = min(C[i - 1, j], C[i, j - 1], C[i-1, j - 1])
                # Deletion
                if min_C == C[i - 1, j]:
                    C[i, j] = C[i - 1, j] + 1
                # Insertion
                elif min_C == C[i, j - 1]:
                    C[i, j] = C[i, j - 1] + 1
                # Replace
                else:
                    C[i, j] = C[i - 1, j - 1] + 1
    # calculate number of operations
    nD, nI, nS = 0, 0, 0
    i, j = n - 1, m - 1
    while i > 0 and j > 0:
        if C[i, j] == C[i - 1, j] + 1:
            nD += 1
            i -= 1
        elif C[i, j] == C[i, j - 1] + 1:
            nI += 1
            j -= 1
        elif C[i, j] == C[i - 1, j - 1] + 1:
            nS += 1
            i -= 1
            j -= 1
        else:
            i -= 1
            j -= 1
    if i > 0:
        nD += i
    if j > 0:
        nI += j

    return (nD + nI + nS) / n, nS, nI, nD


def preprocess(line):
    """
    Removing all punctuations other than [ and ], and setting the context to lowercase.
    """
    punctuations = string.punctuation.replace("[", "").replace("]", '')
    line = line.strip().translate(str.maketrans("", "", punctuations))
    return line.lower().split()


def readFromFIle(FilePath):
    f = open(FilePath, "r")
    fileContext = f.readlines()
    f.close()
    return fileContext

if __name__ == "__main__":
    google_wer, kaldi_wer = [], []
    sys.stdout = open("asrDiscussion1.txt", "w")
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # read from transcripts
            google_trans = readFromFIle(os.path.join(dataDir, speaker, "transcripts.Google.txt"))
            kaldi_trans = readFromFIle(os.path.join(dataDir, speaker, "transcripts.Kaldi.txt"))
            human_trans = readFromFIle(os.path.join(dataDir, speaker, "transcripts.txt"))

            for i, raw_r in enumerate(human_trans):
                r, g_hypo, k_hypo = preprocess(raw_r), preprocess(google_trans[i]), preprocess(kaldi_trans[i])
                wer, numS, numI, numD = Levenshtein(r, g_hypo)

                g_out = "{speaker} Google {i} {wer} S:{numS}, I:{numI}, D:{numD}\n".format(
                    speaker=speaker, i=i, wer=round(wer, 3), numS=numS, numI=numI, numD=numD
                )
                google_wer.append(wer)

                wer, numS, numI, numD = Levenshtein(r, k_hypo)

                k_out = "{speaker} Kaldi {i} {wer} S:{numS}, I:{numI}, D:{numD}\n".format(
                    speaker=speaker, i=i, wer=round(wer, 3), numS=numS, numI=numI, numD=numD
                )
                kaldi_wer.append(wer)
                print(g_out, k_out)
    print("Google wer mean: ", np.mean(google_wer), " std: ", np.std(google_wer))
    print("Kaldi wer mean: ", np.mean(kaldi_wer), " std: ", np.std(kaldi_wer))
    sys.stdout.close()
