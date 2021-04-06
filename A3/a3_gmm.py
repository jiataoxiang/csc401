from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import sys

from scipy.special import logsumexp
dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    Mu = myTheta.mu
    Sigma = myTheta.Sigma
    d = Mu.shape[1]
    # log (a / (b * c)) = log a - log b - log c
    loga = -0.5 * np.sum(np.square(x - Mu[m]) / Sigma[m], axis=-1)
    logb = (d / 2) * np.log(2 * np.pi)
    logc = 0.5 * np.sum(np.log(Sigma[m]))

    return loga - logb - logc

def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M, T = myTheta.mu.shape[0], x.shape[0]
    logOmega = np.log(myTheta.omega).reshape(-1, 1)
    log_Bs = np.zeros((M, T))
    for i in range(M):
        log_Bs[i] = log_b_m_x(i, x, myTheta)
    denom = stableLogsumExp(logOmega + log_Bs)
    # print(denom.shape)
    return (logOmega[m] + log_Bs[m] - denom).reshape(-1)

    
# def log_p_m_x( log_Bs, myTheta):
#     ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
#         See equation 2 of handout
#     '''
#     logOmega = np.log(myTheta.omega).reshape(-1, 1)
#     denominator = stableLogsumExp(logOmega + log_Bs)
#     return logOmega + log_Bs - denominator



def stableLogsumExp(x):
    a = np.max(x, axis=0, keepdims=True)
    return a + logsumexp(x - a, axis=0)


def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    logp = stableLogsumExp(np.log(myTheta.omega).reshape(-1, 1) + log_Bs)
    return np.sum(logp)


def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    # Initialize theta
    myTheta = theta( speaker, M, X.shape[1] )
    T, d = X.shape
    myTheta.mu = X[np.random.choice(T, size=M, replace=False)] # randomly select data point
    myTheta.Sigma = np.ones((M, d)) # initialize to identity matrix
    myTheta.omega = np.full((M, 1), 1.0/M) # sum to 1
    # print(myTheta.omega)

    i = 0
    prev_L, improvement = -float('inf'), float('inf')
    while i < maxIter and improvement > epsilon:
        # compute intermediate result
        log_Bs = np.zeros((M, T))
        for m in range(M):
            log_Bs[m] = log_b_m_x(m, X, myTheta)
        # ==============================================
        # IMPORTANT!!!! USE LINE BELOW TO TRAIN THE MODEL
        # YOU HAVE TO COMMENT OUT THE FUNCTION WITH PROTOTYPE log_p_m_x( log_Bs, myTheta)
        # logp = log_p_m_x(log_Bs, myTheta)
        # ==============================================
        # DON't USE THE CODE BELOW
        logp = np.zeros((M, T))
        for m in range(M):
            logp[m] = log_p_m_x(m, X, myTheta)
        # ==============================================
        L = logLik(log_Bs, myTheta)
        px = np.exp(logp)
        # update parameters 
        myTheta.omega = np.sum(px, axis=1).reshape(-1, 1) / T
        myTheta.mu = np.matmul(px, X) / np.sum(px, axis=1).reshape(-1, 1)
        myTheta.Sigma = np.matmul(px, np.square(X)) / np.sum(px, axis=1).reshape(-1, 1) - np.square(myTheta.mu)
        
        improvement = L - prev_L
        prev_L = L
        i += 1
        
    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    M, d = models[0].Sigma.shape
    T = mfcc.shape[0]

    logLikelihoods = []
    for i, theta in enumerate(models):
        log_Bs = np.zeros((M, T))
        for m in range(M):
            log_Bs[m] = log_b_m_x(m, mfcc, theta)
        L = logLik(log_Bs, theta)
        logLikelihoods.append([L, theta.name, i]) # likelihood, model name, index
    
    best_K = k if k > 0 else 1
    logLikelihoods = sorted(logLikelihoods, key=lambda x: x[0], reverse=True)[:best_K]

    bestModel = logLikelihoods[0][2]
    if k > 0:
        print(models[correctID].name)
        for i in range(k):
            print(logLikelihoods[i][1], logLikelihoods[i][0])
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    # default setting
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    # d = 13
    # k = 5  # number of top speakers to display, <= 0 if none
    # M = 2
    # epsilon = 0.5
    # maxIter = 2
    # num_speaker = 30
    # train a model for each speaker, and reserve data for testing
    sys.stdout = open("gmmLikes.txt", "w")
    for subdir, dirs, files in os.walk(dataDir):
        for i, speaker in enumerate(dirs):
            # print(i)
            # if i >= num_speaker:
            #     continue
            # print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    # print(len(testMFCCs))
    print(accuracy)
    sys.stdout.close()

