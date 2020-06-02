#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df_Xtr = pd.read_csv('Xtr.csv') # Loading the training data 
df_Ytr = pd.read_csv('Ytr.csv') # Loading the training label data
df_Xte = pd.read_csv('Xte.csv') # Loading the testing data


# In[3]:


df_Xtr.iloc[0][1] # The information in the sample data of index 0



df = []
for i in range(df_Xtr.shape[0]):
    l = []
    for j in range(len(df_Xtr.iloc[i][1])):
        char = df_Xtr.iloc[i][1][j]
        if char == 'A':
            l = l+[1, 0, 0, 0]
        elif char == 'C':
            l = l+[0, 1, 0, 0]
        elif char == 'G':
            l = l+[0, 0, 1, 0]
        else :
            l = l+[0, 0, 0, 1]
    df.append(l)


# In[5]:


df_train = pd.DataFrame(df)


# #### <font color='blue'> The training Label data.

# In[6]:


y = df_Ytr['Bound'].values 


df_t = []
for i in range(df_Xte.shape[0]):
    l = []
    for j in range(len(df_Xte.iloc[i][1])):
        char = df_Xte.iloc[i][1][j]
        if char == 'A':
            l = l+[1, 0, 0, 0]
        elif char == 'C':
            l = l+[0, 1, 0, 0]
        elif char == 'G':
            l = l+[0, 0, 1, 0]
        else :
            l = l+[0, 0, 0, 1]
    df_t.append(l)


# In[8]:


df_test = pd.DataFrame(df_t)


# In[9]:


df_test = df_test.values



df_train = df_train.astype('float') 



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[12]:


test = SelectKBest(score_func=chi2, k=100)
fit = test.fit(df_train, y)
np.set_printoptions(precision=3)
df_train = fit.transform(df_train)


# In[13]:


df_test = fit.transform(df_test)



df_train.shape



df_test.shape




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.25)



y_train_keep = y_train



print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))




y_new = []
for i in range(len(y_train)):
    if y_train[i] == 0:
        y_new.append(-1)
    else:
        y_new.append(1)


# In[20]:


y_train[:10]


# In[21]:


y_train = np.array(y_new)
y_train[:10]


def get_primal_from_dual(alpha, X, y, hard_margin=False, C=None, tol=1e-10):
    # w parameter in vectorized form
    w = ((y * alpha).T.dot(X)).flatten()
    # w = X.dot(y[:, np.newaxis]*alphas)
    #print(alpha)
    # sv = Support vectors!
    # Indices of points (support vectors) that lie exactly on the margin
    # Filter out points with alpha == 0
    sv = (alpha > tol)
    # If soft margin, also filter out points with alpha == C
    if not hard_margin:
        if C is None:
            raise ValueError('C must be defined in soft margin mode')
            print(C)
        sv = np.logical_and(sv, (C - alpha > tol))
    #print('y',sv[:100])
    b = y[sv] - X[sv].dot(w)
    b = b[0]
    
    #Display results
    # print('Alphas = {}'.format(alpha[sv]))
    # print('Number of support vectors = {}'.format(sv.sum()))
#     print('w = {}'.format(w))
#     print('b = {}'.format(b))
    
    return w, b



import cvxopt
solver='cvxopt'
from cvxopt import matrix, solvers


# In[24]:


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    print(A)
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = - G.T
        qp_b = - h
        meq = 0
    print(G.shape)
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def cvxopt_qp(P, q, G, h, A, b):
    P = .5 * (P + P.T)
    # print(A.shape)
    cvx_matrices = [
        cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
    ]
    solution = cvxopt.solvers.qp(*cvx_matrices)

    return np.array(solution['x']).flatten()
solve_qp = {'quadprog': quadprog_solve_qp, 'cvxopt': cvxopt_qp}[solver]




def svm_primal_soft_to_qp(X, y, C):
    n, p = X.shape
    assert (len(y) == n)
    
    Xy = np.diag(y).dot(X)
    # Primal formulation, soft margin
    diag_P = np.hstack([np.ones(p), np.zeros(n + 1)])
    # As a regularization, we add epsilon * identity to P
    eps = 1e-12
    diag_P += eps
    P = np.diag(diag_P)
    
    q = np.hstack([np.zeros(p + 1), C * np.ones(n)])
    # y(wx+b)+ei>=1
    G1 = - np.hstack([Xy, y[:, np.newaxis], np.eye(n)])
    # ei>=0
    G2 = - np.hstack([np.zeros((n, p+1)), np.eye(n)])
    G = np.vstack([G1, G2])
    h = - np.hstack([np.ones(n), np.zeros(n)])
    A = None
    b = None
    return P, q, G, h, A, b
C = 10
coefs = solve_qp(*svm_primal_soft_to_qp(X_train, y_train, C=C))
n, p = X_train.shape
w_soft, b_soft, e = coefs[:p], coefs[p], coefs[(p+1):]


def svm_dual_soft_to_qp(X, y, C):
    n, p = X.shape
    assert (len(y) == n)
    
    Xy = np.diag(y).dot(X)
    # Dual formulation, soft margin
    P = Xy.dot(Xy.T)
    # As a regularization, we add epsilon * identity to P
    eps = 1e-20
    P += eps * np.eye(n)
    q = - np.ones(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.hstack([np.zeros(n), C * np.ones(n)])
    A = y[np.newaxis, :]
    b = np.array([0.])
    A = A.astype('float')
    G = G.astype('float')
    return P, q, G, h, A, b
alphas = solve_qp(*svm_dual_soft_to_qp(X_train, y_train, C=C))

w_soft, b_soft =  get_primal_from_dual(alphas, X_train, y_train, C=C ) 

def score_1(X, y, w, b):
    # print(X_train.shape)
    # print(w.shape)
    prediction = np.sign(X@w+b)
    # print(prediction[:4])
    y_new = [1 if pred == 1 else 0 for pred in prediction]

    sum_goodpred = 0

    for i in range(len(y)):
        if y[i]==y_new[i]:
            sum_goodpred+= 1
    accur = sum_goodpred/len(y_new)
    print('the accuracy for weight w, b is: {}%'.format((100*accur)))



y_train = y_train_keep 
y_train[:10]


score_1(X_train, y_train, w_soft, b_soft)


# In[31]:


score_1(X_test, y_test, w_soft, b_soft) 


if __name__=='main':
	solve_qp = {'quadprog': quadprog_solve_qp, 'cvxopt': cvxopt_qp}[solver]
	C = 20
	coefs = solve_qp(*svm_primal_soft_to_qp(X_train, y_train, C=C))
	n, p = X_train.shape
	w_soft, b_soft, e = coefs[:p], coefs[p], coefs[(p+1):]
	alphas = solve_qp(*svm_dual_soft_to_qp(X_train, y_train, C=C))
	w_soft, b_soft =  get_primal_from_dual(alphas, X_train, y_train, C=C ) 
	y_train = y_train_keep 
	score_1(X_train, y_train, w_soft, b_soft)
	score_1(X_test, y_test, w_soft, b_soft) 




