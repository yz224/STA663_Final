import os
import sys
import glob
import numpy as np
import random
import itertools
import math
import scipy.stats as sps
import numpy.linalg as npl
import operator
import pytest

# content of test_sample.py
# Function used to calculate values in P(D|H_1)
def k_n(n,k0=1.0):
    return k0+n

def v_n(n,v0=1.0):
    return v0+n

def S(x,n):
    return np.std(x,axis=0)*n

def sigma_n(x,n,mu0,sigma0=np.matrix([[1.0,0.0],[.0,1.0]]),v0=1.0,k0=1.0):
    return sigma0 + S(x,n) + (k0*n/(k0+n))*(np.dot(np.mean(x,axis=0)-mu0,np.transpose(np.mean(x,axis=0)-mu0)))

def P_DH(x,n,mu0,sigma0=np.matrix([[1.0,0.0],[0.0,1.0]]),v0=1.0,k0=1.0,d=2.0):
    return (math.gamma(v_n(n)/2.0)/math.gamma(v0/2.0))*((npl.det(sigma0)**(v0/2.0))/(npl.det(sigma_n(x,n,mu0))**(v_n(n)/2.0)))*(k0/k_n(n))**(d/2.0)*1/(math.pi)**(n*d/2.0)

# Test functions values
def test_k_n():
    assert k_n(5) == 6
    
def test_v_n():
    assert v_n(3) == 4
    
def test_S():
    test_data = np.array([[[1,0],[2,1]],[[2,2],[1,1]]])
    result_value = np.array([[1,2],[1,0]])
    assert npl.norm(S(test_data,2) - result_value)<= 0.001

def test_PDH():
    test_x = np.array([[[2,3],[2,1]],[[2,2],[1,1]]])
    test_mu = np.array([1,1])
    result_value = 0.006363984588752723270399513922 
    assert abs(P_DH(test_x,2,test_mu) - result_value) < 0.0000001