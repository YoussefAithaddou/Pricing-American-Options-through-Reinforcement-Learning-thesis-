#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import math
import random
import numpy as np
from statistics import mean 
from statistics import variance
from scipy import optimize
import matplotlib.pyplot as plt
import scipy.special
import pandas as pd



#project parameters:

T=1
m=4
dt=1.0/4.0
A="P"
r=0.025
K=140
k=7
s=0.49
S0=130
sigma=0.39
V0 = 1.7
kappa = 2
theta=0.045
rho=-0.5711
c=0.216

N=20000
P=[]
S=[]
G=[]      
epsilon=0
    

    
    
    
#Black-Scholes Model:    
    
def BS(S0, r, m, sigma, T):    
    d=float(T)/float(m)
    SSInf=[S0]
    SSup=[S0]  
    gaussians = np.random.normal(0, 1, m)

    for i in range(m):
        SSInf.append(SSInf[-1]*math.exp((r-(1/2)*(sigma**2))*d + sigma*math.sqrt(d)*gaussians[i]))
        SSup.append(SSup[-1]*math.exp((r-(1/2)*(sigma**2))*d - sigma*math.sqrt(d)*gaussians[i]))

    return ([SSInf[24],SSInf[49] ,SSInf[74] ,SSInf[99] ],  [SSup[24],SSup[49] ,SSup[74] ,SSup[99] ])


#Heston Model:

def H(S0,V0, r, m, sigma,kappa,theta, T):
	dt= 1/100.0
	V=[V0]
	V1=[V0]
	S=[S0] 
	S1=[S0]
	for i in range(m-1):
		x=np.random.normal(0, 1, 1)      
		y=rho*np.random.normal(0, 1, 1)+math.sqrt(1.0-rho*rho)*x
		S.append(float((S[-1]*(1+r*dt+math.sqrt(dt*V[-1])*x ))))
		V.append(float((V[-1]+kappa*(theta-V[-1])*dt+sigma*math.sqrt(dt*V[-1])*y)))
		S1.append(float((S[-1]*(1+r*dt+math.sqrt(dt*V[-1])*(-x) ))))
		V1.append(float((V[-1]+kappa*(theta-V[-1])*dt+sigma*math.sqrt(dt*V[-1])*(-y))))
	return([S[24],S[49] ,S[74] ,S[99] ],[S1[24],S1[49] ,S1[74] ,S1[99] ])





def laguerre(xx,nn):
    L=[]
    xx=xx/K
    for l in range (nn):
        L.append(math.exp(-xx/2)*scipy.special.eval_laguerre(l, xx, out=None))
    return(L)

def laguerre2(x,t):
    L=[1]
    t=t*dt
    
    L.append(math.exp(-x/(K*2)))
    L.append( math.exp(-x/(K*2))*(1 - x/(K*2))  )
    L.append( math.exp(-x/(K*2))*(1 - 2*x/K +((x/K)**2)/2 ))
    L.append(math.sin((math.pi*(-t))/(2*T)+ math.pi/2 ))
    L.append(math.log(T-t))
    L.append(((t/T)**2))
    return(np.matrix(L))


#Least squares Monte Carlo (LSM)

def LSM(S,P, K, A, r, T ,m, k, N, PVV ):
    
    Ex=0
    Cont=0
    BB=[0 for _ in range(m)]
    XX=[]
    YY=[]
    for j in range(m-2,0,-1):
        X=[]
        Y=[]
        for i in range (N):
            #P[i]*=math.exp(-r*dt)
            Y.append(PVV[i][-1])
            X.append(laguerre(S[i][j],k))
        try :
            YY=np.array(Y)  
            XX=np.array(X)  
            BB[j]=np.dot(np.dot(np.linalg.inv(np.dot(np.matrix.transpose(XX),XX)) ,np.matrix.transpose(XX)),YY)
        except :
            BB[j]=0.0
            continue
    pp=[]
    
    
    for i in range(N):
        value=0.0
        exerciced = False
        for j in range(m-1):
            if not exerciced :
                try:
                    conValue=np.dot(np.matrix.transpose(BB[j]),laguerre(S[i][j],k))*math.exp(-r*dt)
                except:
                    conValue=0.0
                if PVV[i][j]>conValue :
                    value=PVV[i][j+1]*math.exp(-r*dt*j)
                    exerciced=True
            if j==m-1 and not exerciced:
                value=PVV[i][-1]*math.exp(-r*dt*j)
        pp.append(value)
 
    return(pp)



#Fitted Q-iteration 

def FQI(S, K, r, T ,m,k, N, PVV ):
    F=[]
    D=[]
    R=[]
    Y=[0.0 for i in range (N)]
    A=np.matrix([[0.0 for i in range (k)]for jj in range (k)])
    B=np.matrix([0.0 for i in range (k)])
    w=np.matrix([0.0 for i in range (k)])
    P=np.matrix([0 for i in range(k)])  
    x1=100

    for i in range (N):
       
        for j in range(0, m-1):
            
            Q=PVV[i][j+1]
            
            
            if (j<m-2) :
                P= laguerre2(S[i][j+1],j+1)
            
            else:
                P=np.matrix([0 for i in range(k)])
            

            A += (laguerre2(S[i][j],j).transpose())*laguerre2(S[i][j],j) 
            B += math.exp(-r*dt)*( max(PVV[i][j+1],float(P*(w.transpose())))) * laguerre2(S[i][j],j)
        
        if (i+1)%10==0:
            w=B*np.linalg. inv(A)
            A=np.matrix([[0.0 for i in range (k)]for i in range (k)])
            B=np.matrix([0.0 for i in range (k)])
    
       
    for i in range (N):
        exerciced = False
        for j in range(m) :
            Q=PVV[i][j]
            if not exerciced :
                x=max(0.0,float(laguerre2(S[i][j],j)*(w.transpose())))
                if Q >=x*math.exp(-r*dt):
                    Y[i]=Q*math.exp(-r*j*dt)
                    exerciced= True
                    break
            if j==m-1 and not exerciced:
                    Y[i]=Q*math.exp(-r*j*dt)
    return(mean(Y))   



#Least-Squares Policy Iteration

def LSPI(S, K, r, T ,m,k, N,PVV ):
    F=[]
    D=[]
    R=[]
    Y=[0.0 for i in range (N)]
    A=np.matrix([[0.0 for i in range (k)]for jj in range (k)])
    B=np.matrix([0.0 for i in range (k)])
    w=np.matrix([0.0 for i in range (k)])
    P=np.matrix([0 for i in range(k)])  
    x1=100
    
    
    
    for i in range (N):
       
        for j in range(0, m-1):
            
            Q=PVV[i][j+1]
            
            x=float(laguerre2(S[i][j+1],j+1)*(w.transpose()))
            if (Q <= x and j<m-2) :
                P= laguerre2(S[i][j+1],j+1)
            
            else:
                P=np.matrix([0 for i in range(k)])
            
            x2=float(P*(w.transpose()))
           
            if Q > x2:
                R=Q 
            else:
                R=0.0

            A += (laguerre2(S[i][j],j).transpose())* ((laguerre2(S[i][j],j)-(math.exp(-r*dt)*P)))
            B += math.exp(-r*dt)* R * laguerre2(S[i][j],j)
        
        if (i+1)%10==0:
            w=B*np.linalg. inv(A)
            A=np.matrix([[0.0 for i in range (k)]for i in range (k)])
            B=np.matrix([0.0 for i in range (k)])
    
       
    for i in range (N):
        exerciced = False
        for j in range(m) :
            Q=PVV[i][j]
            if not exerciced :
                x=max(0.0,float(laguerre2(S[i][j],j)*(w.transpose())))
                if Q >=x*math.exp(-r*dt):
                    Y[i]=Q*math.exp(-r*j*dt)
                    exerciced= True
                    break
            if j==m-1 and not exerciced:
                    Y[i]=Q*math.exp(-r*j*dt)
    return(mean(Y)) 




#Aut-call payOff function

def PayOff(SS,j, x, c):
    SS=SS
    global Ko  
    if j==0:
        if (SS<1.2*140/(1+x)):
            return(-c)
        else:
            Ko=True
            return(SS/140.0*(1+x))
    elif j==1:
        if (SS<1.2*140/(1+x)):
            return(-c)
        else:
            Ko=True
            return(SS/140.0*(1+x))
    elif j==2:
        if (SS<1.2*140/(1+x)):
            return(-c)
        else:
            Ko=True
            return(SS/140.0*(1+x))
        
    elif j==3:
        return(SS/140.0*(1+x))

    

#Option expected avlue function    
def FOp(c):
    P=[]
    global PV
    for i in range (N):      
        Ko=False
        for j in range(4):
            if Ko==True:
                break
            PV[i][j]=PayOff(S[i][j],j, epsilon,c)

    for i in range (N):      
        for j in range(1,4):
            PV[i][j]+=PV[i][j-1]
        P.append(PV[i][-1])
    return(mean(P))




def f( Sjdid, t ):
    global c 
    global PV
    if t==True:
        #calculate the value of c
        c= optimize.newton(FOp, 0.5, tol=1e-03, maxiter=200)
        #print(c)
    for i in range (N):      
        Ko=False
        for j in range(4):
            if Ko==True:
                break
            PV[i][j]=PayOff(S[i][j],j, epsilon,c)

    Ait=LSM(S,P, K, A, r, T ,m,k, N, PV )
    a1=mean(Ait)
    b1=LSPI(S, K, r, T ,m,k, N, PV )
    c1=FQI(S, K, r, T ,m,k, N, PV )

    return a1,b1,c1


data1=[36.628578,37.378242,37.834881,38.059536,38.293949,37.43441,37.585804,37.287888,38.523491,38.166977,37.771385,40.352478,40.643066,40.662605,41.817623,42.533104,42.547756,41.741924,41.790977,41.55064,41.908688,41.734566,41.886616,41.793427,41.918495,42.188255,41.950378,42.418781,42.727783,42.7523,42.884735,42.462925,42.90926,43.125065,43.046589,42.798901,42.303516,42.404068,43.873039,44.365974,44.562164,45.057541,45.643658,46.109615,45.744209,46.143944,47.843437,46.85268,46.286179,45.807961,46.219963,46.281273,46.582916,46.899277,47.581043,47.907207,47.990585,48.311844,49.072079,48.924942,49.199608,48.790058,48.770443,48.858723,48.863628,49.815155,49.994179,50.158489,50.881939,50.803463,50.342419,50.102085,50.178104,49.211864,51.62746,51.291485,51.929108,51.127178,49.74894,49.758747,49.224133,48.542206,45.720963,46.444736,47.001102,46.794319,46.528439,45.073498,45.937603,44.997185,44.229099,44.059231,43.877056,43.667801,43.894287,43.099117,42.663376,44.224174,44.938099,45.597866,46.81155,47.409767,47.958752,47.806126,47.796272,47.449154,47.732269,48.854858,48.71207,49.1035,48.936092,48.886864,48.145855,49.187206,49.172436,48.724384,49.618027,49.908512,50.322102,50.27779,49.24136,49.541714,50.031612,49.667259,50.048847,50.519051,50.344257,50.061153,50.629833,49.874054,51.013874,51.412689,51.37085,50.964642,51.141891,51.61948,51.397919,52.446659,51.311756,50.226093,47.596859,48.497894,49.000103,50.080845,49.668159,49.542126,51.640156,50.103088,49.8535,51.029778,51.981182,51.983654,52.547081,52.502598,50.075905,51.027306,50.451519,50.790073,51.65004,51.583324,50.832081,51.694523,52.705235,52.700287,52.925171,53.550373,55.253017,55.129459,54.056965,54.341152,54.538845,55.050385,54.603104,53.804901,54.049557,53.792549,54.620396,54.338684,54.074268,55.34692,55.500134,54.108864,54.568497,56.098156,56.110516,55.453178,56.1031,56.859283,58.371643,58.287621,58.151711,57.916946,58.141819,58.421066,59.434246,59.298336,60.094048,60.192894,60.934254,61.544628,60.121231,60.113815,61.472961,63.217617,63.632767,63.541344,63.56852,64.302185,64.478165,64.98877,64.929268,65.551399,65.097816,65.87114,66.20327,66.002502,65.234146,64.941666,64.884651,66.022324,65.50679,66.386681,66.240448,65.474564,64.307152,64.874741,65.826523,67.09803,66.158653,66.545319,67.112915,67.283928,68.19854,69.365952,69.502274,69.336212,69.405609,69.261856,70.39209,70.459007,71.856941,71.829674,72.255997,72.783936,74.444603,73.72084,74.308266,73.958794,75.148521,76.744728,76.918221,78.561531,77.500702,77.168564,78.135223,79.000244,78.464882,78.744957,79.124184,78.896149,76.576187,78.742477,80.390747,80.274246,76.714989,76.504311,79.029999,79.674438,80.606384,79.510727,79.888359,79.406372,81.292099,80.713219,80.733086,79.254822,80.402649,79.577805,77.776558,74.082146,71.572823,72.708221,67.955421,67.915665,74.238663,71.880898,75.215065,72.775307,71.808846,66.129333,70.892075,68.429955,61.672176,69.061012,60.176525,62.822491,61.284599,60.815033,56.954155,55.744217,61.336773,60.998886,64.208832,61.550438,63.306961,63.177769,59.853539,60.852295,59.977768,65.210068,64.454796,66.104485,66.581505,67.888336,71.31691,70.665985,71.22747,70.261017,68.80262,66.675911,68.596413,68.330582,70.303253,70.352943,69.212555,71.485863,72.993935,71.818787,72.83493,73.928101,74.690842,75.463509,77.259674,78.475372,77.578537,76.641853,77.112686,76.656792,78.462914,78.009521,79.526665,78.933754,79.441963,78.903862,79.247643,79.282524,79.205299,80.179359,80.550545,80.99398,80.296448,82.583374,83.07164,85.694878,87.89959,83.679497,84.401947,85.445755,87.710258,87.588196,87.623077,87.122337,89.401787,91.310051,89.698242,90.889038,88.096405,90.126732,90.879066,90.707176,90.707176,93.133614,92.844635,95.006996,95.415558,95.582466,95.141525,96.715965,97.381111,96.182846,95.988533,98.011391,96.658661,96.930199,92.518288,92.289093,94.476364,92.924355,94.705559,95.851517,105.886086,108.554153,109.279099,109.675194,113.501678,110.921135,112.533356,109.186623,112.815369,114.81192,114.709602,114.41011,115.363472,115.508217,118.071297,124.1558,125.640739,124.610016,126.304596,124.794701,124.592552,128.817749,133.948898,131.173691,120.671806,120.751671,112.625694,117.117943,113.29454,111.807106,115.161316,115.341011,111.936882,110.149963,106.655991,109.890411,111.617432,106.935509,108.033615,112.086624,114.762009,113.893501,115.610542,116.58886,112.825348,116.299355,112.965111,114.881805,114.771988,116.768547,124.185753,120.891434,120.981277,120.502106,118.81501,115.780251,117.307617,116.668724,115.550644,114.841873,114.851852,116.399178,111.008476,115.121384,108.672516,108.582664,110.249794,114.752022,118.824997,118.690002,116.32,115.970001,119.489998,119.209999,119.260002,120.300003,119.389999,118.029999,118.639999,117.339996,113.849998,115.169998,116.029999,116.589996,119.050003,122.720001,123.080002,122.940002,122.25,123.75,124.379997,121.779999,123.239998,122.410004,121.779999,127.879997,127.809998,128.699997,126.660004,128.229996,131.880005,130.960007,131.970001,136.690002,134.869995,133.720001,132.690002,129.410004,131.009995,126.599998,130.919998,132.050003,128.979996,128.800003,130.889999]

data1=data1[:3]


data={'PriceLSM':[], 'PriceLSPI':[], 'PriceFQI':[], 'DeltaLSM':[], 'DeltaLSPI':[], 'DeltaFQI':[] }
    
for x3 in (data1):
    print("********")
    S=[]
    PV=[[0.0 for r1 in range(4)] for r2 in range (N)]
    S0=x3
    count=0

    while count<(N//2):
        try:
            a,b=H(S0,V0, r, 100, sigma,kappa,theta, T)
            S.append( a)
            S.append( b)
            count+=1
        except:
            continue 
    
    epsilon=0  
    a=np.array(f( S, True))
    PriceLSM, PriceLSPI, PriceFQI =a[0],a[1],a[2]
    data['PriceLSM'].append(PriceLSM)
    data['PriceLSPI'].append(PriceLSPI)
    data['PriceFQI'].append(PriceFQI)
    epsilon=0.1
    b=np.array(f( S, False))
    #print("Price:")
    #print(a)
    #print(b)
    #print(b-a)
    #print("Delta:")
    c=(1/(epsilon*S0))*(b-a)
    DeltaLSM, DeltaLSPI, DeltaFQI =c[0],c[1],c[2]
    data['DeltaLSM'].append(DeltaLSM)
    data['DeltaLSPI'].append(DeltaLSPI)
    data['DeltaFQI'].append(DeltaFQI)
    

rr=pd.DataFrame(data)
rr.to_excel(r"C:\Users\dais2\Downloads\Data.xlsx")


# In[ ]:




