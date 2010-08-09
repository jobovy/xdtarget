import numpy as nu
from scipy import stats
from extreme_deconvolution import extreme_deconvolution

def train(data,ngauss=2,init_xdtarget=None):
    """
    NAME:
    
       train
    
    PURPOSE:

       xd train from a data set

    INPUT:

       data - trainData instance

       ngauss - number of Gaussians to use

       init_xdtarget (optional) - initial xdtarget instance (amp, mean, covar)

    OUTPUT:

       xdtarget instance

    HISTORY:
       2010-08-09 - Written - Bovy (NYU)
    """
    #Initialize
    if init_xdtarget is None:
        initamp= nu.array([1./ngauss for ii in range(ngauss)])
        datameans= nu.array([nu.mean(data.a[(not numpy.isnan(data.a[:,ii])*(not numnpy.isinf(data.a[:,ii])),ii]]) for ii in range(data.da)])
        datastddevs= nu.array([nu.stddev(data.a[(not numpy.isnan(data.a[:,ii])*(not numnpy.isinf(data.a[:,ii])),ii]]) for ii in range(data.da)])
        initmean= nu.zeros((ngauss,data.da))
        initcovar= nu.zeros((ngauss,data.da,data.da))
        for kk in range(ngauss):
            for ii in range(data.da):
                initmean[kk,ii]= datameans[ii]+(2.*stats.uniform.rvs()-1.)*\
                    datastddevs[ii]
                initcovar[kk,ii,ii]= datastddevs[ii]**2.
        init_xdtarget= xdtarget(amp=initamp,mean=initmean,covar=initcovar)
        
    #Run XD
    return xd(data,init_xdtarget)

def xd(data,init_xdtarget):
    initamp= init_xdtarget.amp
    initmean= init_xdtarget.mean
    initcovar= init_xdtarget.covar

    ydata= data.a
    ycovar= ata.acov
    if hasattr(data,'weight'):
        weight= data.weight
    else:
        weight= None
    if hasattr(data,'logweight'):
        logweight= data.logweight
    else:
        logweight= False

    extreme_deconvolution(ydata,ycovar,initamp,initmean,initcovar,
                          weight=weight,logweight=logweight)
                        
    out_xdtarget= xdtarget(amp=initamp,mean=initmean,covar=initcovar)

    return out_xdtarget
  
class xdtarget:
    """class that holds the XD solution and can be used to calculate target
    probabilities"""
    def __init__(self,amp=None,mean=None,covar=None):
        self.amp= amp
        self.mean= mean
        self.covar= covar

class trainData:
    """Class that holds the training data
    
    Initialize with filename or arrays a and acov

    a = [ndata,da]

    acov= [ndata,da(,da)] (if diagonal 2D)

    """
    def __init__(self,**kwargs):
        if kwargs.has_key('filename'):
            pass
        elif kwargs.has_key('a'):
            pass
    self.da= self.a.shape[1]
