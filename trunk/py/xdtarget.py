import re
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
        mask= (nu.isnan(data.a[:,ii]))*(nu.isinf(data.a[:,ii]))
        mask= nu.array([not m for m in mask])
        datameans= nu.array([nu.mean(data.a[mask,ii]) for ii in range(data.da)])
        datastddevs= nu.array([nu.std(data.a[mask,ii]) for ii in range(data.da)])
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
    ycovar= data.acov
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
    
    Initialize with filename (atag, acovtag) or arrays a and acov

    a = [ndata,da]

    acov= [ndata,da(,da)] (if diagonal 2D)

    """
    def __init__(self,**kwargs):
        if kwargs.has_key('filename'):
            tmp_ext= re.split('\.',kwargs['filename'])[-1]
            if tmp_ext == 'gz':
                tmp_ext= re.split('\.',kwargs['filename'])[-2]+'.'+tmp_ext
            if tmp_ext == 'fit' or tmp_ext == 'fits' or \
                    tmp_ext == 'fit.gz' or tmp_ext == 'fits.gz':
                if kwargs.has_key('atag'):
                    atag= kwargs['atag']
                else:
                    atag= 'a'
                if kwargs.has_key('acovtag'):
                    acovtag= kwargs['acovtag']
                else:
                    acovtag= 'acov'
                import pyfits
                hdulist= pyfits.open(kwargs['filename'])
                tbdata= hdulist[1].data
                self.a= nu.array(tbdata.field(atag)).astype('float64')
                self.acov= nu.array(tbdata.field(acovtag)).astype('float64')
        elif kwargs.has_key('a'):
            pass
        self.da= self.a.shape[1]
