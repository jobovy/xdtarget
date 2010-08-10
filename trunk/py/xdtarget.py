import re
import numpy as nu
from scipy import stats, linalg
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
        self.ngauss= len(self.amp)

    def __call__(self,*args):
        """
        NAME:
        
           __call__

        PURPOSE:

           evaluate the log-probability of the input under the density model

        INPUT:

           Either:

              1) xddata object

        OUTPUT:

           array of log-probabilities

        HISTORY:
        
           2010-08-09 - Written - Bovy (NYU)
        
        """
        if isinstance(args[0],xddata):
            return self._eval(args[0].a,args[0].acov)
        else:
            return self._eval(args[0],args[1])

    def _eval(self,a,acov):
        ndata= a.shape[0]
        da= a.shape[1]
        if len(a) == len(acov):
            diagcovar= True
        twopiterm= 0.5*da*nu.log(2.*nu.pi)
        out= nu.zeros(ndata)
        loglike= nu.zeros(self.ngauss)
        for ii in range(ndata):
            for kk in range(self.ngauss):
                if diagcovar:
                    tinv= linalg.inv(self.covar[kk,:,:]+nu.diag(acov[ii,:]))
                else:
                    tinv= linalg.inv(self.covar[kk,:,:]+acov[ii,:,:])
                delta= a[ii,:]-self.mean[kk,:]
                loglike[kk]= nu.log(self.amp[kk])+0.5*nu.log(linalg.det(tinv))\
                    -0.5*nu.dot(delta,nu.dot(tinv,delta))
            out[ii]= _logsum(loglike)
        return out

class xddata:
    """Class that holds the training data
    
    Initialize with filename (atag, acovtag) or arrays a and acov

    a = [ndata,da]

    acov= [ndata,da(,da)] (if diagonal 2D)

    weight=, useweights=, wtag

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
                if kwargs.has_key('wtag'):
                    wtag= kwargs['wtag']
                else:
                    wtag= 'weight'
                import pyfits
                hdulist= pyfits.open(kwargs['filename'])
                tbdata= hdulist[1].data
                self.a= nu.array(tbdata.field(atag)).astype('float64')
                self.acov= nu.array(tbdata.field(acovtag)).astype('float64')
                if kwargs.has_key('useweights') and kwargs['useweights']:
                    weight= nu.array(tbdata.field(wtag)).astype('float64')
        elif kwargs.has_key('a'):
            self.a= kwargs['a']
            self.acov= kwargs['acov']
            if kwargs.has_key('weight'):
                self.weight= kwargs['weight']
        self.da= self.a.shape[1]


def _logsum(array):
    """
    NAME:
       _logsum
    PURPOSE:
       calculate the logarithm of the sum of an array of numbers,
       given as a set of logs
    INPUT:
       array - logarithms of the numbers to be summed
    OUTPUT:
       logarithm of the sum of the exp of the numbers in array
    REVISION HISTORY:
       2009-09-29 -Written - Bovy (NYU)
    """
    #For now Press' log-sum-exp because I am too lazy to implement 
    #my own algorithm for this
    array= nu.array(array)
    c= nu.amax(array)
    return nu.log(nu.nansum(nu.exp(nu.add(array,-c))))+c

