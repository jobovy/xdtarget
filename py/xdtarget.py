import re
from copy import deepcopy
import math as m
import numpy as nu
from scipy import stats, linalg
from matplotlib import pyplot, patches
from extreme_deconvolution import extreme_deconvolution
try:
    from galpy.util import bovy_plot #Latest
except ImportError:
    import bovy_plot
_SQRTTWOPI= -0.5*nu.log(2.*nu.pi)
def train(data,ngauss=2,init_xdtarget=None):
    """
    NAME:
    
       train
    
    PURPOSE:

       xd train from a data set

    INPUT:

       data - xddata instance

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
        datameans= nu.zeros(data.da)
        datastddevs= nu.zeros(data.da)
        for ii in range(data.da):
            mask= (nu.isnan(data.a[:,ii]))*(nu.isinf(data.a[:,ii]))
            mask= nu.array([not m for m in mask])
            datameans[ii]= nu.mean(data.a[mask,ii])
            datastddevs[ii]= nu.std(data.a[mask,ii])
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
    def __init__(self,*args,**kwargs):
        if len(args) > 0: #load from file
            tmp_ext= re.split('\.',args[0])[-1]
            if tmp_ext == 'sav': #pickle
                import pickle
                file= open(args[0],'rb')
                tmp_self= pickle.load(file)
                file.close()
                self.amp= tmp_self.amp
                self.mean= tmp_self.mean
                self.covar= tmp_self.covar
        else:
            self.amp= kwargs['amp']
            self.mean= kwargs['mean']
            self.covar= kwargs['covar']
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

              2) a, acov

        OUTPUT:

           array of log-probabilities

        HISTORY:
        
           2010-08-09 - Written - Bovy (NYU)
        
        """
        if isinstance(args[0],xddata):
            return self._eval(args[0].a,args[0].acov)
        else:
            return self._eval(args[0],args[1])

    def sample(self,nsample=1):
        """
        NAME:

           sample

        PURPOSE:

           sample from the density

        INPUT:

           nsample - number of samples

        OUTPUT:

           array [ndata,da] of samples

        HISTORY:
        
           2010-08-09 - Written - Bovy (NYU)

        """
        #First assign the samples to Gaussians
        cumamp= nu.cumsum(self.amp)
        comp= nu.zeros(nsample).astype('int')
        for ii in range(nsample):
            gauss= stats.uniform.rvs()
            jj= 0
            while (gauss > cumamp[jj]):
                jj+= 1
            comp[ii]= jj
        out= []
        for c in set(list(comp)):
            thiscomp= comp[comp == c]
            thisn= len(thiscomp)
            out.extend(_sample_normal(self.mean[c,:],self.covar[c,:,:],
                                      nsamples=thisn))
        self.samples= nu.array(out).reshape((nsample,self.mean.shape[1]))
        return self.samples

    def scatterplot(self,d1,d2,*args,**kwargs):
        """
        NAME:
        
           scatterplot

        PURPOSE:

           make a scatterplot of the samples

        INPUT:

           d1, d2 - x and y dimension to plot

           hoggscatter - if True, hogg_scatterplot

           +bovy_plot.plot or bovy_plot.scatterplot args and kwargs

        OUTPUT:

           plot to output device

        HISTORY:

           2010-08-09 - Written - Bovy (NYU)

        """
        if kwargs.has_key('hoggscatter'):
            hoggscatter= kwargs['hoggscatter']
            kwargs.pop('hoggscatter')
        else:
            hoggscatter= False
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= str(d1)
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= str(d2)
        if hoggscatter:
            bovy_plot.scatterplot(self.samples[:,d1],self.samples[:,d2],
                                  *args,**kwargs)
        else:
            bovy_plot.bovy_plot(self.samples[:,d1],self.samples[:,d2],
                                *args,**kwargs)

    def plot(self,d1,d2,*args,**kwargs):
        """
        NAME:
        
           plot

        PURPOSE:

           make a plot of the solution

        INPUT:

           d1, d2 - x and y dimension to plot

           dens - make density plot

           xrange, yrange

           npix, npix_x, npix_y

        OUTPUT:

           plot to output device

        HISTORY:

           2010-08-16 - Written - Bovy (NYU)

        """
        if kwargs.has_key('dens') and kwargs['dens']:
            dens= True
            kwargs.pop('dens')
        else:
            dens= False
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= str(d1)
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= str(d2)

        if not dens:
            #Create the ellipses for the Gaussians
            x= nu.zeros(self.ngauss)
            y= nu.zeros(self.ngauss)
            ellipses=[]
            ymin, ymax= self.mean[0,d1], self.mean[0,d1]
            xmin, xmax= self.mean[0,d2], self.mean[0,d2]
            for ii in range(self.ngauss):
                x[ii]= self.mean[ii,d1]
                y[ii]= self.mean[ii,d2]
                #Calculate the eigenvalues and the rotation angle
                ycovar= nu.zeros((2,2))
                ycovar[0,0]= self.covar[ii,d1,d1]
                ycovar[1,1]= self.covar[ii,d2,d2]
                ycovar[0,1]= self.covar[ii,d1,d2]
                ycovar[1,0]= ycovar[0,1]
                eigs= linalg.eig(ycovar)
                angle= m.atan(-eigs[1][0,1]/eigs[1][1,1])/m.pi*180.
                thisellipse= patches.Ellipse(nu.array([x[ii],y[ii]]),
                                             2*nu.sqrt(eigs[0][0]),
                                             2*nu.sqrt(eigs[0][1]),angle)
                ellipses.append(thisellipse)
                if (x[ii]+m.sqrt(ycovar[0,0])) > xmax:
                    xmax= (x[ii]+m.sqrt(ycovar[0,0]))
                if (x[ii]-m.sqrt(ycovar[0,0])) < xmin:
                    xmin= (x[ii]-m.sqrt(ycovar[0,0]))
                if (y[ii]+m.sqrt(ycovar[1,1])) > ymax:
                    ymax= (y[ii]+m.sqrt(ycovar[1,1]))
                if (y[ii]-m.sqrt(ycovar[1,1])) < ymin:
                    ymin= (y[ii]-m.sqrt(ycovar[1,1]))

            fig= pyplot.figure()
            ax= fig.add_subplot(111)
            for e in ellipses:
                ax.add_artist(e)
                e.set_facecolor('none')
            ax.set_xlabel(kwargs['xlabel'])
            ax.set_ylabel(kwargs['ylabel'])
            if not kwargs.has_key('xrange'):
                ax.set_xlim((xmin,xmax))
            else:
                ax.set_xlim((kwargs['xrange'][0],kwargs['xrange'][1]))
            if not kwargs.has_key('yrange'):
                ax.set_ylim((ymin,ymax))
            else:
                ax.set_ylim((kwargs['yrange'][0],kwargs['yrange'][1]))

        else:
            #Create the ellipses for the Gaussians, to determine range
            x= nu.zeros(self.ngauss)
            y= nu.zeros(self.ngauss)
            ellipses=[]
            ymin, ymax= self.mean[0,d1], self.mean[0,d1]
            xmin, xmax= self.mean[0,d2], self.mean[0,d2]
            for ii in range(self.ngauss):
                x[ii]= self.mean[ii,d1]
                y[ii]= self.mean[ii,d2]
                #Calculate the eigenvalues and the rotation angle
                ycovar= nu.zeros((2,2))
                ycovar[0,0]= self.covar[ii,d1,d1]
                ycovar[1,1]= self.covar[ii,d2,d2]
                ycovar[0,1]= self.covar[ii,d1,d2]
                ycovar[1,0]= ycovar[0,1]
                if (x[ii]+3.*m.sqrt(ycovar[0,0])) > xmax:
                    xmax= (x[ii]+3.*m.sqrt(ycovar[0,0]))
                if (x[ii]-3.*m.sqrt(ycovar[0,0])) < xmin:
                    xmin= (x[ii]-3.*m.sqrt(ycovar[0,0]))
                if (y[ii]+3.*m.sqrt(ycovar[1,1])) > ymax:
                    ymax= (y[ii]+3.*m.sqrt(ycovar[1,1]))
                if (y[ii]-3.*m.sqrt(ycovar[1,1])) < ymin:
                    ymin= (y[ii]-3.*m.sqrt(ycovar[1,1]))

            #Get range
            if not kwargs.has_key('xrange'):
                kwargs['xrange']= [xmin,xmax]
            if not kwargs.has_key('yrange'):
                kwargs['yrange']= [ymin,ymax]
            xrange= kwargs['xrange']
            yrange= kwargs['yrange']
            if not kwargs.has_key('npix') and not kwargs.has_key('npix_x'):
                npix_x= 101
            elif kwargs.has_key('npix_x'):
                npix_x= kwargs['npix_x']
                kwargs.pop('npix_x')
            elif kwargs.has_key('npix'):
                npix_x= kwargs['npix']
            if not kwargs.has_key('npix') and not kwargs.has_key('npix_y'):
                npix_y= 101
            elif kwargs.has_key('npix_y'):
                npix_y= kwargs['npix_y']
                kwargs.pop('npix_y')
            elif kwargs.has_key('npix'):
                npix_y= kwargs['npix']
                kwargs.pop('npix')
            if kwargs.has_key('npix'):
                kwargs.pop('npix')

            #compute density
            dens= nu.zeros((npix_x,npix_y))
            xs= nu.linspace(xrange[0],xrange[1],npix_x)
            ys= nu.linspace(yrange[0],yrange[1],npix_y)
            means= nu.zeros((len(self.amp),2))
            covars= nu.zeros((len(self.amp),2,2))
            for kk in range(len(self.amp)):
                means[kk,0]= self.mean[kk,d1]
                means[kk,1]= self.mean[kk,d2]
                covars[kk,0,0]= self.covar[kk,d1,d1]
                covars[kk,1,0]= self.covar[kk,d1,d2]
                covars[kk,0,1]= self.covar[kk,d2,d1]
                covars[kk,1,1]= self.covar[kk,d2,d2]
            thisxd= xdtarget(amp=self.amp,
                             mean= means,
                             covar=covars)
            for ii in range(npix_x):
                for jj in range(npix_y):
                    dens[ii,jj]= thisxd(nu.array([xs[ii],ys[jj]]).reshape((1,2)),
                                        nu.zeros((2,2)))
            dens= nu.exp(dens)
            bovy_plot.bovy_dens2d(dens.T,origin='lower',cmap='gist_yarg',
                                  **kwargs)


    def save(self,filename):
        """
        NAME:
        
           save

        PURPOSE:

           save the xdtarget object to a file

        INPUT:

           filename - name of the file to save the object to

        OUTPUT:

           none

        HISTORY:
        
           2010-08-10 - Written - Bovy (NYU)
        
        """
        tmp_ext= re.split('\.',filename)[-1]
        if tmp_ext == 'sav': #pickle
            import pickle
            file= open(filename,'wb')
            pickle.dump(self,file)
            file.close()

    def _eval(self,a,acov):
        ndata= a.shape[0]
        da= a.shape[1]
        if len(a.shape) == len(acov.shape):
            diagcovar= True
        else:
            diagcovar= False
        out= nu.zeros(ndata)
        loglike= nu.zeros(self.ngauss)
        for ii in range(ndata):
            for kk in range(self.ngauss):
                if self.amp[kk] == 0.:
                    loglike[kk]= nu.finfo(nu.dtype(nu.float64)).min
                    continue
                if diagcovar:
                    tinv= linalg.inv(self.covar[kk,:,:]+nu.diag(acov[ii,:]))
                else:
                    tinv= linalg.inv(self.covar[kk,:,:]+acov[ii,:,:])
                delta= a[ii,:]-self.mean[kk,:]
                loglike[kk]= nu.log(self.amp[kk])+0.5*nu.log(linalg.det(tinv))\
                             -0.5*nu.dot(delta,nu.dot(tinv,delta))+\
                             da*_SQRTTWOPI
            out[ii]= _logsum(loglike)
        return out

class xddata:
    """Class that holds the training data
    
    Initialize with filename (atag, acovtag) or arrays a and acov

    a = [ndata,da]

    acov= [ndata,da(,da)] (if diagonal 2D)

    weight=, useweights=, wtag

    alltags=True

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
                if acovtag.lower() in [name.lower() for name in hdulist[1].columns.names]:
                    self.acov= nu.array(tbdata.field(acovtag)).astype('float64')
                    if self.acov.shape[1] != self.a.shape[1]:
                        self.acov= nu.reshape(self.acov,(self.a.shape[0],self.a.shape[1],self.a.shape[1]))
                else:
                    self.acov= nu.zeros(self.a.shape)
                if kwargs.has_key('useweights') and kwargs['useweights']:
                    self.weight= nu.array(tbdata.field(wtag)).astype('float64')
                if kwargs.has_key('alltags') and kwargs['alltags']:
                    tags= hdulist[1].columns.names
                    tmp_tags= deepcopy(tags)
                    popped= 0
                    for ii in range(len(tags)):
                        if tags[ii].lower() == atag.lower() or \
                           tags[ii].lower() == acovtag.lower():
                            tmp_tags.pop(ii-popped)
                            popped+= 1
                        if kwargs.has_key('useweights') and kwargs['useweights'] and tags[ii].lower() == wtag.lower():
                            tmp_tags.pop(ii)
                    tags= tmp_tags
                    for tag in tags:
                        self.__dict__[tag.lower()]= tbdata.field(tag)
                    self._alltags= True
                    self._tags= [tag.lower() for tag in tags]
                else:
                    self._alltags= False
        elif kwargs.has_key('a'):
            self.a= kwargs['a']
            if kwargs.has_key('acov'):
                self.acov= kwargs['acov']
            else:
                self.acov= nu.zeros(self.a.shape)
            if kwargs.has_key('weight'):
                self.weight= kwargs['weight']
            self._alltags= False
            self._tags= None
        self.da= self.a.shape[1]

    def __getitem__(self,key):
        if not isinstance(key,slice):
            nkey= 1
        else:
            nkey= len(self.a[key,0])
        if len(self.acov.shape) == 2:
            acov= self.acov[key,:]
            dacov= (nkey,self.da)
        else:
            acov= self.acov[key,:,:]
            dacov= (nkey,self.da,self.da)
        if hasattr(self,'weight'):
            out= xddata(a=nu.reshape(self.a[key,:],(nkey,self.da)),
                        acov=nu.reshape(acov,dacov),
                        weight=self.weight[key])
        else:
            out= xddata(a=nu.reshape(self.a[key,:],(nkey,self.da)),
                        acov=nu.reshape(acov,dacov))
        #Also transfer tags
        if self._alltags:
            for tag in self._tags:
                thisshape= self.__dict__[tag].shape
                thistag= nu.reshape(self.__dict__[tag],(thisshape[0],nu.prod(thisshape)/thisshape[0]))
                tmptag= thistag[key,:]
                outshape=[nkey]
                nshape= len(list(thisshape))
                thisshape= [thisshape[ii] for ii in range(nshape)
                            if ii != 0]
                outshape.extend([s for s in thisshape])
                outshape= tuple(outshape)
                out.__dict__[tag]= nu.reshape(tmptag,outshape)
        out._alltags= self._alltags
        out._tags= self._tags
        return out

    def scatterplot(self,d1,d2,*args,**kwargs):
        """
        NAME:
        
           scatterplot

        PURPOSE:

           make a scatterplot of the data

        INPUT:

           d1, d2 - x and y dimension to plot

           hoggscatter - if True, hogg_scatterplot

           +bovy_plot.plot or bovy_plot.scatterplot args and kwargs

        OUTPUT:

           plot to output device

        HISTORY:

           2010-08-09 - Written - Bovy (NYU)

        """
        if kwargs.has_key('hoggscatter'):
            hoggscatter= kwargs['hoggscatter']
            kwargs.pop('hoggscatter')
        else:
            hoggscatter= False
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= str(d1)
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= str(d2)
        if hoggscatter:
            if hasattr(self,'weight'):
                kwargs['weights']= self.weight
            bovy_plot.scatterplot(self.a[:,d1],self.a[:,d2],
                                  *args,**kwargs)
        else:
            bovy_plot.bovy_plot(self.a[:,d1],self.a[:,d2],
                                *args,**kwargs)

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


def _sample_normal(mean,covar,nsamples=1):
    """sample_normal: Sample a d-dimensional Gaussian distribution with
    mean and covar.

    Input:
     
       mean     - the mean of the Gaussian

       covar    - the covariance of the Gaussian

       nsamples - (optional) the number of samples desired

    Output:

       samples; if nsamples != 1 then a list is returned

    History:

       2009-05-20 - Written - Bovy (NYU)

    """
    p= covar.shape[0]
    #First lower Cholesky of covar
    L= linalg.cholesky(covar,lower=True)
    if nsamples > 1:
        out= []
    for kk in range(nsamples):
        #Generate a vector in which the elements ~N(0,1)
        y= nu.zeros(p)
        for ii in range(p):
            y[ii]= stats.norm.rvs()
        #Form the sample as Ly+mean
        thissample= nu.dot(L,y)+mean
        if nsamples == 1:
            return thissample
        else:
            out.append(thissample)
    return out
