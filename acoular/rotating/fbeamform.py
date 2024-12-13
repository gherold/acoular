# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements beamformers in the frequency domain.

.. autosummary::
    :toctree: generated/

    SteeringVectorInduct
    SteeringVectorModeTransformer


"""

# imports from other packages

from warnings import warn

from numpy import (append, arange, arctan2, array, exp, fft, newaxis, nonzero,
                   pi, sign, sqrt, zeros)
from scipy.special import jn, jnp_zeros
from traits.api import (Bool, CArray, Delegate, Float, Int, Property, Trait,
                        cached_property)

from acoular.fbeamform import SteeringVector
from acoular.internal import digest


class SteeringVectorInduct( SteeringVector ):
    """
    Class for implementing a steering vector for sound 
    propagation in a circular duct (without hub).
    """
    
    #: Return not value at array center?
    ret_source_strength = Bool(False)
    
    #: Radius of the duct
    ra = Float(1.0,
               desc = "duct radius")
    
    #: The Mach number in z-direction, defaults to 0.
    ma = Float(0.0, 
        desc="flow mach number in z-direction")    
    
    #: Revolutions per minute of the array; negative values for
    #: clockwise rotation; defaults to 0."
    rpm = Float(0.0,
        desc="revolutions per minute of the virtual array; negative values for clockwise rotation")
    
    #: Cut-on ratio above which to take into account mode;
    #: only within limits of mmax, nmax
    threshold = Float(0.8,
        desc="cut-on ratio threshold")
        
    mmax = Int(150,
               desc="max. radial mode order (+-)")
    
    nmax = Int(30,
               desc="max. azimuthal mode order")
               
    #: zeros of the Bessel function
    sigma = Property(
            depends_on=['mmax','nmax'],)
    
    # internal identifier
    digest = Property(depends_on=['steer_type', 'env.digest', 
                                  'grid.digest', 'mics.digest', '_ref', 
                                  'ra', 'ma', 'rpm', 
                                  'threshold', 'mmax', 'nmax'])
    
    # internal identifier, use for inverse methods, excluding steering vector type
    inv_digest = Property(depends_on=['env.digest', 
                                  'grid.digest', 'mics.digest', '_ref', 
                                  'ra', 'ma', 'rpm', 
                                  'threshold', 'mmax', 'nmax'])
    
 
    @cached_property
    def _get_digest( self ):
        return digest( self ) 


    def _polar(self, xpos):
        """
        convert (x,y,-z) to (r, phi, z)
        """
        return array([sqrt(xpos[0]**2+xpos[1]**2),
                      arctan2(xpos[1],xpos[0]), 
                      -xpos[2]])    
    
    @cached_property
    def _get_sigma( self ):
        """
        Returns array of shape(mmax+1,nmax+1)
        """
        mmax = abs(self.mmax)
        nmax = abs(self.nmax)
        if mmax+nmax > 0:
            sigma = zeros((mmax+1,nmax+1))
            for m in range(mmax+1):
                if m == 0:
                    if nmax > 0:
                        s = jnp_zeros(m,nmax)
                        sigma[m,1:] = s
                else:
                    s = jnp_zeros(m,nmax+1)
                    sigma[m,:] = s
            
            #sigma_full = append(sigma[1:][::-1],sigma, axis = 0) #to get m-index subtract mmax    
            return sigma#_full
        else:
            return array([[0.0]])
    
    
    def cuton_ratio(self, c, freq):
        """
        Returns array of shape (2*mmax+1, nmax+1). For correct indexing substract mmax from first index.
        """
        sigma= append(self.sigma[1:][::-1],self.sigma, axis = 0) #to get m-index subtract mmax
        Omega = self.rpm/60.*2*pi # rotation in rad/s
        mshape = sigma.shape[0]
        m = (arange(mshape) - self.mmax)[:,newaxis]
        
        divisor = c/self.ra*sqrt(1-self.ma*self.ma)*sigma + Omega*m
        divisor[int(mshape/2.),0] = 1
        cor = 2*pi * freq / divisor
        cor[int(mshape/2.),0] = 999.
        return cor    
    
    def transfer( self, freq, ind=None ):
        """
        Induct transfer function [ngrid x nmic]
        """
        if ind is not None: raise NotImplementedError()
        gp = self._polar(self.grid.gpos)
        # append reference point to the mic array positions
        mp = append(self._polar(self.mics.mpos),array([[0.],[0.],[0.]]), axis=1)
        c = self.env.c
        Ma = -self.ma
        
        Omega = self.rpm/60.*2*pi # rotation in rad/s        
        
        dphi = mp[1][newaxis,:]-gp[1][:,newaxis]
        dz = mp[2][newaxis,:]-gp[2][:,newaxis]
        
        #dphi0 = -gp[1][:,newaxis]
        #dz0 = -gp[2][:,newaxis]
        
        pm = sign(dz) # are we downstream or upstream from source
        chi2 = 1 - Ma*Ma # helper constant
        #transfer  = pi [mics] / p0 [array center]
        mn = nonzero(self.cuton_ratio(c,freq) >= self.threshold)
        
        ra = self.ra        
        p_i = zeros((gp.shape[1],mp.shape[1]), dtype = 'complex128')
        #p_0 = zeros((gpos.shape[1],1), dtype = 'complex128')
        for mm,n in zip(*mn):
            m_pm = mm - self.mmax # pos/neg m
            m = abs(m_pm) # only abs mode value
            sigmn = self.sigma[m,n]
            
            k = (2*pi*freq - m_pm*Omega) / c # evtl mode <-> m ?
            amn = sqrt(1 - chi2 * (sigmn / k / ra)**2 + 0j )
            kmn = k * (pm * amn - Ma) / chi2
            
            Fmn = 0.5
            if m!=0 or n!=0:
                Fmn *= (1-(m/sigmn)**2) * jn(m, sigmn)**2 + \
                       ( m/sigmn * jn (m, 0) )**2
            #else:
            #kmn0 = k * (sign(dz0) * amn - Ma) / chi2
            #p_0 += jn(m, sigmn * gp[0]/ra)[:,newaxis] / \
            #       (amn * Fmn) * \
            #       exp(1j * m   * dphi0) * \
            #       exp(1j * kmn0 * dz0)
            
            p_i += jn(m, sigmn * mp[0]/ra)[newaxis,:] * \
                   jn(m, sigmn * gp[0]/ra)[:,newaxis] / \
                   (amn * Fmn) * \
                   exp(1j * m_pm * dphi) * \
                   exp(1j * kmn  * dz)
        if self.ret_source_strength:
            return p_i[:,:-1]
        else:
            # last column contains reference point    
            return p_i[:,:-1]/p_i[:,-1][:,newaxis]
        

class SteeringVectorModeTransformer( SteeringVector ):
    
    steer = Trait(SteeringVector)
    
    #: Channels to be used and their order, so that neighboring mics are 
    #: channel neighbors as well (last is neigbor to first). 
    #: If left emtpy, mics are just used in their current order. 
    channel_order = CArray(dtype=int, value=array([]), 
        desc="list of mic channels in correct order")

    # following lines are currently only workaround as mics trait is needed 
    # to check number of channels. this can easily result in buggy behaviour!
    mics = Delegate('steer')
    grid = Delegate('steer')
    env = Delegate('steer')
    ref = Delegate('steer')
    steer_type = Delegate('steer')
    # internal identifier
    digest = Property( 
        depends_on = ['steer.digest', 'channel_order'], 
        )
    
    # internal identifier, use for inverse methods, excluding steering vector type
    inv_digest = Property( 
        depends_on = ['steer.inv_digest', 'channel_order'])
    
    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @cached_property
    def _get_inv_digest( self ):
        return digest( self )
    
    # dummy attributes to overwrite SteeringVector attributes
    r0 = Property(desc="array center to grid distances")
    rm = Property(desc="all array mics to grid distances")
    
    def _get_r0 ( self ):
        return None

    def _get_rm ( self ):
        return None

    def transfer ( self, f, ind=None ):
        """
        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmodes) containing the transfer matrix for the given frequency
        """
        temp = self.steer.transfer(f)
        return fft.fft(temp[:,self.channel_order], None, 1, norm="ortho") 
    
    def steer_vector(self, f, ind=None):
        """
        Calculates the steering vectors based on the modal transfer function
        
        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the steering vectors of the gridpoints addressed by 
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed
        
        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmodes) containing the steering vectors for the given frequency
        """
        func = self.steer._steer_funcs_freq[self.steer_type]        
        temp = func(self.steer.transfer(f, ind))
       
        return fft.fft(temp[:,self.channel_order], None, 1, norm="ortho")