# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, E1103, C0103, R0901, R0902, R0903, R0904
#pylint: disable-msg=W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2021, Acoular Development Team.
#------------------------------------------------------------------------------
"""Estimation of power spectra and related tools

.. autosummary::
    :toctree: generated/

    PowerSpectraComplex
    PowerSpectraDR
    synthetic
"""
from warnings import warn

from numpy import array, ones, hanning, hamming, bartlett, blackman, \
dot, newaxis, zeros, empty, fft, linalg, isrealobj, absolute, diag, \
searchsorted, isscalar, fill_diagonal, arange, zeros_like, sum
from traits.api import HasPrivateTraits, Int, Property, Instance, Trait, \
Range, Bool, cached_property, property_depends_on, Delegate, Float


from acoular.internal import digest
from acoular.spectra import PowerSpectra


class PowerSpectraDR( PowerSpectra ):

    #: Number of iterations for CSM approximation.
    n_iter = Int(50,
        desc="number of iterations for CSM approximation")


    # internal identifier
    digest = Property( 
        depends_on = ['_source.digest', 'calib.digest', 'block_size', 
            'window', 'overlap', 'precision', 'n_iter'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @property_depends_on('digest')
    def _get_csm(self):
        # my csm recon algo
        #csm = self.calc_csm()
        fullcsm = self.calc_csm()
        csm = fullcsm[self.ind_low:self.ind_high]
        nc = csm.shape[-1]
        idiag = arange(nc)
        # calculate amplitude of entries
        for ind in range(csm.shape[0]):
            ac = absolute(csm[ind])
            # get csm diagonal
            dia = diag(ac)
            # get position and value of maximum in diagonal (position maybe not needed)
            # indmaxdia = argmax(dia)
            # max_dia = dia[indmaxdia]
            max_dia = max(dia)
            # get maximum from off-diagonal values
            max_off = (ac-diag(dia)).max()
            # calculate 1st diag approximation with correction factor 
            new_dia = dia * (max_off / max_dia)
            # loop to approximate further
            for i in range(self.n_iter):
                # from diag new theoretical mode amplitudes
                mode_amps = new_dia**0.5
                # calculate csm amp estimate
                new_ac = mode_amps[:,newaxis] * mode_amps[newaxis,:]
                # calculate difference to actual csm
                diff = ac-new_ac # probably mostly positive vals
                # set diag of diff to 0 (this is unwanted info)
                diff[idiag, idiag] = 0 
                # correct diagonal by average offset
                new_dia += sum(diff,axis=0)/(nc-1)
                # set negative values to zero
                new_dia[new_dia<0]=0
            # set approximated new diag into csm
            csm[ind, idiag, idiag] = new_dia
        fullcsm[self.ind_low:self.ind_high] = csm
        return fullcsm


def synthetic (data, freqs, f, num=3):
    """
    Returns synthesized frequency band values of spectral data.
    
    If used with :meth:`Beamformer.result()<acoular.fbeamform.BeamformerBase.result>` 
    and only one frequency band, the output is identical to the result of the intrinsic 
    :meth:`Beamformer.synthetic<acoular.fbeamform.BeamformerBase.synthetic>` method.
    It can, however, also be used with the 
    :meth:`Beamformer.integrate<acoular.fbeamform.BeamformerBase.integrate>`
    output and more frequency bands.
    
    Parameters
    ----------
    data : array of floats
        The spectral data (sound pressures in Pa) in an array with one value 
        per frequency line.
        The number of entries must be identical to the number of
        grid points.
    freq : array of floats
        The frequencies that correspon to the input *data* (as yielded by
        the :meth:`PowerSpectra.fftfreq<acoular.spectra.PowerSpectra.fftfreq>`
        method).
    f : float or list of floats
        Band center frequency/frequencies for which to return the results.
    num : integer
        Controls the width of the frequency bands considered; defaults to
        3 (third-octave band).
        
        ===  =====================
        num  frequency band width
        ===  =====================
        0    single frequency line
        1    octave band
        3    third-octave band
        n    1/n-octave band
        ===  =====================

    Returns
    -------
    array of floats
        Synthesized frequency band values of the beamforming result at 
        each grid point (the sum of all values that are contained in the band).
        Note that the frequency resolution and therefore the bandwidth 
        represented by a single frequency line depends on 
        the :attr:`sampling frequency<acoular.tprocess.SamplesGenerator.sample_freq>` 
        and used :attr:`FFT block size<acoular.spectra.PowerSpectra.block_size>`.
    """
    if isscalar(f):
        f = (f,)
    if num == 0:
        # single frequency lines
        res = list()
        for i in f:
            ind = searchsorted(freqs, i)
            if ind >= len(freqs):
                warn('Queried frequency (%g Hz) not in resolved '
                     'frequency range. Returning zeros.' % i, 
                     Warning, stacklevel = 2)
                h = zeros_like(data[0])
            else:
                if freqs[ind] != i:
                    warn('Queried frequency (%g Hz) not in set of '
                         'discrete FFT sample frequencies. '
                         'Using frequency %g Hz instead.' % (i,freqs[ind]), 
                         Warning, stacklevel = 2)
                h = data[ind]
            res += [h]      
    else:
        # fractional octave bands
        res = list()
        for i in f:
            f1 = i*2.**(-0.5/num)
            f2 = i*2.**(+0.5/num)
            ind1 = searchsorted(freqs, f1)
            ind2 = searchsorted(freqs, f2)
            if ind1 == ind2:
                warn('Queried frequency band (%g to %g Hz) does not '
                     'include any discrete FFT sample frequencies. '
                     'Returning zeros.' % (f1,f2), 
                     Warning, stacklevel = 2)
                h = zeros_like(data[0])
            else:
                h = sum(data[ind1:ind2], 0)
            res += [h]
    return array(res)

