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
"""
from warnings import warn

from numpy import (absolute, arange, diag,  newaxis,
                 sum)
from traits.api import Int, Property, cached_property, property_depends_on

from acoular.internal import digest
from acoular.spectra import PowerSpectra


class PowerSpectraDR( PowerSpectra ):

    #: Number of iterations for CSM approximation.
    n_iter = Int(50,
        desc="number of iterations for CSM approximation")


    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 'block_size', 
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
