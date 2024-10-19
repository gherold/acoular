# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, acoular Development Team.
#------------------------------------------------------------------------------
"""
Implements support for circular microphone arrays 

.. autosummary::
    :toctree: generated/
    
    MicRing
    MicGeomCirc
"""


from os import path

from numpy import argsort, array, cos, int64, pi, sin, zeros
from traits.api import (CArray, Float, HasPrivateTraits, List, Property,
                        cached_property, on_trait_change)

from acoular.internal import digest
from acoular.microphones import MicGeom


class MicRing(HasPrivateTraits):
    """
    Helper class to define a circular ring of microphones.
    """
    
    #: List of microphone channels in the ring in anti-clockwise order.
    mics = List(
        desc = "list of microphones in this ring")
    
    #: Radius of the ring.
    r = Float(
        desc = "radius of ring")
    
    #: Angle of the first microphone in :attr:`mics` list in degrees.    
    phi0 = Float(
        desc = "angle of first mic in ring")
    
    #: Axial position of the microphone ring.
    z = Float(
        desc = "z position of ring")

class MicGeomCirc(MicGeom):
    """
    Provides the geometric arrangement of microphones in a circular array.
    
    The geometric arrangement of microphones is read in from an 
    xml-source with ring-wise entries of the 
    attributes z, r, phi0 and "miclist" (i.e. the list of uniformly
    distributed microphone channels in one ring). 
    Alternatively, the geometry can also be generated in-script.
    Invalid channels are defined by simply not including them in the
    xml file or script.
    """

    
    # internal identifier
    digest = Property( depends_on = ['mpos', ])

    #: List of microphone rings in the array (the list contains
    #: instances of the :class:`MicRing` class.
    #: If a compatible xml file containing a circular microphone geometry is given, the list will be generated automatically.
    #: It can, however also generated using scripts 
    #: (then :attr:`micindex` also has to be set).
    ringlist = List(
        desc = "list of microphone rings and their properties")    
    
    #: List that gives the indices of channels that should not be considered.
    #: Is set automatically depending on :attr:`micindex`.
    invalid_channels = Property( depends_on = ['mic_index', ],
        desc = "list of invalid channels")

    #: Array containing microphone indices in ring-wise order.
    #: Is set automatically when geometry is read from file.
    mic_index = CArray(
                desc = "indices of mics in ring-wise order as read from file")
        
    #: List to translate absolute mic_index to relative (for use 
    #: with mpos after removing invalids). Is set automatically.
    #:
    #: Usage: ``mpos[:,self.indices[self.ringlist[iring].mics[imic]]]``
    indices = Property( depends_on  = ['mic_index', ],
        desc = "list to translate absolute mic_index to relative (for use \
with mpos after removing invalids)")    
        # usage: mpos[:,self.indices[self.ringlist[iring].mics[imic]]] 
        # (^ to get translated results)
        # but: mpos_tot[:,self.ringlist[iring].mics[imic]]
    

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_invalid_channels( self ):
        #find invalid channels to omit them later when reading measured data...
        if (len(self.mic_index) == 0
         or len(self.mic_index) == (max(self.mic_index) + 1)):
            return []
        return list(set(range(max(self.mic_index) + 1)) - set(self.mic_index))

    @cached_property
    def _get_indices( self ):
        # use non-existing mic-position as invalid value...
        ind = zeros(max(self.mic_index)+1, dtype=int64) \
              + max(self.mic_index)+1
        ind[self.mic_index] = array(argsort(argsort(self.mic_index)))        
        return ind

    @cached_property
    def _get_num_mics( self ):
        return self.mic_index.shape[0]
        
    @on_trait_change('basename')
    def import_mpos( self ):
        """
        Import the microphone positions from .xml file.
        Automatically called when the file name changes.
        """
        if not path.isfile(self.from_file):
            # no file there
            self.mpos_tot = array([], 'd')
            #self.num_mics = 0
            print('\nGeometry file (', self.basename, '. xml ) not found!\n')
            return
            
        # open the xml file
        import xml.dom.minidom
        doc = xml.dom.minidom.parse(self.from_file)
        self.ringlist = []
        
        # create the list of rings with microphones
        for el in doc.getElementsByTagName('pos'):
            self.ringlist.append( \
                MicRing( mics = [ int(a) for a in  el.getAttribute('miclist').split()], \
                            r = float(el.getAttribute('r')), \
                         phi0 = float(el.getAttribute('phi0')), \
                            z = float(el.getAttribute('z')) ) )
        self.set_mpos()
                                           
    
    @on_trait_change('ringlist')
    def set_mpos(self):
        # list of index positions of mics (in data file)...
        num_rings = len(self.ringlist) # how many rings?        
        mic_index = sum([self.ringlist[a].mics for a in range(num_rings) ], [])

        if  len(mic_index) != len(set(mic_index)) :
        # duplicates in list?
            self.mpos_tot = array([], 'd')
            print('\nThere are duplicates in the microphone list.')
            print('Please check geometry file (', self.basename, '. xml )!\n')
            return
            
        num_rmics = array([len(self.ringlist[a].mics) for a in range(num_rings)])# how many mics in each ring?

        rphiz = zeros((max(mic_index)+1, 3))

        for iring in range(num_rings):
            for imic in range(num_rmics[iring]): # calculate polar coords...
                rphiz[self.ringlist[iring].mics[imic], :] = \
                        [ self.ringlist[iring].r, 
                          ((self.ringlist[iring].phi0+imic*360.0 / 
                                          num_rmics[iring])%360.0)*pi/180.0, 
                          self.ringlist[iring].z ]

        # calculate cartesian coords...
        self.mpos_tot = array( [rphiz[:, 0] * cos(rphiz[:, 1]), 
                                rphiz[:, 0] * sin(rphiz[:, 1]), 
                                rphiz[:, 2]], 'd' )
        self.mic_index = mic_index                          


