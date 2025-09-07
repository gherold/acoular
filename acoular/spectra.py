# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Estimation of power spectra and related tools.

.. autosummary::
    :toctree: generated/

    BaseSpectra
    PowerSpectra
    PowerSpectraImport
"""

from abc import abstractmethod

import numpy as np
from scipy import fft
from traits.api import (
    ABCHasStrictTraits,
    Bool,
    CArray,
    Delegate,
    Enum,
    Float,
    Instance,
    Int,
    Map,
    Property,
    Union,
    cached_property,
    property_depends_on,
)

# acoular imports
from .base import SamplesGenerator
from .configuration import config
from .deprecation import deprecated_alias
from .fastFuncs import calcCSM
from .grids import LatLongSphereGrid
from .h5cache import H5cache
from .h5files import H5CacheFileBase
from .internal import digest
from .microphones import MicGeom
from .tools.utils import find_basename
from .trajectory import Trajectory


@deprecated_alias(
    {'numchannels': 'num_channels', 'time_data': 'source'}, read_only=['numchannels'], removal_version='25.10'
)
class BaseSpectra(ABCHasStrictTraits):
    """
    Base class for handling spectral data in Acoular.

    This class defines the basic structure and functionality for computing and managing spectral
    data derived from time-domain signals. It includes properties for configuring the Fast Fourier
    Transformation (FFT), including overlap, and other parameters critical for spectral analysis.
    """

    #: Data source; an instance of :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    #: Sampling frequency of the output signal, delegated from :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of time microphones, delegated from :attr:`source`.
    num_channels = Delegate('source')

    #: Window function applied during FFT. Can be one of:
    #:
    #: - ``'Rectangular'`` (default)
    #:
    #: - ``'Hanning'``
    #:
    #: - ``'Hamming'``
    #:
    #: - ``'Bartlett'``
    #:
    #: - ``'Blackman'``
    window = Map(
        {
            'Rectangular': np.ones,
            'Hanning': np.hanning,
            'Hamming': np.hamming,
            'Bartlett': np.bartlett,
            'Blackman': np.blackman,
        },
        default_value='Rectangular',
        desc='type of window for FFT',
    )

    #: Overlap factor for FFT block averaging. One of:
    #:
    #: - ``'None'`` (default)
    #:
    #: - ``'50%'``
    #:
    #: - ``'75%'``
    #:
    #: - ``'87.5%'``
    overlap = Map({'None': 1, '50%': 2, '75%': 4, '87.5%': 8}, default_value='None', desc='overlap of FFT blocks')

    #: FFT block size. Must be one of: ``128``, ``256``, ``512``, ``1024``, ... ``65536``.
    #: Default is ``1024``.
    block_size = Enum(
        1024,
        128,
        256,
        512,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        desc='number of samples per FFT block',
    )

    #: Precision of the FFT, corresponding to NumPy dtypes. Default is ``'complex128'``.
    precision = Enum('complex128', 'complex64', desc='precision of the fft')

    #: A unique identifier for the spectra, based on its properties. (read-only)
    digest = Property(depends_on=['precision', 'block_size', 'window', 'overlap'])

    @abstractmethod
    def _get_digest(self):
        """Return internal identifier."""

    def fftfreq(self):
        """
        Compute and return the Discrete Fourier Transform sample frequencies.

        This method generates the frequency values corresponding to the FFT bins for the
        configured :attr:`block_size` and sampling frequency from the data source.

        Returns
        -------
        :obj:`numpy.ndarray` or :obj:`None`
            Array of shape ``(`` :attr:`block_size` ``/ 2 + 1,)`` containing the sample frequencies.
            If :attr:`source` is not set, returns ``None``.

        Examples
        --------
        Using normally distributed data for time samples as in
        :class:`~acoular.sources.TimeSamples`.

        >>> import numpy as np
        >>> from acoular import TimeSamples
        >>> from acoular.spectra import PowerSpectra
        >>>
        >>> data = np.random.rand(1000, 4)
        >>> ts = TimeSamples(data=data, sample_freq=51200)
        >>> print(ts.num_channels, ts.num_samples, ts.sample_freq)
        4 1000 51200.0
        >>> ps = PowerSpectra(source=ts, block_size=128, window='Blackman')
        >>> ps.fftfreq()
        array([    0.,   400.,   800.,  1200.,  1600.,  2000.,  2400.,  2800.,
                3200.,  3600.,  4000.,  4400.,  4800.,  5200.,  5600.,  6000.,
                6400.,  6800.,  7200.,  7600.,  8000.,  8400.,  8800.,  9200.,
                9600., 10000., 10400., 10800., 11200., 11600., 12000., 12400.,
               12800., 13200., 13600., 14000., 14400., 14800., 15200., 15600.,
               16000., 16400., 16800., 17200., 17600., 18000., 18400., 18800.,
               19200., 19600., 20000., 20400., 20800., 21200., 21600., 22000.,
               22400., 22800., 23200., 23600., 24000., 24400., 24800., 25200.,
               25600.])
        """
        if self.source is not None:
            return abs(fft.fftfreq(self.block_size, 1.0 / self.source.sample_freq)[: int(self.block_size / 2 + 1)])
        return None

    # generator that yields the time data blocks for every channel (with optional overlap)
    def _get_source_data(self):
        bs = self.block_size
        temp = np.empty((2 * bs, self.num_channels))
        pos = bs
        posinc = bs / self.overlap_
        for data_block in self.source.result(bs):
            ns = data_block.shape[0]
            temp[bs : bs + ns] = data_block  # fill from right
            while pos + bs <= bs + ns:
                yield temp[int(pos) : int(pos + bs)]
                pos += posinc
            else:
                temp[0:bs] = temp[bs:]  # copy to left
                pos -= bs


class Spectra (BaseSpectra):
       
    #: Flag, if true (default), the result is cached in h5 files and need not
    #: to be recomputed during subsequent program runs.
    cached = Bool(True, 
        desc="cached flag")   

    #: Number of FFT blocks to average, readonly
    #: (set from block_size and overlap).
    num_blocks = Property(
        desc="overall number of FFT blocks")
    
    #: A unique identifier for the spectra, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'precision', 'block_size', 'window', 'overlap'])
    
    # hdf5 cache file
    h5f = Instance( H5CacheFileBase, transient = True )
    
    #: Name of the cache file without extension, readonly.
    basename = Property( depends_on = 'time_data.digest', 
        desc="basename for cache file")
    
    
    #: The auto power spectrum, 
    #: (number of frequencies, numchannels) array of float;
    #: readonly.
    aps = Property( 
        desc="auto power spectrum")


    @cached_property
    def _get_digest( self ):
        return digest( self )

    @cached_property
    def _get_basename( self ):
        if 'basename' in self.time_data.all_trait_names():
            return self.time_data.basename
        else: 
            return self.time_data.__class__.__name__ + self.time_data.digest

    @property_depends_on('time_data.numsamples, block_size, overlap')
    def _get_num_blocks ( self ):
        return self.overlap_*self.time_data.numsamples//self.block_size-\
        self.overlap_+1



    def calc_power_spectrum( self ):
        """ power spectrum calculation """
        t = self.source
        wind = self.window_( self.block_size )
        weight = np.dot( wind, wind )
        wind = wind[np.newaxis, :].swapaxes( 0, 1 )
        numfreq = int(self.block_size/2 + 1)

        bs = self.block_size
        
        # use faster rfft if input is real, otherwise full fft
        if np.isrealobj(next(t.result(1))):
            fft_func = fft.rfft
            temp = np.empty((2*bs, t.numchannels))
        else:
            fft_func = fft.fft
            temp = np.empty((2*bs, t.numchannels), dtype = self.precision)
            
        powspec = np.zeros((numfreq,t.numchannels), dtype=self.precision_)
        pos = bs
        posinc = bs/self.overlap_
        for data in t.result(bs):
            ns = data.shape[0]
            temp[bs:bs+ns] = data
            while pos+bs <= bs+ns:
                ft = fft_func(temp[int(pos):int(pos+bs)]*wind, None, 0)[:numfreq].astype(self.precision)
                powspec += (ft*ft.conjugate()).real
                pos += posinc
            temp[0:bs] = temp[bs:]
            pos -= bs
        
        # onesided spectrum: multiplication by 2.0=sqrt(2)^2
        powspec *= (2.0/self.block_size/weight/self.num_blocks)
        return powspec


    @property_depends_on('digest')
    def _get_aps ( self ):
        """
        Main work is done here:
        Cross spectral matrix is either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        #self._handle_dual_calibration()
        if (
                config.global_caching == 'none' or 
                (config.global_caching == 'individual' and self.cached == False)
            ):
            return self.calc_power_spectrum()
        else:
            return self._get_filecache('aps')


    def _get_filecache( self, traitname ):
        """
        function handles result caching of csm, eigenvectors and eigenvalues
        calculation depending on global/local caching behaviour.  
        """
        if traitname == 'aps':
            func = self.calc_power_spectrum
            numfreq = int(self.block_size/2 + 1)
            shape = (numfreq, self.numchannels)
            precision = self.precision_
        else:
            raise NotImplementedError('Only auto-power spectrum supported.')

        H5cache.get_cache_file( self, self.basename ) 
        if not self.h5f: # in case of global caching readonly
            return func() 

        nodename = traitname + '_' + self.digest 
        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
            #print("remove existing node",nodename)
            self.h5f.remove_data(nodename) # remove old data before writing in overwrite mode
        
        if not self.h5f.is_cached(nodename): 
            if config.global_caching == 'readonly': 
                return func()
#            print("create array, data not cached for",nodename)
            self.h5f.create_compressible_array(nodename,shape,precision)
            
        ac = self.h5f.get_data_by_reference(nodename)
        if ac[:].sum() == 0: # only initialized
#            print("write {} to:".format(traitname),nodename)
            ac[:] = func()
            self.h5f.flush()
        return ac




    def fftfreq ( self ):
        """
        Return the Discrete Fourier Transform sample frequencies.
        
        Returns
        -------
        f : ndarray
            Array of length *block_size/2+1* containing the sample frequencies.
        """
        return abs(fft.fftfreq(self.block_size, 1./self.source.sample_freq)\
                    [:int(self.block_size/2+1)])


class CollectGridTrajSpectra(Spectra):
       
    #: Number of channels in output (= number of sectors).
    num_channels = Property( depends_on = ['grid', ])
    
    

    #: :class:`~acoular.trajectory.Trajectory` or derived object.
    #: Start time is assumed to be the same as for the samples.
    trajectory = Instance(Trajectory, 
        desc="trajectory of the grid center")
    
    
    #: if set to True, ignore first and last pos in traj for rotational orientation
    z_orientation = Bool(False)
    
    
    #: approximate the trajectory with a straight line and define it as z axis
    #: this will help comparing different trajectories with each other
    rotation = Property(depends_on = ['trajectory','time_data.digest', 'z_orientation'])
    
    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Instance(MicGeom, 
        desc="microphone geometry")
    
    
    grid = Instance(LatLongSphereGrid,
                 desc="grid for spectra collection")
    
    ### for debugging and checking
    rotraj =  Instance(Trajectory(),Trajectory)
    _sph_coords = CArray()
    _grid_num_blocks = CArray()
    ### ------------------


    #: The total number of blocks for each grid point 
    #: (number of frequencies, numchannels) array of float;
    #: readonly.
    blocks_per_grid  = Property( 
                    desc="number of blocks per grid point")    
    
    # internal identifier
    digest = Property( 
        depends_on = ['time_data.digest', 'block_size', 
            'window', 'overlap', 'precision', 'trajectory.digest',
            'mics.digest', 'grid.digest'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    
    def _get_num_channels(self):
        return self.grid.size
    
    
    @property_depends_on('digest')
    def _get_blocks_per_grid ( self ):
        """

        """
        #self._handle_dual_calibration()
        if (
                config.global_caching == 'none' or 
                (config.global_caching == 'individual' and self.cached == False)
            ):
            # trigger calculation
            return self.calc_grid_num_blocks()
        else:
            return self._get_filecache('blocks_per_grid')
    
    
    
    @cached_property
    def _get_rotation ( self ):
        if self.z_orientation:
            Ry_neg = np.array([[ 1, 0, 0],
                            [ 0, 1, 0],
                            [ 0, 0, 1]])
        else:
            
            t_end = self.source.num_samples/self.source.sample_freq
            xstart = np.array(self.trajectory.location(0))
            xend = np.array(self.trajectory.location(t_end))
            vec = xend - xstart
    
            # distance in xz plane (y is ignored b/c gravity should orient drone), RAR
            #r_xz = (vec[0]**2 + vec[2]**2)**0.5     #xz plane, RAR
            
            # distance in plane parallel to ground, Wesendorf: dim 0, 1 (xy)
            rtraj = (vec[0]**2 + vec[1]**2)**0.5 


            # get angle alpha
            sin_alpha = vec[0]/rtraj
            cos_alpha = vec[1]/rtraj
            #cos_alpha = -vec[2]/r_xz # minus sign because left-oriented z axis
    
            
            # get negative (!) angle rotation matrix for left-oriented system
            # (negative because we want to rotate the coords in the other direction)
            Ry_RAR = np.array([[ cos_alpha, 0, sin_alpha],
                            [         0, 1,         0],
                            [-sin_alpha, 0, cos_alpha]])# MicGeom-Setup im RAR
            
            Ry_Wes = np.array([[-cos_alpha, sin_alpha,  0],
                            [         0,         0, -1],
                            [ sin_alpha, cos_alpha,  0]])# MicGeomSetup in Wesendorf
            
            Ry_WesAIAA = np.array([[ 0, 1, 0],
                                [ 0, 0,-1],
                                [ 1, 0, 0]])# vereinfachtes MicGeomSetup in Wesendorf (funktioniert für AIAA, Ri "Zurück")
            Ry_WesDAGA = np.array([[ 0,-1, 0],
                                [ 0, 0,-1],
                                [-1, 0, 0]])# vereinfachtes MicGeomSetup in Wesendorf (funktioniert für DAGA, Ri "Hin", wenn unten auch unverändert)
            Ry_neg = Ry_Wes
        
        return Ry_neg
    
    def calc_grid_num_blocks( self ):
        # trigger calculation if necessary
        if self._grid_num_blocks[:].sum()==0:
            self.calc_power_spectrum()
        return self._grid_num_blocks
    
    def calc_power_spectrum( self ):
        """ power spectrum calculation """
        
        # rotation matrix
        rot = self.rotation
        # initialize rotated trajectory
        #rotraj = Trajectory()
        for key in self.trajectory.points.keys():
            self.rotraj.points[key] = tuple(rot @ self.trajectory.points[key])#changed because coord system changed
            #self.rotraj.points[key] = tuple(self.trajectory.points[key])
        
        # get relative orientation of mic geom
        mpos = rot @ self.mics.pos    #changed because coord system changed
        #mpos = self.mics.mpos    
        
        # some abbreviations for spectra calculation
        t = self.source
        wind = self.window_( self.block_size )
        weight = np.dot( wind, wind )
        wind = wind[np.newaxis, :].swapaxes( 0, 1 )
        numfreq = int(self.block_size/2 + 1)
        bs = self.block_size
        
        # use faster rfft if input is real, otherwise full fft
        if np.isrealobj(next(t.result(1))):
            fft_func = fft.rfft
            temp = np.zeros((2*bs, t.num_channels), dtype=self.precision_)
        else:
            fft_func = fft.fft
            temp = np.zeros((2*bs, t.num_channels), dtype=self.precision)
            
        # allocate array with gridpos-specific averaging number
        grid_num_blocks = np.zeros((self.num_channels,),dtype=np.uint32)
        # allocate array fpr spectra results
        powspec = np.zeros((numfreq, self.num_channels), dtype=self.precision_)
        pos = bs
        posinc = bs/self.overlap_
        
        # duration of on block of length "bs" divided by overlap
        dt = 1/t.sample_freq * posinc
        # start trajectory at "center" of first block
        t_start = 1/t.sample_freq * bs/2
        # define trajectory whose position advances each block
        trajblock = self.rotraj.traj(t_start, delta_t=dt)
        
        ### temporary, for checking and debugging
        spherical_coords = np.zeros((3, t.numchannels, self.num_blocks))
        isc = 0       
        ### ------------------
        
        for data in t.result(bs):
            ### temporary, for checking and debugging
            print(f'{isc/self.num_blocks*100:.2f}',end=' | ', flush=True)
            ### ------------------
            ns = data.shape[0]
            temp[bs:bs+ns] = data
            while pos+bs <= bs+ns:
                ft = fft_func(temp[int(pos):int(pos+bs)]*wind, None, 0)[:numfreq].astype(self.precision)
                
                trajpos = next(trajblock)
                # relative vector, from trajectory position to mics
                xrel = mpos - np.array(trajpos)[:, np.newaxis]
                
                # get relative spherical coords
                r = np.norm(xrel, axis=0)
                thetas = np.arccos(xrel[2]/r) # polar angle / latitude (0..180°) 0..pi
                phis =  np.arctan2(xrel[1], xrel[0]) # azimuthal angle / longitude (0..360°) 0..2pi
                #thetas = arccos(-xrel[0]/r) # polar angle / latitude (0..180°) 0..pi
                #phis =  arctan2(-xrel[2], -xrel[1]) # azimuthal angle / longitude (0..360°) 0..2pi
                
                
                ### temp for debugging
                spherical_coords[0,:,isc] = r
                spherical_coords[1,:,isc] = thetas
                spherical_coords[2,:,isc] = phis
                isc += 1
                ### ------------------
                
                # loop through mics and look to which grid portion spectra belong
                for ichannel, (theta, phi) in enumerate(zip(thetas, phis)):
                    ispec = ft[:,ichannel]
                    grid_index = self.grid.index(phi, theta)
                    powspec[:, grid_index] += (ispec*ispec.conjugate()).real
                    grid_num_blocks[grid_index] += 1
                #print(f'<{grid_index}> {phi:.1f} {theta:.1f}',end=' -- ', flush=True)                
                pos += posinc
            temp[0:bs] = temp[bs:]
            pos -= bs
        
        # onesided spectrum: multiplication by 2.0=sqrt(2)^2
        powspec *= (2.0/self.block_size/weight/grid_num_blocks[np.newaxis,:])
        
        ### temp for debugging
        self._sph_coords = spherical_coords
        self._grid_num_blocks = grid_num_blocks # may be kept later on?
        ### ------------------
        
        return powspec
    
    
    def _get_filecache( self, traitname ):
        """
        function handles result caching of csm, eigenvectors and eigenvalues
        calculation depending on global/local caching behaviour.  
        """
        if traitname == 'aps':
            func = self.calc_power_spectrum
            numfreq = int(self.block_size/2 + 1)
            shape = (numfreq, self.num_channels)
            precision = self.precision_
        elif traitname == 'blocks_per_grid':
            func = self.calc_grid_num_blocks
            numfreq = int(self.block_size/2 + 1)
            shape = (self.num_channels,)
            precision = 'uint32'
        else:
            raise NotImplementedError('Only auto-power spectrum and number of blocks supported.')

        H5cache.get_cache_file( self, self.basename ) 
        if not self.h5f: # in case of global caching readonly
            return func() 

        nodename = traitname + '_' + self.digest 
        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
            #print("remove existing node",nodename)
            self.h5f.remove_data(nodename) # remove old data before writing in overwrite mode
        
        if not self.h5f.is_cached(nodename): 
            if config.global_caching == 'readonly': 
                return func()
#            print("create array, data not cached for",nodename)
            self.h5f.create_compressible_array(nodename,shape,precision)
            
        ac = self.h5f.get_data_by_reference(nodename)
        if ac[:].sum() == 0: # only initialized
#            print("write {} to:".format(traitname),nodename)
            ac[:] = func()
            self.h5f.flush()
        return ac




class CollectDetailedGridTrajSpectra(CollectGridTrajSpectra):
       

    #: ( num_blocks, numchannels, 5)
    spec_history_per_grid = Property( 
                    desc="the full spectral history for each grid point")    


    #: The total number of blocks for each grid point 
    #: (number of frequencies, numchannels) array of float;
    #: readonly.
    blocks_per_grid  = Property( 
                    desc="number of blocks per grid point")    
    
    
    
    @property_depends_on('digest')
    def _get_spec_history_per_grid ( self ):
        """

        """
        #self._handle_dual_calibration()
        if (
                config.global_caching == 'none' or 
                (config.global_caching == 'individual' and self.cached == False)
            ):
            # trigger calculation
            return self.calc_spec_history_per_grid()
        else:
            return self._get_filecache('spec_history_per_grid')
    
    _history_per_grid = CArray()
    
    
    def calc_spec_history_per_grid( self ):
        # trigger calculation if necessary
        if self._history_per_grid[:].sum()==0:
            self.calc_power_spectrum()
        return self._history_per_grid
    
    def calc_power_spectrum( self ):
        """ power spectrum calculation """
        
        freqs = self.fftfreq()
        freqinds = np.array([np.searchsorted(freqs, freq) for freq in [500,1000,2000,4000,8000]])
        
        # rotation matrix
        rot = self.rotation
        # initialize rotated trajectory
        #rotraj = Trajectory()
        for key in self.trajectory.points.keys():
            self.rotraj.points[key] = tuple(rot @ self.trajectory.points[key])
        
        # get relative orientation of mic geom
        mpos = rot @ self.mics.mpos    
        
        # some abbreviations for spectra calculation
        t = self.time_data
        wind = self.window_( self.block_size )
        weight = np.dot( wind, wind )
        wind = wind[np.newaxis, :].swapaxes( 0, 1 )
        numfreq = int(self.block_size/2 + 1)
        bs = self.block_size
        
        # use faster rfft if input is real, otherwise full fft
        if np.isrealobj(next(t.result(1))):
            fft_func = fft.rfft
            temp = np.zeros((2*bs, t.numchannels), dtype=self.precision_)
        else:
            fft_func = fft.fft
            temp = np.zeros((2*bs, t.numchannels), dtype=self.precision)
            
        # allocate array with gridpos-specific averaging number
        grid_num_blocks = np.zeros((self.numchannels,),dtype=np.uint32)
        
        # allocate history array, for 5 representative frequencies (500,1k,2k,4k,8k Hz)
        history_per_grid = np.empty((self.num_blocks,self.numchannels,5),dtype=self.precision_)
        history_per_grid[:] = np.nan
        # allocate array fpr spectra results
        powspec = np.zeros((numfreq, self.numchannels), dtype=self.precision_)
        pos = bs
        posinc = bs/self.overlap_
        
        # duration of on block of length "bs" divided by overlap
        dt = 1/t.sample_freq * posinc
        # start trajectory at "center" of first block
        t_start = 1/t.sample_freq * bs/2
        # define trajectory whose position advances each block
        trajblock = self.rotraj.traj(t_start, delta_t=dt)
        
        ### temporary, for checking and debugging
        spherical_coords = np.zeros((3, t.numchannels, self.num_blocks))
        isc = 0       
        ### ------------------
        
        for data in t.result(bs):
            ### temporary, for checking and debugging
            print(f'{isc/self.num_blocks*100:.2f}',end=' | ', flush=True)
            ### ------------------
            ns = data.shape[0]
            temp[bs:bs+ns] = data
            while pos+bs <= bs+ns:
                ft = fft_func(temp[int(pos):int(pos+bs)]*wind, None, 0)[:numfreq].astype(self.precision)
                
                trajpos = next(trajblock)
                # relative vector, from trajectory position to mics
                xrel = mpos - np.array(trajpos)[:, np.newaxis]
                
                # get relative spherical coords
                r = np.norm(xrel, axis=0)
                thetas = np.arccos(xrel[2]/r) # polar angle / latitude (0..180°) 0..pi
                phis =  np.arctan2(xrel[1], xrel[0]) # azimuthal angle / longitude (0..360°) 0..2pi
                
                ### temp for debugging
                spherical_coords[0,:,isc] = r
                spherical_coords[1,:,isc] = thetas
                spherical_coords[2,:,isc] = phis
                ### ------------------
                
                # loop through mics and look to which grid portion spectra belong
                for ichannel, (theta, phi) in enumerate(zip(thetas, phis)):
                    ispec = ft[:,ichannel]
                    grid_index = self.grid.index(phi, theta)
                    localspec = (ispec*ispec.conjugate()).real
                    powspec[:, grid_index] += localspec
                    grid_num_blocks[grid_index] += 1
                    history_per_grid[isc, grid_index] = localspec[freqinds]
                #print(f'<{grid_index}> {phi:.1f} {theta:.1f}',end=' -- ', flush=True)                
                isc += 1
                pos += posinc
            temp[0:bs] = temp[bs:]
            pos -= bs
        
        # onesided spectrum: multiplication by 2.0=sqrt(2)^2
        powspec *= (2.0/self.block_size/weight/grid_num_blocks[np.newaxis,:])
        
        ### temp for debugging
        self._sph_coords = spherical_coords
        self._grid_num_blocks = grid_num_blocks # may be kept later on?
        self._history_per_grid = history_per_grid
        ### ------------------
        
        return powspec
    
    
    def _get_filecache( self, traitname ):
        """
        function handles result caching of csm, eigenvectors and eigenvalues
        calculation depending on global/local caching behaviour.  
        """
        if traitname == 'aps':
            func = self.calc_power_spectrum
            numfreq = int(self.block_size/2 + 1)
            shape = (numfreq, self.numchannels)
            precision = self.precision_
        elif traitname == 'blocks_per_grid':
            func = self.calc_grid_num_blocks
            shape = (self.numchannels,)
            precision = 'uint32'
        elif traitname == 'spec_history_per_grid':
            func = self.calc_spec_history_per_grid
            shape = (self.num_blocks,self.numchannels,5)
            precision = self.precision_
        else:
            raise NotImplementedError('Only auto-power spectrum and number of blocks supported.')

        H5cache.get_cache_file( self, self.basename ) 
        if not self.h5f: # in case of global caching readonly
            return func() 

        nodename = traitname + '_' + self.digest 
        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
            #print("remove existing node",nodename)
            self.h5f.remove_data(nodename) # remove old data before writing in overwrite mode
        
        if not self.h5f.is_cached(nodename): 
            if config.global_caching == 'readonly': 
                return func()
#            print("create array, data not cached for",nodename)
            self.h5f.create_compressible_array(nodename,shape,precision)
            
        ac = self.h5f.get_data_by_reference(nodename)
        if ac[:].sum() == 0: # only initialized
#            print("write {} to:".format(traitname),nodename)
            ac[:] = func()
            self.h5f.flush()
        return ac



class PowerSpectra(BaseSpectra):
    """
    Provides the cross-spectral matrix of multichannel time-domain data and its eigen-decomposition.

    This class is designed to compute the cross-spectral matrix (CSM) efficiently using the Welch
    method :cite:`Welch1967` with support for windowing and overlapping data segments. It also
    calculates the eigenvalues and eigenvectors of the CSM, allowing for spectral analysis and
    advanced signal processing tasks.

    Key features:
        - **Efficient Calculation**: Computes the CSM using FFT-based methods.
        - **Caching**: Results can be cached in HDF5 files to avoid redundant calculations for
          identical inputs and parameters.
        - **Lazy Evaluation**: Calculations are triggered only when attributes like :attr:`csm`,
          :attr:`eva`, or :attr:`eve` are accessed.
        - **Dynamic Input Handling**: Automatically recomputes results when the input data or
          parameters change.
    """

    #: The data source for the time-domain samples. It must be an instance of
    #: :class:`SamplesGenerator<acoular.base.SamplesGenerator>` or a derived class.
    source = Instance(SamplesGenerator)

    # Shadow trait, should not be set directly, for internal use.
    _ind_low = Int(1, desc='index of lowest frequency line')

    # Shadow trait, should not be set directly, for internal use.
    _ind_high = Union(Int(-1), None, desc='index of highest frequency line')

    #: Index of lowest frequency line to compute. Default is ``1``. Only used by objects that fetch
    #: the CSM. PowerSpectra computes every frequency line.
    ind_low = Property(_ind_low, desc='index of lowest frequency line')

    #: Index of highest frequency line to compute. Default is ``-1``
    #: (last possible line for default :attr:`~BaseSpectra.block_size`).
    ind_high = Property(_ind_high, desc='index of lowest frequency line')

    # Stores the set lower frequency, for internal use, should not be set directly.
    _freqlc = Float(0)

    # Stores the set higher frequency, for internal use, should not be set directly.
    _freqhc = Union(Float(0), None)

    # Saves whether the user set indices or frequencies last, for internal use only, not to be set
    # directly, if ``True``, indices are used for setting the :attr:`freq_range` interval.
    # Default is ``True``.
    _index_set_last = Bool(True)

    #: A flag indicating whether the result should be cached in HDF5 files. Default is ``True``.
    cached = Bool(True, desc='cached flag')

    #: The number of FFT blocks used for averaging. This is derived from the
    #: :attr:`~BaseSpectra.block_size` and :attr:`~BaseSpectra.overlap` parameters. (read-only)
    num_blocks = Property(desc='overall number of FFT blocks')

    #: 2-element array with the lowest and highest frequency. If the higher frequency is larger than
    #: the max frequency, the max frequency will be the upper bound.
    freq_range = Property(desc='frequency range')
    # If set, will overwrite :attr:`_freqlc`  and :attr:`_freqhc` according to the range. The
    # freq_range interval will be the smallest discrete frequency inside the half-open interval
    # [_freqlc, _freqhc[ and the smallest upper frequency outside of the interval.

    #: The sequence of frequency indices between :attr:`ind_low` and :attr:`ind_high`. (read-only)
    indices = Property(desc='index range')

    #: The name of the cache file (without the file extension) used for storing results. (read-only)
    basename = Property(depends_on=['source.digest'], desc='basename for cache file')

    #: The cross spectral matrix, 
    #: (number of frequencies, numchannels, numchannels) array of complex;
    #: readonly.
    csm = Property( 
        desc="cross spectral matrix")
    
    
    diagonal = Enum('original', 'original', 'zero', 'reconstruct')
    
       
    
    #TODO: letztendlich verwendete CSM soll konstruiert wereden aus original 
    #CSM + ersetzter separater Hauptdiagonale, die je nach "diagonal"-trait berechnet
    #bzw. gecached wird
    
    #: The floating-number-precision of entries of csm, eigenvalues and 
    #: eigenvectors, corresponding to numpy dtypes. Default is 64 bit.
    precision = Enum('complex128', 'complex64', 
                      desc="precision csm, eva, eve")

    #: Eigenvalues of the cross spectral matrix as an
    #: (number of frequencies) array of floats, readonly.
    eva = Property( 
        desc="eigenvalues of cross spectral matrix")

    #: The eigenvectors of the cross spectral matrix, stored as an array of shape ``(n, m, m)`` of
    #: floats for ``n`` frequencies and ``m`` channels as in :attr:`~BaseSpectra.num_channels`.
    #: (read-only)
    eve = Property(desc='eigenvectors of cross spectral matrix')

    #: A unique identifier for the spectra, based on its properties.  (read-only)
    digest = Property(
        depends_on=['source.digest', 'block_size', 'window', 'overlap', 'precision'],
    )

    #: The HDF5 cache file used for storing the results if :attr:`cached` is set to ``True``.
    h5f = Instance(H5CacheFileBase, transient=True)

    @property_depends_on(['source.num_samples', 'block_size', 'overlap'])
    def _get_num_blocks(self):
        return self.overlap_ * self.source.num_samples / self.block_size - self.overlap_ + 1

    @property_depends_on(['source.sample_freq', 'block_size', 'ind_low', 'ind_high'])
    def _get_freq_range(self):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._ind_high is None:
                return np.array([fftfreq[self.ind_low], None])
            return fftfreq[[self.ind_low, self.ind_high]]
        return None

    def _set_freq_range( self, freq_range ):# by setting this the user sets _freqlc and _freqhc
        self._index_set_last = False
        self._freqlc = freq_range[0]
        if len(freq_range)==1: # take next available frequency as upper limit
            self._freqhc = freq_range[0]+self.fftfreq()[1]
        else: # 2 entries?
            self._freqhc = freq_range[1]

    @property_depends_on(['source.sample_freq', 'block_size', '_ind_low', '_freqlc'])
    def _get_ind_low(self):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._index_set_last:
                return min(self._ind_low, fftfreq.shape[0] - 1)
            return np.searchsorted(fftfreq[:-1], self._freqlc)
        return 0

    @property_depends_on(['source.sample_freq', 'block_size', '_ind_high', '_freqhc'])
    def _get_ind_high(self):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._index_set_last:
                if self._ind_high is None:
                    return None
                return min(self._ind_high, fftfreq.shape[0] - 1)
            if self._freqhc is None:
                return None
            return np.searchsorted(fftfreq[:-1], self._freqhc)
        return None

    def _set_ind_high(self, ind_high):  # by setting this the user sets the lower index
        self._index_set_last = True
        self._ind_high = ind_high

    def _set_ind_low(self, ind_low):  # by setting this the user sets the higher index
        self._index_set_last = True
        self._ind_low = ind_low

    @property_depends_on(['block_size', 'ind_low', 'ind_high'])
    def _get_indices(self):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            try:
                indices = np.arange(fftfreq.shape[0], dtype=int)
                if self.ind_high is None:
                    return indices[self.ind_low :]
                return indices[self.ind_low : self.ind_high]
            except IndexError:
                return range(0)
        return None

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        return find_basename(self.source, alternative_basename=self.source.__class__.__name__ + self.source.digest)

    def calc_csm(self):
        """
        Calculate the CSM for the given source data.

        This method computes the CSM by performing a block-wise Fast Fourier Transform (FFT) on the
        source data, applying a window function, and averaging the results. Only the upper
        triangular part of the matrix is computed for efficiency, and the lower triangular part is
        constructed via transposition and complex conjugation.

        Returns
        -------
        :obj:`numpy.ndarray`
            The computed cross spectral matrix as an array of shape ``(n, m, m)`` of complex values
            for ``n`` frequencies and ``m`` channels as in :attr:`~BaseSpectra.num_channels`.

        Examples
        --------
        >>> import numpy as np
        >>> from acoular import TimeSamples
        >>> from acoular.spectra import PowerSpectra
        >>>
        >>> data = np.random.rand(1000, 4)
        >>> ts = TimeSamples(data=data, sample_freq=51200)
        >>> print(ts.num_channels, ts.num_samples, ts.sample_freq)
        4 1000 51200.0
        >>> ps = PowerSpectra(source=ts, block_size=128, window='Blackman')
        >>> ps.csm.shape
        (65, 4, 4)
        """
        t = self.source
        wind = self.window_(self.block_size)
        weight = np.dot(wind, wind)
        wind = wind[np.newaxis, :].swapaxes(0, 1)
        numfreq = int(self.block_size / 2 + 1)
        csm_shape = (numfreq, t.num_channels, t.num_channels)
        csm_upper = np.zeros(csm_shape, dtype=self.precision)
        # get time data blockwise
        for data in self._get_source_data():
            ft = fft.rfft(data * wind, None, 0).astype(self.precision)
            calcCSM(csm_upper, ft)  # only upper triangular part of matrix is calculated (for speed reasons)
        # create the full csm matrix via transposing and complex conj.
        # create the full csm matrix via transposingand complex conj.
        if self.diagonal == 'zero': # this is only for the time being, do this outside the function later on!
            [np.fill_diagonal(csm_upper[cntFreq, :, :], 0) for cntFreq in range(csm_upper.shape[0])]
            csm_lower = csm_upper.conj().transpose(0,2,1)
        else:    
            csm_lower = csm_upper.conj().transpose(0,2,1)
            [np.fill_diagonal(csm_lower[cntFreq, :, :], 0) for cntFreq in range(csm_lower.shape[0])]
        
        csm = csm_lower + csm_upper
        # onesided spectrum: multiplication by 2.0=sqrt(2)^2
        return csm * (2.0 / self.block_size / weight / self.num_blocks)

    def calc_ev(self):
        """
        Calculate eigenvalues and eigenvectors of the CSM for each frequency.

        The eigenvalues represent the spectral power, and the eigenvectors correspond to the
        principal components of the matrix. This calculation is performed for all frequency slices
        of the CSM.

        Returns
        -------
        :class:`tuple` of :obj:`numpy.ndarray`
            A tuple containing:
                - :attr:`eva` (:obj:`numpy.ndarray`): Eigenvalues as a 2D array of shape ``(n, m)``,
                  where ``n`` is the number of frequencies and ``m`` is the number of channels. The
                  datatype depends on the precision.
                - :attr:`eve` (:obj:`numpy.ndarray`): Eigenvectors as a 3D array of shape
                  ``(n, m, m)``. The datatype is consistent with the precision of the input data.

        Notes
        -----
        - The precision of the eigenvalues is determined by :attr:`~BaseSpectra.precision`
          (``'float64'`` for ``complex128`` precision and ``'float32'`` for ``complex64``
          precision).
        - This method assumes the CSM is already computed and accessible via :attr:`csm`.

        Examples
        --------
        >>> import numpy as np
        >>> from acoular import TimeSamples
        >>> from acoular.spectra import PowerSpectra
        >>>
        >>> data = np.random.rand(1000, 4)
        >>> ts = TimeSamples(data=data, sample_freq=51200)
        >>> ps = PowerSpectra(source=ts, block_size=128, window='Hanning')
        >>> eva, eve = ps.calc_ev()
        >>> print(eva.shape, eve.shape)
        (65, 4) (65, 4, 4)
        """
        if self.precision == 'complex128':
            eva_dtype = 'float64'
        elif self.precision == 'complex64':
            eva_dtype = 'float32'
        #        csm = self.csm #trigger calculation
        csm_shape = self.csm.shape
        eva = np.empty(csm_shape[0:2], dtype=eva_dtype)
        eve = np.empty(csm_shape, dtype=self.precision)
        for i in range(csm_shape[0]):
            (eva[i], eve[i]) = np.linalg.eigh(self.csm[i])
        return (eva, eve)

    def calc_eva(self):
        """
        Calculate eigenvalues of the CSM.

        This method computes and returns the eigenvalues of the CSM for all frequency slices.

        Returns
        -------
        :obj:`numpy.ndarray`
            A 2D array of shape ``(n, m)`` containing the eigenvalues for ``n`` frequencies and
            ``m`` channels. The datatype depends on :attr:`~BaseSpectra.precision` (``'float64'``
            for ``complex128`` precision and ``'float32'`` for ``complex64`` precision).

        Notes
        -----
        This method internally calls :meth:`calc_ev` and extracts only the eigenvalues.
        """
        return self.calc_ev()[0]

    def calc_eve(self):
        """
        Calculate eigenvectors of the Cross Spectral Matrix (CSM).

        This method computes and returns the eigenvectors of the CSM for all frequency slices.

        Returns
        -------
        :obj:`numpy.ndarray`
            A 3D array of shape ``(n, m, m)`` containing the eigenvectors for ``n`` frequencies and
            ``m`` channels. Each slice ``eve[f]`` represents an ``(m, m)`` matrix of eigenvectors
            for frequency ``f``. The datatype matches the :attr:`~BaseSpectra.precision` of the CSM
            (``complex128`` or ``complex64``).

        Notes
        -----
        This method internally calls :meth:`calc_ev()` and extracts only the eigenvectors.
        """
        return self.calc_ev()[1]

    def _get_filecache(self, traitname):
        # Handle caching of results for CSM, eigenvalues, and eigenvectors.
        # Returns the requested data (``csm``, ``eva``, or ``eve``) as a NumPy array.
        if traitname == 'csm':
            func = self.calc_csm
            numfreq = int(self.block_size / 2 + 1)
            shape = (numfreq, self.source.num_channels, self.source.num_channels)
            precision = self.precision
        elif traitname == 'eva':
            func = self.calc_eva
            shape = self.csm.shape[0:2]
            if self.precision == 'complex128':
                precision = 'float64'
            elif self.precision == 'complex64':
                precision = 'float32'
        elif traitname == 'eve':
            func = self.calc_eve
            shape = self.csm.shape
            precision = self.precision

        H5cache.get_cache_file(self, self.basename)
        if not self.h5f:  # in case of global caching readonly
            return func()

        nodename = traitname + '_' + self.digest
        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
            # print("remove existing node",nodename)
            self.h5f.remove_data(nodename)  # remove old data before writing in overwrite mode

        if not self.h5f.is_cached(nodename):
            if config.global_caching == 'readonly':
                return func()
            #            print("create array, data not cached for",nodename)
            self.h5f.create_compressible_array(nodename, shape, precision)

        ac = self.h5f.get_data_by_reference(nodename)
        if ac[:].sum() == 0:  # only initialized
            #            print("write {} to:".format(traitname),nodename)
            ac[:] = func()
            self.h5f.flush()
        return ac

    @property_depends_on(['digest'])
    def _get_csm(self):
        if config.global_caching == 'none' or (config.global_caching == 'individual' and self.cached is False):
            return self.calc_csm()
        return self._get_filecache('csm')

    @property_depends_on(['digest'])
    def _get_eva(self):
        if config.global_caching == 'none' or (config.global_caching == 'individual' and self.cached is False):
            return self.calc_eva()
        return self._get_filecache('eva')

    @property_depends_on(['digest'])
    def _get_eve(self):
        if config.global_caching == 'none' or (config.global_caching == 'individual' and self.cached is False):
            return self.calc_eve()
        return self._get_filecache('eve')

    def synthetic_ev(self, freq, num=0):
        """
        Retrieve synthetic eigenvalues for a specified frequency or frequency range.

        This method calculates the eigenvalues of the CSM for a single frequency or a synthetic
        frequency range. If ``num`` is set to ``0``, it retrieves the eigenvalues at the exact
        frequency. Otherwise, it averages eigenvalues across a range determined by ``freq`` and
        ``num``.

        Parameters
        ----------
        freq : :class:`float`
            The target frequency for which the eigenvalues are calculated. This is the center
            frequency for synthetic averaging.
        num : :class:`int`, optional
            The number of subdivisions in the logarithmic frequency space around the center
            frequency ``freq``.

            - ``0`` (default): Only the eigenvalues for the exact frequency line are returned.
            - Non-zero:

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
        :obj:`numpy.ndarray`
            An array of eigenvalues. If ``num == 0``, the eigenvalues for the single frequency are
            returned. For ``num > 0``, a summed array of eigenvalues across the synthetic frequency
            range is returned.

        Examples
        --------
        >>> import numpy as np
        >>> from acoular import TimeSamples
        >>> from acoular.spectra import PowerSpectra
        >>> np.random.seed(0)
        >>>
        >>> data = np.random.rand(1000, 4)
        >>> ts = TimeSamples(data=data, sample_freq=51200)
        >>> ps = PowerSpectra(source=ts, block_size=128, window='Hamming')
        >>> ps.synthetic_ev(freq=5000, num=5)
        array([0.00048803, 0.0010141 , 0.00234248, 0.00457097])
        >>> ps.synthetic_ev(freq=5000)
        array([0.00022468, 0.0004589 , 0.00088059, 0.00245989])
        """
        f = self.fftfreq()
        if num == 0:
            # single frequency line
            return self.eva[np.searchsorted(f, freq)]
        f1 = np.searchsorted(f, freq * 2.0 ** (-0.5 / num))
        f2 = np.searchsorted(f, freq * 2.0 ** (0.5 / num))
        if f1 == f2:
            return self.eva[f1]
        return np.sum(self.eva[f1:f2], 0)


class PowerSpectraImport(PowerSpectra):
    """
    Provides a dummy class for using pre-calculated CSMs.

    This class does not calculate the CSM. Instead, the user can inject one or multiple existing
    CSMs by setting the :attr:`csm` attribute. This can be useful when algorithms shall be
    evaluated with existing CSMs. The frequency or frequencies contained by the CSM must be set via
    the :attr:`frequencies` attribute. The attr:`num_channels` attributes is determined on the basis
    of the CSM shape. In contrast to the :class:`PowerSpectra` object, the attributes
    :attr:`sample_freq`, :attr:`source`, :attr:`block_size`, :attr:`window`, :attr:`overlap`,
    :attr:`cached`, and :attr:`num_blocks` have no functionality.
    """

    #: The cross-spectral matrix stored in an array of shape ``(n, m, m)`` of complex for ``n``
    #: frequencies and ``m`` channels.
    csm = Property(desc='cross spectral matrix')

    #: The frequencies included in the CSM in ascending order. Accepts list, array, or a single
    #: float value.
    frequencies = Union(None, CArray, Float, desc='frequencies included in the cross-spectral matrix')

    #: Number of time data channels, inferred from the shape of the CSM.
    num_channels = Property(depends_on=['digest'])

    #: :class:`PowerSpectraImport` does not consume time data; source is always ``None``.
    source = Enum(None, desc='PowerSpectraImport cannot consume time data')

    #: Sampling frequency of the signal. Default is ``None``
    sample_freq = Enum(None, desc='sampling frequency')

    #: Block size for FFT, non-functional in this class.
    block_size = Enum(None, desc='PowerSpectraImport does not operate on blocks of time data')

    #: Windowing method, non-functional in this class.
    window = Enum(None, desc='PowerSpectraImport does not perform windowing')

    #: Overlap between blocks, non-functional in this class.
    overlap = Enum(None, desc='PowerSpectraImport does not consume time data')

    #: Caching capability, always disabled.
    cached = Enum(False, desc='PowerSpectraImport has no caching capabilities')

    #: Number of FFT blocks, always ``None``.
    num_blocks = Enum(None, desc='PowerSpectraImport cannot determine the number of blocks')

    # Shadow trait, should not be set directly, for internal use.
    _ind_low = Int(0, desc='index of lowest frequency line')

    # Shadow trait, should not be set directly, for internal use.
    _ind_high = Union(None, Int, desc='index of highest frequency line')

    #: A unique identifier for the spectra, based on its properties. (read-only)
    digest = Property(depends_on=['_csmsum'])

    #: Name of the cache file without extension. (read-only)
    basename = Property(depends_on=['digest'], desc='basename for cache file')

    # Shadow trait for storing the CSM, for internal use only.
    _csm = Union(None, CArray(shape=(None, None, None)), desc='cross spectral matrix')

    # Checksum for the CSM to trigger digest calculation, for internal use only.
    _csmsum = Float()

    def _get_basename(self):
        return 'csm_import_' + self.digest

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_num_channels(self):
        return self.csm.shape[1]

    def _get_csm(self):
        return self._csm

    def _set_csm(self, csm):
        if csm is not None:
            if csm.shape[1] != csm.shape[2]:
                msg = 'The cross spectral matrix must have the following shape: \
                (number of frequencies, num_channels, num_channels)!'
                raise ValueError(msg)
            self._csm = csm
            self._csmsum = np.real(self._csm).sum() + (np.imag(self._csm) ** 2).sum()  # to trigger new digest creation

    @property_depends_on(['digest'])
    def _get_eva(self):
        return self.calc_eva()

    @property_depends_on(['digest'])
    def _get_eve(self):
        return self.calc_eve()

    def fftfreq(self):
        """
        Return the Discrete Fourier Transform sample frequencies.

        The method checks the type of :attr:`frequencies` and returns the corresponding frequency
        array. If :attr:`frequencies` is not defined, a warning is raised.

        Returns
        -------
        :obj:`numpy.ndarray`
            Array containing the frequencies.
        """
        if isinstance(self.frequencies, float):
            return np.array([self.frequencies])
        if isinstance(self.frequencies, np.ndarray):
            return self.frequencies
        return self.frequencies


class PowerSpectraDR( PowerSpectra ):

    #: Number of iterations for CSM approximation.
    n_iter = Int(50,
        desc="number of iterations for CSM approximation")

    # internal identifier
    digest = Property( 
        depends_on = ['time_data.digest', 'calib.digest', 'block_size', 
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
        idiag = np.arange(nc)
        # calculate amplitude of entries
        for ind in range(csm.shape[0]):
            ac = np.absolute(csm[ind])
            # get csm diagonal
            dia = np.diag(ac)
            # get position and value of maximum in diagonal (position maybe not needed)
            # indmaxdia = argmax(dia)
            # max_dia = dia[indmaxdia]
            max_dia = max(dia)
            # get maximum from off-diagonal values
            max_off = (ac-np.diag(dia)).max()
            # calculate 1st diag approximation with correction factor 
            new_dia = dia * (max_off / max_dia)
            # loop to approximate further
            for i in range(self.n_iter):
                # from diag new theoretical mode amplitudes
                mode_amps = new_dia**0.5
                # calculate csm amp estimate
                new_ac = mode_amps[:,np.newaxis] * mode_amps[np.newaxis,:]
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