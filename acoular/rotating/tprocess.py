# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2020, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements processing in the time domain.

.. autosummary::
    :toctree: generated/

    Trigger
    FeatureTrigger
    SpaceModesTransformer
    VirtualRotator
    VirtualRotatorAngle
    AngleTracker
    TrajectoryAnglesFromTrigger
    VirtualRotatorModal
    VirtualRotatorSpatial
    RotationalSpeedDetector
    RotationalSpeedDetector2
    RotationalSpeedDetector3
    RotationalSpeed

"""
from warnings import warn

# imports from other packages
from numpy import (absolute, angle, append, arange, argmax, argmin, argsort,
                   array, ceil, delete, empty, exp, fft, flatnonzero, hstack,
                   inf, integer, isscalar, linspace, matrix, mean, median,
                   newaxis, pad, pi, roll, sign, sinc, sqrt, zeros)
from scipy.interpolate import interp1d
from scipy.signal import butter, decimate, filtfilt
from traits.api import (Any, Bool, Float, Instance, Int, List, ListInt,
                        Property, Trait, Tuple, cached_property)

from acoular.base import SamplesGenerator, TimeOut
from acoular.grids import Grid
from acoular.internal import digest

from .microphones import MicGeomCirc
from .trajectory import AngleTrajectory, TrajectoryAnglesFromTrigger


class Trigger( TimeOut ):
    """
    Class for identifying trigger signals.
    Gets samples from :attr:`source` and stores the trigger samples in :attr:`trigger_data`.
    """
    #: Data source; :class:`~acoular.tprocess.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)
    
    #: Threshold of trigger. The sign is relevant.
    threshold = Float(1.0)
    
    #: Array that holds the indeces of the trigger instants. Read-only.
    trigger_data = Property(depends_on=['source.digest', 'threshold', 'min_trig_dist'])
    
    #: Minimum allowable trigger distance in seconds (think 1/(2*expected_rotational_speed))
    #: Defaults to 1/100 s
    min_trig_dist = Float(0.01)
    
    #: Number of trigger instants per revolution (use every n-th peak)    
    num_trig_per_rev = Int(1)
    
    
    # internal identifier
    digest = Property(depends_on=['source.digest', 'threshold', 'min_trig_dist', 'num_trig_per_rev'])
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_trigger_data(self):
        
        
        num = 2048  # number samples for result-method of source
        pm = sign(self.threshold)
        
        
        # get all samples which surpasse the threshold
        peak_loc = array([], dtype='int')  # all indices which surpass the threshold

        d_samples = 0
        for trigger_signal in self.source.result(num):

            local_trigger = flatnonzero(pm*trigger_signal >= pm*self.threshold)
            if local_trigger.size>0:
                peak_loc = append(peak_loc,local_trigger + d_samples)
            d_samples += num

        # calculate differences between found peaks
        d_peak = peak_loc - append(-self.min_trig_dist*self.sample_freq,peak_loc[:-1])
        
        # only keep those peaks that are far enough from each other
        peak_loc = peak_loc[d_peak > self.min_trig_dist*self.sample_freq]
        if peak_loc.size <= 1:
            warn('Not enough trigger info: %s' % str(peak_loc), Warning, stacklevel = 2)

        return peak_loc[::self.num_trig_per_rev]
    






class FeatureTrigger(Trigger):
    """
    Class for identifying trigger signals.
    Gets samples from :attr:`source` and stores the trigger samples in :meth:`trigger_data`.
    
    The algorithm searches for peaks which are above/below a signed threshold.
    A estimate for approximative length of one revolution is found via the greatest
    number of samples between the adjacent peaks.
    The algorithm then defines hunks as percentages of the estimated length of one revolution.
    If there are multiple peaks within one hunk, the algorithm just takes one of them 
    into account (e.g. the first peak, the peak with extremum value, ...).
    In the end, the algorithm checks if the found peak locations result in rpm that don't
    vary too much.
    """
    #: Data source; :class:`~acoular.tprocess.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)
    
    #: Threshold of trigger. Has different meanings for different 
    #: :attr:`~acoular.tprocess.Trigger.trigger_type`. The sign is relevant.
    #: If a sample of the signal is above/below the positive/negative threshold, 
    #: it is assumed to be a peak.
    #: Default is None, in which case a first estimate is used: The threshold
    #: is assumed to be 75% of the max/min difference between all extremums and the 
    #: mean value of the trigger signal. E.g: the mean value is 0 and there are positive
    #: extremums at 400 and negative extremums at -800. Then the estimated threshold would be 
    #: 0.75 * -800 = -600.
    threshold = Float(None)
    
    #: Maximum allowable variation of length of each revolution duration. Default is
    #: 2%. A warning is thrown, if any revolution length surpasses this value:
    #: abs(durationEachRev - meanDuration) > 0.02 * meanDuration
    max_variation_of_duration = Float(0.02)
    
    #: Defines the length of hunks via lenHunk = hunk_length * maxOncePerRevDuration.
    #: If there are multiple peaks within lenHunk, then the algorithm will 
    #: cancel all but one out (see :attr:`~acoular.tprocess.Trigger.multiple_peaks_in_hunk`).
    #: Default is to 0.1.
    hunk_length = Float(0.1)
    
    # offset, factor
    correction = Tuple((0.0, 1.0))
    
    #: Type of trigger.
    #:
    #: 'dirac': a single puls is assumed (sign of  
    #: :attr:`~acoular.tprocess.Trigger.trigger_type` is important).
    #: Sample will trigger if its value is above/below the pos/neg threshold.
    #: 
    #: 'rect' : repeating rectangular functions. Only every second 
    #: edge is assumed to be a trigger. The sign of 
    #: :attr:`~acoular.tprocess.Trigger.trigger_type` gives information
    #: on which edge should be used (+ for rising edge, - for falling edge).
    #: Sample will trigger if the difference between its value and its predecessors value
    #: is above/below the pos/neg threshold.
    #: 
    #: Default is 'dirac'.
    trigger_type = Trait('dirac', 'rect')
    
    #: Identifier which peak to consider, if there are multiple peaks in one hunk
    #: (see :attr:`~acoular.tprocess.Trigger.hunk_length`). Default is to 'extremum', 
    #: in which case the extremal peak (maximum if threshold > 0, minimum if threshold < 0) is considered.
    multiple_peaks_in_hunk = Trait('extremum', 'first')
    
    #: Tuple consisting of 3 entries: 
    #: 
    #: 1.: -Vector with the sample indices of the 1/Rev trigger samples
    #: 
    #: 2.: -maximum of number of samples between adjacent trigger samples
    #: 
    #: 3.: -minimum of number of samples between adjacent trigger samples
    trigger_data = Property(depends_on=['source.digest', 'threshold', 'max_variation_of_duration', \
                                        'hunk_length', 'trigger_type', 'multiple_peaks_in_hunk'])
    
    # internal identifier
    digest = Property(depends_on=['source.digest', 'threshold', 'max_variation_of_duration', \
                                        'hunk_length', 'trigger_type', 'multiple_peaks_in_hunk'])
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_trigger_data(self):
        self._check_trigger_existence()
        triggerFunc = {'dirac' : self._trigger_dirac,
                       'rect' : self._trigger_rect}[self.trigger_type]
        nSamples = 2048  # number samples for result-method of source
        threshold = self._threshold(nSamples)
        
        # get all samples which surpasse the threshold
        peakLoc = array([], dtype='int')  # all indices which surpasse the threshold
        triggerData = array([])
        x0 = []
        dSamples = 0
        offset, factor = self.correction
        for triggerSignal in self.source.result(nSamples):
            triggerSignal = offset + triggerSignal*factor
            localTrigger = flatnonzero(triggerFunc(x0, triggerSignal, threshold))
            if not len(localTrigger) == 0:
                peakLoc = append(peakLoc, localTrigger + dSamples)
                triggerData = append(triggerData, triggerSignal[localTrigger])
            dSamples += nSamples
            x0 = triggerSignal[-1]
        if len(peakLoc) <= 1:
            raise Exception('Not enough trigger info. Check *threshold* sign and value!')

        peakDist = peakLoc[1:] - peakLoc[:-1]
        maxPeakDist = max(peakDist)  # approximate distance between the revolutions
        
        # if there are hunks which contain multiple peaks -> check for each hunk, 
        # which peak is the correct one -> delete the other one.
        # if there are no multiple peaks in any hunk left -> leave the while 
        # loop and continue with program
        multiplePeaksWithinHunk = flatnonzero(peakDist < self.hunk_length * maxPeakDist)
        while len(multiplePeaksWithinHunk) > 0:
            peakLocHelp = multiplePeaksWithinHunk[0]
            indHelp = [peakLocHelp, peakLocHelp + 1]
            if self.multiple_peaks_in_hunk == 'extremum':
                values = triggerData[indHelp]
                deleteInd = indHelp[argmin(abs(values))]
            elif self.multiple_peaks_in_hunk == 'first':
                deleteInd = indHelp[1]
            peakLoc = delete(peakLoc, deleteInd)
            triggerData = delete(triggerData, deleteInd)
            peakDist = peakLoc[1:] - peakLoc[:-1]
            multiplePeaksWithinHunk = flatnonzero(peakDist < self.hunk_length * maxPeakDist)
        
        # check whether distances between peaks are evenly distributed
        meanDist = mean(peakDist)
        diffDist = abs(peakDist - meanDist)
        faultyInd = flatnonzero(diffDist > self.max_variation_of_duration * meanDist)
        if faultyInd.size != 0:
            warn('In Trigger-Identification: The distances between the peaks '
                 '(and therefor the lengths of the revolutions) vary too much '
                 '(check samples %s).' % str(peakLoc[faultyInd]), Warning, stacklevel = 2)
        return peakLoc, max(peakDist), min(peakDist)
    
    def _trigger_dirac(self, x0, x, threshold):
        # x0 not needed here, but needed in _trigger_rect
        return self._trigger_value_comp(x, threshold)
    
    def _trigger_rect(self, x0, x, threshold):
        # x0 stores the last value of the the last generator cycle
        xNew = append(x0, x)
       #indPeakHunk = abs(xNew[1:] - xNew[:-1]) > abs(threshold)  # with this line: every edge would be located
        indPeakHunk = self._trigger_value_comp(xNew[1:] - xNew[:-1], threshold)
        return indPeakHunk
    
    def _trigger_value_comp(self, triggerData, threshold):
        if threshold > 0.0:
            indPeaks= triggerData > threshold
        else:
            indPeaks = triggerData < threshold
        return indPeaks
    
    def _threshold(self, nSamples):
        if self.threshold == None:  # take a guessed threshold
            # get max and min values of whole trigger signal
            maxVal = -inf
            minVal = inf
            meanVal = 0
            cntMean = 0
            for triggerData in self.source.result(nSamples):
                maxVal = max(maxVal, triggerData.max())
                minVal = min(minVal, triggerData.min())
                meanVal += triggerData.mean()
                cntMean += 1
            meanVal /= cntMean
            
            # get 75% of maximum absolute value of trigger signal
            maxTriggerHelp = [minVal, maxVal] - meanVal
            argInd = argmax(abs(maxTriggerHelp))
            thresh = maxTriggerHelp[argInd] * 0.75  # 0.75 for 75% of max trigger signal
            warn('No threshold was passed. An estimated threshold of %s is assumed.' % thresh, Warning, stacklevel = 2)
        else:  # take user defined  threshold
            thresh = self.threshold
        return thresh
    
    def _check_trigger_existence(self):
        nChannels = self.source.numchannels
        if not nChannels == 1:
            raise Exception('Trigger signal must consist of ONE channel, instead %s channels are given!' % nChannels)
        return 0



class SpaceModesTransformer ( TimeOut ):
    """
    Class to transform mic signals into mode domain or vice versa via spatial fft.
    """
    
    #: Channels to be used and their order, so that neighboring mics are 
    #: channel neighbors as well (last is neigbor to first). 
    #: If left emtpy, mics are just used in their current order. 
    channel_order = ListInt([],
        desc="list of mic channels in correct order")

    #: Direction of transformation:
    #: If 'False' (default), transform from spatial to mode domain. 
    #: If 'True', transform from mode to spatial domain
    inverse = Bool(False,
                   desc = 'direction of transform')

    #: Array of modes that are calculated. Read-only.
    #: If source input is from mode domain, order is assumed to be 
    #: [0, 1, 2, ... , M/2-1, -M/2, ..., -2, -1]
    modes = Property( depends_on = ['source.digest', 'numchannels'])

    #: Number of channels/modes, is set automatically.
    numchannels = Property(depends_on = ['channel_order', \
        'source.sigest'], desc="number of valid channels")

    # internal identifier
    digest = Property( depends_on = ['source.digest', 'channel_order'])

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_numchannels( self ):
        if self.channel_order:
            return len(self.channel_order)
        else:
            return self.source.numchannels


    @cached_property
    def _get_modes( self ):
        if not self.inverse:
            nc = self.numchannels
        else:
            nc = self.source.numchannels
        return fft.fftfreq(nc, 1/nc).astype(int)
        
    def result(self, num):
        """ 
        Python generator: yields transformed result.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`). 
            The last block may be shorter than num.
        """
        
        # set up fft or ifft      
        if not self.inverse: # transform space -> modes
            fft_func = fft.fft
            
            # index slice: all channels
            channels_out = slice(None, None, None)
            if self.channel_order: # use index array for resorting neighbors
                channels_in = array(self.channel_order)
            else: # otherwise use all channels in that order
                channels_in = slice(None, None, None)
        
        else: # transform modes -> space
            fft_func = fft.ifft
            # index slice: all modes 
            channels_in = slice(None, None, None)
            if self.channel_order: # resort output channels to original order
                channels_out = argsort(array(self.channel_order))
            else: # otherwise use all channels in current order
                channels_out = slice(None, None, None)
            
        # the actual generator
        for block in self.source.result(num):
            yield fft_func(block[:,channels_in], None, axis=1, norm="ortho")[:,channels_out]





class VirtualRotator ( TimeOut ):
    """
    Class that provides the facilities to virtually rotate a microphone 
    array depending on a given constant rpm.
    
    The measured time data is recalculated using interpolation 
    between samples such that the resulting data
    represents that of a virtually rotating array.
    """

    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 'mpos.digest', 'rpm','interpolation', '__class__' ])
    
    # basename for cache file
    basename = Property( depends_on = 'source.digest', 
        desc="basename for cache file" )
    
    #: Type of interpolation between microphones to be used.
    interpolation = Trait('linear','sinc',
                          desc='type of interpolation between mics')
    
    #: :class:`~.microphones.MicGeomCirc` object that provides the 
    #: microphone locations
    mpos = Trait(MicGeomCirc, 
        desc="circular microphone geometry")
        
    #: Number of revolutions per minute; use negative values 
    #: for clockwise rotation, defaults to 0.
    rpm = Float(0.0,
        desc="revolutions per minute of the virtual array, negative values for clockwise rotation")
    # maybe change to rad/s ?

    @cached_property
    def _get_basename( self ):
        if 'basename' in self.source.all_trait_names():
            return self.source.basename
        else: 
            return self.source.__class__.__name__ + self.source.digest
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    
    
    def result(self, num):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        if self.rpm == 0.0: # if it doesn't rotate, do nothing...
            print("No virtual array rotation!")
            for block in self.source.result(num):
            # effectively no processing
                yield block
                
        else:

            mpos = self.mpos        
            
            num_rings = len(mpos.ringlist) 
            num_rmics = [len(mpos.ringlist[a].mics) for a in range(num_rings)]
    
            j = 0
            
            # number of mics in ring of corresponding mic indices
            num_rmics_long = []

            # start mic for every ring, needed for later ringwise re-sorting
            mics_base = []

            # relative indices for mics in rings
            ring_base = []
            
            imic = 0

            for nmics in num_rmics: 
                num_rmics_long += [nmics]*nmics
                mics_base += [imic]*nmics
                ring_base += list(range(nmics))
                imic += nmics
                
            mics_base = array(mics_base)
            ring_base = array(ring_base)
            
            # predefine constants and indices for faster use in loop...
            rpm2imic = self.rpm * 6.0 / self.sample_freq
            mm_index = mpos.mic_index[:]
            mmm_index = mpos.indices[mm_index]

            newdata = empty((num, len(mpos.mic_index))) 
            
            for block in self.source.result(num):
                ns = block.shape[0] # number of samples in block
                
                isamples = arange(j, j+ns)
                
                
                
                if self.interpolation == 'sinc':
                    # this implementation has to be sped up, current one is inefficient...
                    # walk through the rings:
                    for iring in range(num_rings):
                        N = num_rmics[iring]
                        ringmics = mpos.ringlist[iring].mics # indices of mics in current ring
                        rel_ind = arange(N)-N//2 # how should the relative weighting be done 
                        
                        imic0 = ((rpm2imic * isamples)%360.0 ) * N / 360.0
                        imic0int  = imic0.astype(integer) 
                        imic0frac = imic0 - imic0int
                        
                        # kernel contains weights for neighboring mics (use all mics in ring)
                        # weights depend on fraction between mics
                        kernel = sinc(imic0frac[:,newaxis] - rel_ind[newaxis, : ]) 
                        # roll the kernel by full mics according to angle:
                        for i in range(ns):
                            kernel[i]=roll(kernel[i],imic0int[i]-N//2)
                        
                        # get data of this ring from block
                        p = block[:,ringmics]
                        
                        # go through all mics, roll kernel accordingly, sum weighted signals
                        for n in range(N):
                            # fill newdata with sinc interpolated data
                            newdata[:ns,ringmics[n]] = sum( p * roll(kernel, n, axis=1), axis=1)
                
                else: #if self.interpolation == 'linear':
                    # calculate how many mics further the rotating array is now...
                    imic0 = array(( ( rpm2imic * matrix(isamples) )%360.0 ).T * \
                                                    matrix(num_rmics_long) / 360.0)
                    imic0int  = imic0.astype(integer) 
                    imic0frac = imic0 - imic0int# = imic0 % 1 # fractional part
    
                    # calculate indices of mics between which to interpolate...    
                    rmi1 = mics_base + (ring_base+imic0int) % array(num_rmics_long)
                    rmi2 = mics_base+(ring_base+imic0int+1) % array(num_rmics_long)
    
                    ns_index = arange(ns)[:, newaxis] 
                    
                    # fill newdata with linearly interpolated data...
                    """
                    newdata[:, mmm_index] = \
                             (1 - imic0frac) * block[ ns_index, mm_index[rmi1] ] \
                               + (imic0frac) * block[ ns_index, mm_index[rmi2] ]
                    """
                    newdata[:ns, mmm_index] = \
                             (1 - imic0frac) * block[ ns_index, mmm_index[rmi1] ] \
                               + (imic0frac) * block[ ns_index, mmm_index[rmi2] ]
                    #"""
                j += ns
                
                yield newdata[:ns]




class VirtualRotatorAngle ( VirtualRotator ):
    """
    Class that provides the facilities to virtually rotate a microphone 
    array depending on sampled angles.
    
    The measured time data is recalculated using interpolation 
    between samples such that the resulting data
    represents that of a virtually rotating array.
    
    The following assumptions are made:
       * The revolutions per minute vary little enough too use their average
         for calculating the retarded time at the microphones
       * the center of the focus grid can be used as reference for calculating 
         the retarded time (instead of each grid point)
    """

    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 
                      'angle.digest',
                      'mpos.digest',
                      'grid.z',
                      'phi_trigger',
                      'c',
                      'interpolation',
                      '__class__' ])
    
    #: Grid-derived object that provides the grid locations.
    grid = Trait(Grid, 
        desc="beamforming grid")   

    #: Delay for rotation. If set, overrides delay clac'ed from grid position. 
    #TODO: Make this ring-specific (as array or similar)    
    delay = Any(-1.)

    #: Type of interpolation between microphones to be used.
    interpolation = Trait('linear','sinc',
                          desc='type of interpolation between mics')
    
    #: Speed of sound, defaults to 343 m/s.
    c = Float(343., 
        desc="speed of sound")

    #: Angular position of the trigger (if only one trigger is used);
    #: defaults to 0. Use this parameter to align the sources to the
    #: rotating geometry.
    phi_trigger = Float(0.,
                        desc = 'angle of first trigger')
 
    #: :class:`AngleTrajectory` instance, that delivers the angles
    #: for virtual rotation.
    angle = Trait(AngleTrajectory,
                  desc = 'interpolated angles')

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    def result(self, num):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """

        self.rpm = self.angle.rpm
        if self.rpm == 0.0: # if it doesn't rotate, do nothing...
            print("No virtual array rotation!")
            for block in self.source.result(num):
            # effectively no processing
                yield block
                
        else:
            
            mpos = self.mpos  
            
            # speed of sound in sample units (meter/sample)
            c = self.c/self.sample_freq
            
            # number of mic rings
            num_rings = len(mpos.ringlist) 
            
            # list of number of mics in rings
            num_rmics = [len(mpos.ringlist[a].mics) for a in range(num_rings)]
    
            # list of distances btw mic rings and grid center
            r_rings = array([mpos.ringlist[a].r for a in range(num_rings)])
            z_rings = array([mpos.ringlist[a].z for a in range(num_rings)])
            
            
            if self.delay == -1:            
                dr_ring = sqrt( r_rings**2 + (self.grid.z - z_rings)**2 ) 
                #dr_ring *= 0 # no delay between vent. rot. and array response
                # sound travel time from grid center to ring
                dt_ring = dr_ring / c 
            else:
                dt_ring = array(self.delay).reshape(-1,)*self.sample_freq
            
            
            # starting angles of rings, also adjust for trigger position
            phi_start = self.angle.location(self.angle.startind - dt_ring)
            phi_start -= self.phi_trigger
            
            
            # number of mics in ring of corresponding mic indices
            num_rmics_long = []

            # start mic for every ring, needed for later ringwise re-sorting
            mics_base = []

            # relative indices for mics in rings
            ring_base = []
            
            # id of ring each mic belongs to            
            ring_id_long = []
            
            # calculate all "help" arrays needed
            imic = 0
            iring = 0
            for nmics in num_rmics: 
                num_rmics_long += [nmics]*nmics
                mics_base += [imic]*nmics
                ring_base += list(range(nmics))
                ring_id_long += [iring]*nmics
                imic += nmics
                iring += 1
                
                
            mics_base = array(mics_base)
            ring_base = array(ring_base)
            num_rmics_long = array(num_rmics_long)
            
            # predefine constants and indices for faster use in loop...
            mm_index = mpos.mic_index[:]
            mmm_index = mpos.indices[mm_index]

            #adata = self.angle.result().next()
            iangle = arange(num)
            
            newdata = empty((num, len(mpos.mic_index))) 

            for block in self.source.result(num):
                
                ns = block.shape[0] # number of samples in block
                
                # get the angles at the retarded time for all mic rings                
                new_angles = (array([self.angle.location(iangle[:ns]-dt) for dt in dt_ring])  \
                                    - phi_start.reshape(num_rings,1) ).T % 360.0
                
                iangle += ns
                
                if self.interpolation == 'sinc':
                    # this implementation has to be sped up, current one is inefficient...
                    # walk through the rings:
                    for iring in range(num_rings):
                        N = num_rmics[iring]
                        ringmics = mpos.ringlist[iring].mics # indices of mics in current ring
                        rel_ind = arange(N)-N//2 # how should the relative weighting be done 
                        
                        imic0 = new_angles[:,iring] / 360.0 * N
                        imic0int  = imic0.astype(integer) 
                        imic0frac = imic0 - imic0int
                        
                        # kernel contains weights for neighboring mics (use all mics in ring)
                        # weights depend on fraction between mics
                        kernel = sinc(imic0frac[:,newaxis] - rel_ind[newaxis, : ]) 
                        # roll the kernel by full mics according to angle:
                        for i in range(ns):
                            kernel[i]=roll(kernel[i],imic0int[i]-N//2)
                        
                        # get data of this ring from block
                        p = block[:,ringmics]
                        
                        # go through all mics, roll kernel accordingly, sum weighted signals
                        for n in range(N):
                            # fill newdata with sinc interpolated data
                            #TODO: roll here is probably buggy, as it spills / shifts one time sample
                            newdata[:ns,ringmics[n]] = sum( p * roll(kernel, n, axis=1), axis=1)

                else: #if self.interpolation == 'linear':
                    # calculate how many mics further the rotating array is now...
                    # ring_id_long to put  new_angles in block form
                    imic0 = new_angles[:,ring_id_long] / 360.0 * num_rmics_long
                    imic0int  = imic0.astype(integer) 
                    imic0frac = imic0 - imic0int# = imic0 % 1 # fractional part
    
                    # calculate indices of mics between which to interpolate...    
                    rmi1 = mics_base + (ring_base+imic0int) % num_rmics_long
                    rmi2 = mics_base+(ring_base+imic0int+1) % num_rmics_long
        
                    # fill newdata with linearly interpolated data...                
                    newdata[:ns, mmm_index] = \
                             (1 - imic0frac) * \
                             array([block[a,:][mmm_index[rmi1[a,:]]] for a in range(ns)]) \
                             + (imic0frac) * \
                             array([block[a,:][mmm_index[rmi2[a,:]]] for a in range(ns)])
                
                yield newdata[:ns]




class VirtualRotatorModal ( TimeOut ):
    """
    Class that provides the facilities to virtually rotate modes
    depending on a given constant rps.
    
    The measured time data is recalculated using modal decomposition.
    """
    
    # Data source; :class:`~acoular.tprocess.Time` or derived object.
    #source = Trait(SpaceModesTransformer)
    # source can be any TimeInOut, but is expected to be in Spatial FFT domain and order
        
    #: Rotational speed in rps. Positive, if rotation is around positive z-axis sense,
    #: which means from x to y axis.
    rotational_speed = Any(0.0)

    
    #: Average time delay between grid and mic planes
    delay = Float(0.0)
    
    
    # internal identifier
    rotspeed_digest = Property(depends_on = ['rotational_speed'])
    
    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 'rotspeed_digest', 'delay', '__class__' ])
    
    @cached_property
    def _get_rotspeed_digest( self ):
        if isscalar(self.rotational_speed):
            return self.rotational_speed
        else:
            return self.rotational_speed.digest
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    
    
    def result(self, num):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        
        if not self.rotational_speed: # rps==0
            for block in self.source.result(num):
                # effectively no processing
                yield block
        else:
            
            # list of modes
            modes = fft.fftfreq(self.numchannels, 1/self.numchannels).astype(int)
            
          
            # constant rotation 
            if isscalar(self.rotational_speed):
                # initialize time vector
                tvec = arange(num)
                # calculate rotational speed in revs per sample
                revs_per_sample = self.rotational_speed / self.sample_freq
                            
                for block in self.source.result(num):
                    # get actual size of block
                    bsize = block.shape[0]
                    
                    # phase shift of time signal
                    phase_shift = exp(2j*pi * revs_per_sample * 
                                      modes[newaxis, :] * 
                                      tvec[:bsize, newaxis])
                    
                    yield block * phase_shift
                    tvec += num
           
            elif isinstance(self.rotational_speed, TrajectoryAnglesFromTrigger): 
                sample_delay = self.delay*self.sample_freq
                tvec = arange(num)
                theta0 = self.rotational_speed.location(self.rotational_speed.first_trigger-sample_delay)#-self.rotational_speed.location(0)
                for block in self.source.result(num):
                    # get actual size of block
                    bsize = block.shape[0]
                    # get rotation angles of array
                    theta = self.rotational_speed.location(tvec[:bsize]-sample_delay) - theta0
                    phase_shift = exp(1j * theta[:,newaxis] * modes[newaxis, :] )
                    yield block * phase_shift
                    tvec += num
           
            elif isinstance(self.rotational_speed, TimeOut):
                
                for theta, block in zip(self.rotational_speed.result(num), self.source.result(num)):
                    phase_shift = exp(1j * theta[:,newaxis] * modes[newaxis, :] )
                    yield block * phase_shift
            else:
                raise Exception('Unexpected data type (not constant or TimeInOut!')
            


class VirtualRotatorSpatial ( TimeOut ):
    """
    Class that provides the facilities to virtually rotate a microphone 
    array depending on sampled angles.
    
    The measured time data is recalculated using interpolation 
    between samples such that the resulting data
    represents that of a virtually rotating array.
    
    The following assumptions are made:
       * The revolutions per minute vary little enough too use their average
         for calculating the retarded time at the microphones
       * the center of the focus grid can be used as reference for calculating 
         the retarded time (instead of each grid point)
    """

    channel_order = ListInt([],
        desc="list of mic channels in correct order")

    #: Rotational speed in rps. Positive, if rotation is around positive z-axis sense,
    #: which means from x to y axis.
    rotational_speed = Any(0.0)

    
    #: Average time delay between grid and mic planes
    delay = Float(0.0)
    
    
    # internal identifier
    rotspeed_digest = Property()
    
    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 'rotspeed_digest', 'delay', 'channel_order', '__class__' ])
    
    
    def _get_rotspeed_digest( self ):
        if isscalar(self.rotational_speed):
            return self.rotational_speed
        else:
            return self.rotational_speed.digest
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    
    
    def result(self, num):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """

        if not self.rotational_speed: # rps==0
            for block in self.source.result(num):
                # effectively no processing
                yield block
        else:
            channel_order = self.channel_order
            num_mics = len(channel_order)
            
            # constant rotation 
            # if isscalar(self.rotational_speed):
            #     # initialize time vector
            #     tvec = arange(num)
            #     # calculate rotational speed in revs per sample
            #     revs_per_sample = self.rotational_speed / self.sample_freq
                            
            #     for block in self.source.result(num):
            #         # get actual size of block
            #         bsize = block.shape[0]
                    
            #         # phase shift of time signal
            #         phase_shift = exp(2j*pi * revs_per_sample * 
            #                           modes[newaxis, :] * 
            #                           tvec[:bsize, newaxis])
                    
            #         yield block * phase_shift
            #         tvec += num
           
            if isinstance(self.rotational_speed, TrajectoryAnglesFromTrigger): 
                sample_delay = self.delay*self.sample_freq
                tvec = arange(num)
                theta0 = self.rotational_speed.location(self.rotational_speed.first_trigger-sample_delay)
                prot = zeros((num,num_mics))
                for block in self.source.result(num):
                    # get actual size of block
                    bsize = block.shape[0]
                    
                    # get rotation angles of array
                    theta = self.rotational_speed.location(tvec[:bsize]-sample_delay) - theta0
                    
                    relpos = theta/2/pi*num_mics
                    i = 0
                    for rel,irel in zip(relpos,relpos.astype(int)):
                        mics_lower = roll(channel_order,-irel)
                        mics_upper = roll(channel_order,-irel-1)
                    
                        w_upper = rel-irel
                        w_lower = 1-w_upper
                        prot[i,channel_order] = w_lower*block[i,mics_lower]+\
                                                w_upper*block[i,mics_upper]
                        i += 1
                    yield prot[:bsize]
                    tvec += num
           
            # elif isinstance(self.rotational_speed, TimeInOut):
                
            #     for theta, block in zip(self.rotational_speed.result(num), self.source.result(num)):
            #         phase_shift = exp(1j * theta[:,newaxis] * modes[newaxis, :] )
            #         yield block * phase_shift
            else:
                raise Exception('Unexpected data type (not constant or TimeOut!')
                


class RotationalSpeedDetector( TimeOut ):
    """
    Class to calculate rotational speed for each sample. 
    This implementation is preliminary, interface will change in the future.
    """
    
    #: Maximum delta f of returned spectra in Hz
    df = Float(0.1)
    
    #: number of samples in one block for rotation detection
    block_size = Int(10000)
    
    #: Upper rotational frequency limit that should be resolved
    fmax = Float(100.0)
    
    #: Tuple with (lower, upper) frequency for band pass prior to cross correlation
    band_pass = Tuple((2500.0, 0.0))
    
    #: Result will be yielded for every `sample_step`-th sample. 
    #: If >1, this will result in a decreased `sample_freq`.
    sample_step = Int(10000)

    #: List of delta m (=b-a) to consider
    dm_list = List( [1,3,5,7,9] )#
    
    #: maximum cross mode to consider (max. nmodes/2 )
    m_max = Int(26)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Property( depends_on = ['source.sample_freq', 
                                          'sample_step'])
    
    #: Discrete resolved rotational frequencies
    rps_freqs = Property( depends_on = ['source.sample_freq', 'fmax',
                                        '_ns_new'])
    
    # number of samples in intermediate fft blocks
    _ns_new = Property( depends_on = ['source.sample_freq', 'df',
                                      'block_size', 'dm_list'])
    
    # max. index of data to consider (corresponding to fmax)
    _maxfreqind = Property( depends_on = ['source.sample_freq', 
                                          'fmax', '_ns_new'])
    
    @cached_property 
    def _get_sample_freq( self ):
        return self.source.sample_freq/self.sample_step
    
    @cached_property 
    def _get__ns_new( self ):
        # number of samples
        return max( int(self.source.sample_freq/self.df)+1, 
                        self.block_size * max(self.dm_list) )

    @cached_property 
    def _get__maxfreqind( self ):
        return int(self.fmax * self._ns_new / self.source.sample_freq)
    
    @cached_property 
    def _get_rps_freqs( self ):
        fftfreqs = fft.fftfreq(self._ns_new, 1./self.source.sample_freq)
        return hstack((fftfreqs[-self._maxfreqind:], fftfreqs[:self._maxfreqind]))     
        
    
    
    def result( self, num ):
        """
        see "12_frot_detection_weight_and_combine.py" for tests
        assumes that input is in mode order
        """

        bs = self.block_size
        step = self.sample_step
        modelist = self.source.modes
        nmodes = len(modelist)
        
        f_low, f_high = self.band_pass
                                ### band pass filter ##############################
        if f_low>0:  
            b, a = butter(8, f_low/self.source.sample_freq*2, btype='highpass')
        #TODO: f_high currently not very nicely implemented
        if f_high>0:
            bh, ah = butter(8, f_high/self.source.sample_freq*2, btype='lowpass')
        #    data = filtfilt(b, a, data, axis=0)
        
        mf_ind = self._maxfreqind
        
        # allocate arrays that hold the needed data
        pmfpart = zeros((2*mf_ind, self.m_max+1), dtype=complex)
        result_full = zeros((num, 2*mf_ind))
        
        result_step = zeros(2*mf_ind)
        
        pmt = zeros((self.block_size+num,nmodes), dtype=complex)
        
        ind = 0
        ind_out = 0
        
        for block in self.source.result( num ):
            #print(ind,end=',')
            ns = block.shape[0]
            ind1 = ind+ns
            
            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]
            
            ind = ind1
                        
            if ind >= bs:
                # highpass
                if f_low>0:  
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                    if f_high>0:
                        pmt_temp = filtfilt(bh, ah, pmt_temp, axis=0)
                else:
                    pmt_temp = pmt[:bs]

                for dm in self.dm_list:
            
                    # calculate cross-mode product, only need half of the modes, since everything is mirrored
                    # shift left/right, so that all mode combinations are present (and none doubled) in the resulting array
                    ind_m_b = roll(modelist[::-1], int(0.5-dm/2))[:nmodes//2]
                    ind_m_a = roll(modelist      , int(0.0+dm/2))[:nmodes//2]
                    pmtprod = pmt_temp[:,ind_m_a] * pmt_temp[:,ind_m_b]
                
                    # ns_full = 2**int(round(log2(self._ns_new/dm))) # base 2 numbers too "rough" for interpolation between modes, otherwise, fft would be about 3 times as fast!
                    ns_full = int(self._ns_new/dm)
                    pmf = fft.fft(pad(pmtprod,[(0, ns_full-bs),(0, 0)]), None, 0, norm="ortho")
                
                    # fill array with only part of the data
                    pmfpart[:mf_ind, :] = pmf[-mf_ind:, :self.m_max+1]
                    pmfpart[mf_ind:, :] = pmf[ :mf_ind, :self.m_max+1]
                
                    # weighting array
                    # derivative of angle -> small if rotational velocity is found
                    pmfanglediff=sum(abs(angle(pmfpart[:,1:] * pmfpart[:,:-1].conjugate())), axis=1)
                    
                    #get result for this mode
                    apmf = absolute(pmfpart)
                    result_dm = 1/pmfanglediff * sum(apmf/median(apmf,axis=0), axis=1)
                    #result_dm = 1/pmfanglediff * sum(absolute(pmfpart), axis=1)
                    
                    # average to get overall result
                    #result_step *= result_dm/result_dm.max()
                    result_step += result_dm#/result_dm.max()
                result_full[ind_out,:] = result_step#/result_step.max()
                result_step[:] = zeros(2*mf_ind)
                ind_out += 1
                if ind_out >= num:
                    yield result_full
                    ind_out = 0
            
                # shift rest of data according to step size
                ind -= step#self.block_size
                if ind > 0:
                    pmt[:ind] = pmt[step:step+ind]
                
        if ind_out > 0:
            yield result_full[:ind_out]
        
class RotationalSpeedDetector2( RotationalSpeedDetector ):
    """
    Class to calculate rotational speed for each sample. 
    This implementation is preliminary, interface will change in the future.
    This version does spline interpolation instead of FFT with zero-padding
    
    --> This is way faster, but not as accurate as RSD(1)
    --> It is not as as accurate as RSD3
    --> RSD2 not to be used for now
    """       
        
    def result( self, num ):
        """
        see "12_frot_detection_weight_and_combine.py" for tests
        assumes that input is in mode order
        """
        
        bs = self.block_size
        step = self.sample_step
        modelist = self.source.modes
        nmodes = len(modelist)
        fmax= self.fmax
        
        f_low, f_high = self.band_pass
                                ### band pass filter ##############################
        if f_low>0:  
            b, a = butter(8, f_low/self.source.sample_freq*2, btype='highpass')
    
        
        nfreqs = 2 * int(fmax / self.df)
        ffreqs = linspace(-fmax,fmax,nfreqs,endpoint=False)

        
        # allocate arrays that hold the needed data
        # largest array that is needed
        maxmaxind = int(fmax / self.source.sample_freq * bs * absolute(self.dm_list).max())
        pmfpart = zeros((2*maxmaxind, self.m_max+1), dtype=complex)
        
        result_full = zeros((num, nfreqs))
        
        #result_step = ones(nfreqs)
        result_step = zeros(nfreqs)
        
        pmt = zeros((self.block_size+num,nmodes), dtype=complex)
        
        ind = 0
        ind_out = 0
        
        for block in self.source.result( num ):
            #print(ind,end=',')
            ns = block.shape[0]
            ind1 = ind+ns
            
            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]
            
            ind = ind1
                        
            if ind >= self.block_size:
                # highpass
                if f_low>0:  
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                else:
                    pmt_temp = pmt[:bs]

                for dm in self.dm_list:
                    
                    # get index that corresponds to frequency for this mode
                    maxind = int(fmax / self.source.sample_freq * bs * abs(dm))
                    # calculate cross-mode product, only need half of the modes, since everything is mirrored
                    # shift left/right, so that all mode combinations are present (and none doubled) in the resulting array
                    ind_m_b = roll(modelist[::-1], int(0.5-dm/2))[:nmodes//2]
                    ind_m_a = roll(modelist      , int(0.0+dm/2))[:nmodes//2]
                    pmtprod = pmt_temp[:,ind_m_a] * pmt_temp[:,ind_m_b]
                
                    #ns_full = int(self._ns_new/dm)
                    #pmf = fft.fft(pad(pmtprod,[(0, ns_full-bs),(0, 0)]), None, 0, norm="ortho")
                    pmf = fft.fft(pmtprod, None, 0, norm="ortho")
                    
                    
                    # fill array with only part of the data
                    pmfpart[:maxind, :] = pmf[-maxind:, :self.m_max+1]
                    pmfpart[maxind:2*maxind, :] = pmf[ :maxind, :self.m_max+1]
                
                    # weighting array
                    # derivative of angle -> small if rotational velocity is found
                    pmfanglediff=-sum(abs(angle(pmfpart[:2*maxind,1:] * pmfpart[:2*maxind,:-1].conjugate())), axis=1)
                    # most values will not correspond to a rotation, so subtract average
                    pmfanglediff-=pmfanglediff.min()*1.1
                    # normalize max. to 1
                    pmfanglediff/=pmfanglediff.max()
                    
                    #get result for this mode
                    #result_dm = pmfanglediff**4 * absolute(prod(pmfpart[:2*maxind], axis=1))
                    #result_dm = pmfanglediff**4 * absolute(prod(pmfpart[:2*maxind], axis=1))
                    result_dm = pmfanglediff * sum(absolute(pmfpart[:2*maxind]), axis=1)
                    #               result_dm/result_dm.max()  
                    finter = interp1d(linspace(-fmax,fmax,2*maxind,endpoint=False),
                                      result_dm,
                                      kind = 'nearest',
                                      fill_value="extrapolate")
                    result_dm_inter = finter(ffreqs)
                    # average to get overall result
                    #result_step *= result_dm_inter/result_dm_inter.max()
                    result_step += result_dm_inter#/result_dm_inter.max()
                result_full[ind_out,:] = result_step#/result_step.max()
                #result_full[ind_out,:] = (result_step/result_step.max())**40
                result_step[:] = 0
                ind_out += 1
                if ind_out >= num:
                    yield result_full
                    ind_out = 0
            
                # shift rest of data according to step size
                ind -= step#self.block_size
                if ind > 0:
                    pmt[:ind] = pmt[step:step+ind]
                
        if ind_out > 0:
            yield result_full[:ind_out]
        


class RotationalSpeedDetector3( TimeOut ):
    """
    Class to calculate rotational speed for each sample. 
    This implementation is preliminary, interface will change in the future.
    --> about 4x faster than RSD(1), but not as accurate. 
    --> more accurate than RSD2
    """
    
    #: Maximum delta f of returned spectra in Hz
    df = Float(0.1)
    
    #: number of samples in one block for rotation detection
    block_size = Int(10000)
    
    #: Upper rotational frequency limit that should be resolved
    fmax = Float(100.0)
    
    #: Tuple with (lower, upper) frequency for band pass prior to cross correlation
    band_pass = Tuple((2500.0, 0.0))
    
    #: Result will be yielded for every `sample_step`-th sample. 
    #: If >1, this will result in a decreased `sample_freq`.
    sample_step = Int(10000)

    #: List of delta m (=b-a) to consider
    dm_list = List( [1,3,5,7,9] )#
    
    #: maximum cross mode to consider (max. nmodes/2 )
    m_max = Int(26)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Property( depends_on = ['source.sample_freq', 
                                          'sample_step'])
    
    #: Discrete resolved rotational frequencies
    rps_freqs = Property( depends_on = ['df', 'fmax'])
    
    # number of samples in output
    _ns_new = Property( depends_on = ['source.sample_freq', 'df'])
    

    @cached_property 
    def _get_sample_freq( self ):
        return self.source.sample_freq/self.sample_step
    
    @cached_property 
    def _get__ns_new( self ):
        # number of samples to return
        return int(ceil(2*self.fmax/self.df))

   
    @cached_property 
    def _get_rps_freqs( self ):
        return roll(fft.fftfreq(self._ns_new, 1./(self.df*self._ns_new)),self._ns_new//2)
        
    
    
    def result( self, num ):
        """
        see "12_frot_detection_weight_and_combine.py" for tests
        assumes that input is in mode order
        """
        
        
        
        bs = self.block_size
        fs = self.source.sample_freq
        step = self.sample_step
        modelist = self.source.modes
        nmodes = len(modelist)
        ns_new = self._ns_new
        
        f_low, f_high = self.band_pass
                                ### band pass filter ##############################
        if f_low>0:  
            b, a = butter(8, f_low/fs*2, btype='highpass')
        #TODO: f_high currently not used 
        if f_high>0:
            bh, ah = butter(8, f_high/self.source.sample_freq*2, btype='lowpass')
        #    data = filtfilt(b, a, data, axis=0)
        #if f_high>0:
        #    b, a = butter(8, f_high/fs*2, btype='lowpass')
        #    data = filtfilt(b, a, data, axis=0)
        
        # allocate arrays that hold the needed data
        result_full = zeros((num, ns_new))
        
        result_step = zeros(ns_new)
        
        pmt = zeros((self.block_size+num,nmodes), dtype=complex)
        
        
        # fill array with only part of the data
        #pmfpart[:mf_ind, :] = pmf[-mf_ind:, :self.m_max+1]
        #pmfpart[mf_ind:, :] = pmf[ :mf_ind, :self.m_max+1]
        
        #TODO 1: Diese Variante ist durch alleiniges Downsampling zwar schnell, aber doch etwas ungenau
        #        Ggf. intelligenteres (fractional) Downsampling?
        
        #TODO 2: Durch anti-aliasing-Fitler bei decimate gibt es starken Amplitudenabfall
        #        nach außen, d.h. fmax muss deutlich erhöht werden, um noch auswerten zu können
        
        
        ind = 0
        ind_out = 0
        
        
        for block in self.source.result( num ):
            #print(ind,end=',')
            ns = block.shape[0]
            ind1 = ind+ns
            
            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]
            
            ind = ind1
                        
            if ind >= bs:
                # highpass
                if f_low>0:  
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                    if f_high>0:
                        pmt_temp = filtfilt(bh, ah, pmt_temp, axis=0)
                else:
                    pmt_temp = pmt[:bs]

                for dm in self.dm_list:
            
                    # calculate cross-mode product, only need half of the modes, since everything is mirrored
                    # shift left/right, so that all mode combinations are present (and none doubled) in the resulting array
                    ind_m_b = roll(modelist[::-1], int(0.5-dm/2))[:nmodes//2]
                    ind_m_a = roll(modelist      , int(0.0+dm/2))[:nmodes//2]
                    pmtprod = pmt_temp[:,ind_m_a] * pmt_temp[:,ind_m_b]
                    
                    # subsampling factor
                    D = int(round(fs/(2*self.fmax*dm)))
                    
                    pmtprod = decimate(pmtprod,D, axis=0)
                    ns_down = pmtprod.shape[0]#int(ceil(ns/D))
                    #fs_down = fs / D
                    #print('dm:',dm,', D:',D,', ns_down:',ns_down)                    
                    
                    npad = ns_new-ns_down
                    if npad > 0:
                        pmf = fft.fft(pad(pmtprod,[(0, npad),(0, 0)]), None, axis=0, norm="ortho")
                    else:
                        print("df to high!!! Discarding data!")
                        pmf = fft.fft(pmtprod[:ns_new], None, axis=0, norm="ortho")
                
                
                    # weighting array
                    # derivative of angle -> small if rotational velocity is found
                    pmfanglediff=sum(abs(angle(pmf[:,1:] * pmf[:,:-1].conjugate())), axis=1)
                    
                    #get result for this mode
                    apmf = absolute(pmf)
                    result_dm = 1/pmfanglediff * sum(apmf/median(apmf[:ns_new//4],axis=0), axis=1)
                    
                    # average to get overall result
                    result_step += result_dm
                result_full[ind_out,:] = roll(result_step,ns_new//2)
                result_step *= 0
                ind_out += 1
                if ind_out >= num:
                    yield result_full
                    ind_out = 0
            
                # shift rest of data according to step size
                ind -= step#self.block_size
                if ind > 0:
                    pmt[:ind] = pmt[step:step+ind]
                
        if ind_out > 0:
            yield result_full[:ind_out]



class RotationalSpeed ( TimeOut ):
    '''
    Class to calculate rotational speed for each sample. 
    This implementation is preliminary, interface will change in the future.
    TODO: Maybe include real-time trigger identification.
    '''
    
    #: Number of successive rps values to average over, defaults to 3.
    naverage = Int(4, 
        desc = "number of samples to average over")

    #: Trigger data from :class:`acoular.tprocess.Trigger`.
    trigger = Instance(Trigger) 

    # delay of rot signal (due to distance to array)
    delay = Float(0.0)

    #: Trigger signals per revolution,
    #: defaults to 1.
    trigger_per_revo = Int(1,
                           desc ="trigger signals per revolution")
    
    #: Flag to set counter-clockwise (1) or clockwise (-1) rotation,
    #: defaults to -1.
    rot_direction = Int(-1,
                        desc ="mathematical direction of rotation")
    
    # internal identifier
    digest = Property(depends_on=['source.digest', 
                                  'trigger.digest', 
                                  'trigger_per_revo',
                                  'rot_direction',
                                  'start_angle'])

    @cached_property 
    def _get_digest( self ):
        return digest(self)
    
    def result(self, num, rpn = False):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        rpn : bool
            If true, yield result in revolution per sample instead of rev per second. 
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        
        if rpn:
            unit = 1
        else:
            unit = self.sample_freq
        
        self.trigger.source = self.source
        ns = self.numsamples
        
        inds_trigger, maxdist, mindist = self.trigger.trigger_data
        # trigger distances in samples
        dtrig = inds_trigger[1:]-inds_trigger[:-1]
        
        #average rpn between triggers
        rpn_avg = 1/dtrig
        
        ###########
        # # calculate moving average of rps
        # rps_rolling_mean = uniform_filter1d(rpn_avg, 
        #                                     self.naverage, 
        #                                     mode='nearest', 
        #                                     origin=-1)  \
        #                    * unit * self.rot_direction
        
        # # setup interpolation function to calculate per-sample
        # linear_fun = interp1d(inds_trigger[1:],
        #                       rps_rolling_mean,
        #                       fill_value="extrapolate")
        
        # #rps_track = linear_fun(arange(ns)) * self.sample_freq
        # #...yield rps_track[ind*num:(ind+1)*num]
        
        # indarray = arange(num)
        # lastblockind = ns%num
        # for ind in range(ns//num):
        #     yield linear_fun(indarray+ind*num)
        # else:
        #     if lastblockind: yield linear_fun(indarray[:lastblockind]+(ind+1)*num)
        ############
        
        rpn_track1 = zeros(ns)

        # set time before first trigger to same rps as after
        rpn_track1[:inds_trigger[0]] += rpn_avg[0]
        ind0 = inds_trigger[0]
        for ind1, rpni in zip(inds_trigger[1:], rpn_avg):
            rpn_track1[ind0:ind1] += rpni
            ind0 = ind1
        # set time after last trigger to same rps as before
        rpn_track1[ind1:] += rpni
        
        # rpn or rps
        rpn_track1 *= unit * self.rot_direction
        nroll =  int(round(abs(self.delay*self.sample_freq)))
        print('roll samples:',nroll)
        rpn_track1 = roll( rpn_track1, nroll)
        
        nblocks = ns//num + (ns%num>0)
        for ind in range(nblocks):
            yield rpn_track1[ind*num:(ind+1)*num]