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
 
"""
from warnings import warn

# imports from other packages
from numpy import (append, argmax, argmin, array, delete, flatnonzero, inf,
                   mean, sign)
from traits.api import (Float, Instance, Int,
                        Property, Trait, Tuple, cached_property)

from acoular.base import SamplesGenerator, TimeOut
from acoular.internal import digest

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
        nChannels = self.source.num_channels
        if not nChannels == 1:
            raise Exception('Trigger signal must consist of ONE channel, instead %s channels are given!' % nChannels)
        return 0

