# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, C0103, R0901, R0902, R0903, R0904, W0232
"""
Classes dealing with rotating systems.

.. autosummary::
    :toctree: generated/
   
    AngleTrajectory
    TrajectoryAnglesFromTrigger
"""

from numpy import array, empty, median, sign, size, where, pi
from scipy.interpolate import splev, splrep
from traits.api import (Bool, Dict, Float, Int, Property, Trait, Tuple,
                        cached_property, on_trait_change, Instance)

from acoular.internal import digest
from acoular.base import SamplesGenerator
from acoular.trajectory import Trajectory

from .trigger import Trigger


class AngleTrajectory ( Trajectory ):
    """
    Describes the trajectory as angular positions.
    
    The angles are derived from sampled trigger signals.
    Does spline interpolation of angles between samples.
    """
    
    #: Data source (:class:`~acoular.base.SamplesGenerator` object with 
    #: trigger signal).
    source = Instance(SamplesGenerator)
    
    #: Dictionary that assigns discrete time instants (keys) to 
    #: sampled angles along the trajectory (values)
    points = Dict(key_trait = Float, value_trait = Float, 
        desc = "sampled angles along the trajectory")
    
    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 
                      'ntrig',
                      'channel',
                      'rotation',
                      'factor',
                      '__class__'])
    
    
    
    #: factor tp multiply rpm with (for slowing down or speeding up virtual rotation)
    factor = Float(1.0,
                  desc = "rpm factor")
    
    #: Number of triggers per rotation
    ntrig = Float(1.0,
                  desc = "number of triggers per rotation")
    
    #: Channel which contains the trigger 
    #: signal; defaults to -1 (=last channel).
    channel = Int(-1,
                  desc = "channel with trigger signal")
    
    #: Number of revolutions per minute; calculated automatically 
    #: from trigger signal
    rpm = Property( depends_on = 'digest')

    #: Array that is empty if all triggers are plausible (and none are missing).
    #: If one or more triggers are missing, this array contains values between 
    #: -1 and 1, their absolute value indicating the relative position of the 
    #: disruption in the used time data (0 ~ start of time data, 1 ~ end of
    #: time data). 
    #: Note that for one sequence of missing triggers, rpm_check will
    #: contain TWO consecutive entries: one (positive) indicating the 
    #: beginning of the disruption and one (negative) indicating its end.
    rpm_check  = Property( depends_on = 'digest')
                
    #: Flag to set counter-clockwise (1) or clockwise (-1) rotation,
    #: defaults to -1.
    rotation = Int(-1,
                   desc ="mathematical direction of rotation")
    
    #: 
    invert = Bool(False)
    
    #: Index of first trigger; is set automatically
    startind = Property( depends_on = 'digest')
                         
    # spline data, internal use
    tck = Property(depends_on = 'digest')    
    
    # some local variables, only to be called by getter routines
    _tck = Tuple()
    _rpm = Float()
    _startind = Int()
    
    # internal flag to determine whether trigger signal has been processed
    _triggerflag = Bool(False) 
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    
    
    
    def trigger( self, num = 1024):
        """
        Set all channels except one (trigger channel defined 
        by :attr:`channel`) to "invalid" and
        yield the isolated trigger channel.
        
        Parameters
        ----------
        num : integer, defaults to 1024
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, 1). 
            The last block may be shorter than num.

        """
        if self.invert:
            factor = -1
            add = 4.9 #TODO: this is very crude, and only works iv trigger signal is 5V!
        else:
            factor = 1
            add = 0
        
        for temp in self.source.result(num):
            yield add + factor * temp[:,self.channel]
        
    # Note: a&r assumes that the first trigger found is the one defining the 
    # starting angle (might pose a problem with multiple triggers per rotation)
    def _angles_and_rpm( self, num=1024 ):
        """
        Internal function that calculates the angles of rotation and the rpm
        depending on the trigger signal. It will be called automatically.
        
        Parameters
        ----------
        num : integer, defaults to 1024
            This parameter defines the size of the blocks to be handled.
        
        Returns
        -------
            This function doesn't return a value itself, but sets the class
            attributes :attr:`points` and :attr:`rpm`.
        """

        self.points.clear()#__init__
        trigger_gen = self.trigger(num)
        sign_r = self.rotation
        trimax = max([max(_) for _ in self.trigger(num)]) 

        itrig = 0
        lastind = 0
        ilastind = 0 # TODO: find a way to get rid of unnecessary variables later
        
        ref_ind = 5 # index difference for trigger determination
        ind = ref_ind # so you can subtract ref_ind when looking for trigger position
        ntrig = self.ntrig
        ns =  self.source.numsamples

        #dang = 360.0/ntrig # don't use this anymore, since it is not given that angles between trigger are exact portion of full circle
        dang = 360.
        
        angle0 = 0        
        
        trigger = empty(num + ref_ind)
        trigger[0:num] = next(trigger_gen)
        
        while ind < ns :
            
            rel_ind = ind % num
            if rel_ind == 0:
                # put the last values from last step at the very end
                trigger[num:num+ref_ind] = trigger[num-ref_ind:num]
                new_trigger = next(trigger_gen)
                # refill the array
                trigger[0:new_trigger.shape[0]] = new_trigger
            # find the sudden increase, using min 20 samples distance 
            # is arbitrary
            # [ind%1024-ref_ind] can be negative -> take last array values
            if (trigger[rel_ind]  > trimax/2.0
                and trigger[rel_ind - ref_ind] < trimax/10.0
                and ind - ilastind > 20):
                
                if itrig%ntrig == 0: # only count every full revolution
                    dind = ind - lastind                
                    
                    # use sample number as time unit
                    self.points[ind] = angle0 * sign_r
                                           
                    if itrig//ntrig == 1:
                        startind = lastind                    
                        # assume the rotation before the first trigger is the
                        # same as afterwards.
                        # calculate additional 10 earlier "trigger" points
                        # for better interpolation
                        for iback in range(1,11):
                            self.points[startind - iback * dind] = \
                                            sign_r * (angle0 - (iback + 1) * dang)
    
                    angle0 += dang
                            
                    lastind = ind
                ilastind = ind
                    
                itrig += 1
                
            ind+=1

        if itrig > ntrig:
            self._startind = startind
            # calculate additional 10 later "trigger" points
            # for better interpolation
            for iback in range(1,11):
                self.points[lastind + iback * dind] = \
                                        sign_r * (angle0 + (iback - 1) * dang)
            #self._rpm = self.rotation * ( (itrig-1.0) * self.source.sample_freq) \
            #            / (lastind-startind) * 60.0 / ntrig
            self._rpm = self.rotation * ( (itrig//ntrig-1.0) * self.source.sample_freq) \
                        / (lastind-startind) * 60.0 
            t = sorted(self.points.keys())
            xp = array([self.points[i] for i in t]).T
            k = min(3, len(self.points)-1)
            self._tck = splrep(t, xp, s=0, k=k)
        else:
            print("No or only one trigger found!")
            self._rpm = 0
            self._tck = splrep([0,1], [0,0], s=0, k=1)
        
        self._triggerflag = True
        
    @on_trait_change('digest')
    def _reset_triggerflag( self ):
        self._triggerflag = False


    @cached_property
    def _get_rpm_check( self ):
        if not self._triggerflag:
            self._angles_and_rpm()
            
        plausible_limit = 0.3 # allow up to 30% deviation
        
        # get the samples of trigger (w/o artificially calculated ones)        
        trig = array(sorted(self.points.keys())[10:-10])
        # number of samples between triggers
        d_trig  = trig[1:]-trig[:-1]
        d2_trig = d_trig[1:]-d_trig[:-1] # second derivative
        # if the trigger signal is non-continuous, 
        # it shows in the second derivative
        d2_trig /= float(median(d_trig)) # get relative change
        
        # earlier version: just return whether signal has gaps or not 
        #max_deviation = max(abs(d2_trig))
        #return (max_deviation <= plausible_limit)
        
        # new version: return relative position of gaps or empty array
        ind_miss = where(abs(d2_trig) > plausible_limit)[0]
        return ind_miss * sign(d2_trig[ind_miss])/float(size(d2_trig)-1)

    @cached_property
    def _get_rpm( self ):
        if not self._triggerflag:
            self._angles_and_rpm()
        return self._rpm*self.factor

    @cached_property
    def _get_tck( self ):
        if not self._triggerflag:
            self._angles_and_rpm()
        return self._tck

    @cached_property
    def _get_startind( self ):
        if not self._triggerflag:
            self._angles_and_rpm()
        return int(round(self._startind*self.factor))
        
    def location(self, t, der=0):
        """ 
        Returns angles in deg for one or more instants in time.
        
        Parameters
        ----------
        t : array of floats
            Instances in time to calculate the positions at.
        der : integer
            The order of derivative of the spline to compute, defaults to 0.
        
        Returns
        -------
        arrays of floats
            Angles in degree at the given times; array has the same shape as t .
        """

        return (splev(t, self.tck, der)*self.factor) % 360.0


#TODO: Re-implement this in a clean way
class TrajectoryAnglesFromTrigger ( Trajectory ):
    """
    Describes the trajectory as angular positions.
    
    The angles are derived from sampled trigger signals.
    Does spline interpolation of angles between samples.
    """
    
    #: Data source (:class:`~acoular.tprocess.Trigger` object with 
    #: trigger indices).
    trigger = Trait(Trigger)
    
    #: Dictionary that assigns discrete time instants (keys) to 
    #: sampled angles along the trajectory (values). Read-only.
    #: TODO: check if this public trait is actually needed.
    points = Property(depends_on = 'digest')

    # internal identifier
    digest = Property( 
        depends_on = ['trigger.digest', 
                      'ntrig',
                      'start_index',
                      'rot_direction',
                      '__class__'])
    
    # this is needed for varying rotational speed, so that no extrapolation is needed
    #: Index of Trigger to be considered as start index
    start_index = Int(0,
                    desc ="index of trigger to be considered as start")

    #: Number of triggers per rotation
    ntrig = Int(1,
                  desc = "number of triggers per rotation")
    
    #: Number of revolutions per minute; calculated automatically 
    #: from trigger signal
    rps_average = Property( depends_on = 'digest')
    
    #: Index of the first trigger point
    first_trigger = Property(depends_on = 'trigger')

    #: rotation angle in radians for first trigger position
    start_angle = Float(0.0,
                   desc ="rotation angle for trigger position")
                
    #: Flag to set counter-clockwise (1) or clockwise (-1) rotation,
    #: defaults to -1.
    rot_direction = Int(-1,
                    desc ="mathematical direction of rotation")
    
    # spline data, internal use
    tck = Property(depends_on = 'digest')    
    
    # some local variables, only to be called by getter routines
    _tck = Tuple()
    _rps_average = Float()
    _first_trigger = Float()
    
    _points = Dict(key_trait = Float, value_trait = Float)

    
    # internal flag to determine whether trigger signal has been processed
    _angles_flag = Bool(False) 
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    
    
        
    # Note: a&r assumes that the first trigger found is the one defining the 
    # starting angle (might pose a problem with multiple triggers per rotation)
    def _angles_and_rps( self, num=1024 ):
        """
        Internal function that calculates the angles of rotation and the rpm
        depending on the trigger signal. It will be called automatically.
        
        Parameters
        ----------
        num : integer, defaults to 1024
            This parameter defines the size of the blocks to be handled.
        
        Returns
        -------
            This function doesn't return a value itself, but sets the class
            attributes :attr:`points` and :attr:`rpm`.
        """
        
        trig_inds = self.trigger.trigger_data - self.start_index
        # dictionary needs float keys
        trig_inds = trig_inds.astype(float)        

        self._points.clear()#__init__
        
        self._points.update({ind : self.rot_direction*2*pi*i for i,ind in enumerate(trig_inds[::self.ntrig])})
        #self._points.update(self.add_points) # additional opints that can be set by user, not currently used
        
        self._first_trigger = trig_inds[trig_inds>=0][0]
        # assume the rotation before the first trigger is the
        # same as afterwards.
        # calculate additional 10 earlier "trigger" points
        # for better interpolation
        for iback in range(1,11):
            self._points[trig_inds[0] - iback * (trig_inds[1]-trig_inds[0])] = \
                            self.rot_direction * ( -2 * pi * iback )

        # calculate additional 10 later "trigger" points
        # for better interpolation
        for iback in range(1,11):
            self._points[trig_inds[-1] + iback * (trig_inds[-1]-trig_inds[-2])] = \
                            self._points[trig_inds[-1]]+self.rot_direction * ( 2 * pi * iback )

        t = sorted(self._points.keys())
        xp = array([self._points[i] for i in t]).T
        k = min(3, len(self._points)-1)
        self._tck = splrep(t, xp, s=0, k=k)

        self._rps_average = self.rot_direction * \
                             ( (len(trig_inds)//self.ntrig-1.0) * \
                               self.trigger.source.sample_freq) / \
                             (trig_inds[-1]-trig_inds[0]) 
        
        self._angles_flag = True
        
    @on_trait_change('digest')
    def _reset_angles_flag( self ):
        self._angles_flag = False


    @cached_property
    def _get_rps_average( self ):
        if not self._angles_flag:
            self._angles_and_rps()
        return self._rps_average


    @cached_property
    def _get_first_trigger( self ):
        if not self._angles_flag:
            self._angles_and_rps()
        return self._first_trigger

    @cached_property
    def _get_tck( self ):
        if not self._angles_flag:
            self._angles_and_rps()
        return self._tck

    @cached_property
    def _get_points( self ):
        if not self._angles_flag:
            self._angles_and_rps()
        return self._points


    def location(self, t, der=0):
        """ 
        Returns angles in deg for one or more instants in time.
        
        Parameters
        ----------
        t : array of floats
            Instances in time to calculate the positions at.
        der : integer
            The order of derivative of the spline to compute, defaults to 0.
        
        Returns
        -------
        arrays of floats
            Angles in degree at the given times; array has the same shape as t .
        """

        return (splev(t, self.tck, der)+self.start_angle) % (2*pi)