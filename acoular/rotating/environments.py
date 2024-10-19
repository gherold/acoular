# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, C0103, R0901, R0902, R0903, R0904, W0232
"""
Classes dealing with rotating systems.

.. autosummary::
    :toctree: generated/
    
    AxialRotatingFlowEnvironment
    EnvironmentRot
    EnvironmentRotFlow
    
"""

from numpy import (array, cos, float32, float64, isscalar, newaxis, ones, pi,
                   sin, sqrt, sum, zeros)
from traits.api import CArray, Float, Property, cached_property

from acoular.environments import Environment, UniformFlowEnvironment
from acoular.internal import digest


class AxialRotatingFlowEnvironment( UniformFlowEnvironment ):
    """
    An acoustic environment for rotating array and grid, taking into account
    a uniform flow.

    This class provides the facilities to calculate the travel time (distances)
    between grid point locations and microphone locations where microphone array
    and focus grid are rotating around a common axis.
    """

    # The unit vector that gives the direction of the flow, defaults to
    # flow in positive z-direction.
    #fdv = CArray( dtype=float64, shape=(3, ), value=array((0, 0, 1.0)), 
    #    desc="flow direction")
    # fdv is ignored here, only flow and rotation in z axis is considered
    
    #: Angle in radians for grid rotation correction. The expected input 
    #: value is the angle of the grid rotation of the first sample.
    #: This means that the resulting map will be rotated by the _negative_ 
    #: angle as opposed to if it is 0.
    #: Defaults to 0.
    start_angle = Float(0.,
                   desc ="grid correction angle")
    
    #: Rotational speed of the _medium_ in rps (i.e. if used in VRA context,
    #: this is the negative the rotational speed of the array). 
    #: Positive, if rotation is around positive z-axis sense,
    #: which means from x to y axis.
    rotational_speed = Float(0.0)
    
    
    #: Aborting criterion for iterative calculation of distances; 
    #: use lower values for better precision (takes more time);
    #: defaults to 0.001.
    precision = Float(1e-3,
        desc="abort criterion for iteration to find distances; the lower the more precise")  
    
    
    # internal identifier
    digest = Property( 
        depends_on=['start_angle', 'precision', 'ma', 'rotational_speed'])#,'fdv'],) 



    @cached_property
    def _get_digest( self ):
        return digest( self )
        

    def _r( self, gpos, mpos=0.0):
        """
        Calculates the virtual distances between grid point locations and
        microphone locations or the origin. These virtual distances correspond
        to travel times of the sound.

        Parameters
        ----------
        c : float
            The speed of sound to use for the calculation.
        gpos : array of floats of shape (3, N)
            The locations of points in the beamforming map grid in 3D cartesian
            co-ordinates.
        mpos : array of floats of shape (3, M), optional
            The locations of microphones in 3D cartesian co-ordinates. If not
            given, then only one microphone at the origin (0, 0, 0) is
            considered.

        Returns
        -------
        array of floats
            The distances in a twodimensional (N, M) array of floats. If M==1, 
            then only a onedimensional array is returned.
        """
        c=self.c
        
        prec = abs(self.precision) # ensure positive values for abort criterion
        
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]            

        # ensure fdv being n unit vector
        #fdv = self.fdv / sqrt((self.fdv*self.fdv).sum())
        
        # rotation matrix for starting angle
        rotgrid = array([[ cos(self.start_angle), -sin(self.start_angle), 0],
                         [ sin(self.start_angle),  cos(self.start_angle), 0],
                         [ 0                    ,  0                    , 1]])
        
        # correct grid angle
        gp = (rotgrid @ gpos)[:,:,newaxis]
        
        # initialize "mic-specific" grid positions
        gpr = zeros((*gp.shape[:2], mpos.shape[1]))
        gpr[2] += gp[2]
        
        
        # calculate initial distances
        mpos = mpos[:, newaxis, :]
        rmv = gp - mpos
        rm = sqrt(sum(rmv*rmv, 0))
        
        # calculate relative Mach numbers
        ma_rel = self.ma * rmv[2] / rm
        rm *= 1/(-ma_rel + sqrt(ma_rel*ma_rel - self.ma*self.ma + 1))

        """
        # direction unit vector from mic points to grid points
        rmv_unit = rmv / rm
        # relative Mach number (scalar product)
        ma_rel = (rmv_unit * mav[:,newaxis,newaxis]).sum(0)
        rm /= (1 - ma_rel)
        """
        
        if self.rotational_speed != 0.0:
            omega = self.rotational_speed * 2*pi # angular velocity
            while True:
                rm_last = rm
                
                # sound travel time from each gridpt to each mic
                t_mf = rm / c  
                
                # calculate angle where gridpoint was when sound was emitted
                phi = omega * t_mf 

                # rotation of grid positions...
                gpr[0] = gp[0]*cos(phi) - gp[1]*sin(phi)
                gpr[1] = gp[0]*sin(phi) + gp[1]*cos(phi)
                
                rmv = gpr - mpos
                rm = sqrt(sum(rmv*rmv, 0))
                
                ma_rel = self.ma * rmv[2] / rm
                rm *= 1/(-ma_rel + sqrt(ma_rel*ma_rel - self.ma*self.ma + 1))
                
                rel_change = sum(abs(rm-rm_last)) / sum(rm)
                if  rel_change < prec :
                    break
        
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        #print 'done!'
        return rm


class EnvironmentRot( Environment ):
    """
    An acoustic environment for rotating array and grid.

    This class provides the facilities to calculate the travel time (distances)
    between grid point locations and microphone locations where microphone array
    and focus grid are rotating around a common axis.
    """
    
    #: Revolutions per minute of the array; negative values for
    #: clockwise rotation; defaults to 0."
    rpm = Float(0.0,
        desc="revolutions per minute of the virtual array; negative values for clockwise rotation")
 
     #: Aborting criterion for iterative calculation of distances; 
     #: use lower values for better precision (takes more time);
     #: defaults to 0.001.
    precision = Float(1e-3,
        desc="abort criterion for iteration to find distances; the lower the more precise")    
        
    # internal identifier
    digest = Property( 
        depends_on=['rpm','precision'], 
        )

    # traits_view = View(
    #         [
    #             ['rpm', 'precision{abort criterion}'], 
    #             '|[Rotating Array + Grid]'
    #         ]
    #     )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    def _r( self, gpos, mpos=0.0):
        """
        Calculates the virtual distances between grid point locations and
        microphone locations or the origin. These virtual distances correspond
        to travel times of the sound.

        Parameters
        ----------
        c : float
            The speed of sound to use for the calculation.
        gpos : array of floats of shape (3, N)
            The locations of points in the beamforming map grid in 3D cartesian
            co-ordinates.
        mpos : array of floats of shape (3, M), optional
            The locations of microphones in 3D cartesian co-ordinates. If not
            given, then only one microphone at the origin (0, 0, 0) is
            considered.

        Returns
        -------
        array of floats
            The distances in a twodimensional (N, M) array of floats. If M==1, 
            then only a onedimensional array is returned.
        """
        c=self.c
        #print('Calculating distances...')
        prec = abs(self.precision) # ensure positive values for abort criterion
        
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]            
        mpos = mpos[:, newaxis, :]
        rmv = gpos[:, :, newaxis]-mpos
        rm = sqrt(sum(rmv*rmv, 0))
        if self.rpm != 0.0:
            omega = self.rpm/60.0*2*pi # angular velocity
            while True:
                rm_last = rm
                t_mf = rm / c # sound travel time from each gridpt to each mic
                phi = -omega * t_mf # where was gridpt when sound was emitted
                
                # rotation of grid positions...
                gpos_r = array((  gpos[0,:,newaxis]*cos(phi) 
                                    - gpos[1,:,newaxis]*sin(phi),
                                  gpos[0,:,newaxis]*sin(phi) 
                                    + gpos[1,:,newaxis]*cos(phi),
                                  gpos[2,:,newaxis]*ones(phi.shape)   ))
                
                rmv = gpos_r - mpos
                rm = sqrt(sum(rmv*rmv, 0))
                rel_change = sum(abs(rm-rm_last)) / sum(rm)
                
                if  rel_change < prec :
                    break
        
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        #print 'done!'
        return rm



class EnvironmentRotFlow( EnvironmentRot ):
    """
    An acoustic environment for rotating array and grid, taking into account
    a uniform flow.

    This class provides the facilities to calculate the travel time (distances)
    between grid point locations and microphone locations where microphone array
    and focus grid are rotating around a common axis.
    """
    
    #: The Mach number, defaults to 0.
    ma = Float(0.0, 
        desc="flow mach number")

    #: The unit vector that gives the direction of the flow, defaults to
    #: flow in positive z-direction.
    fdv = CArray( dtype=float64, shape=(3, ), value=array((0, 0, 1.0)), 
        desc="flow direction")
    
    # internal identifier
    digest = Property( 
        depends_on=['rpm','precision','ma','fdv'], 
        )
        
    # traits_view = View(
    #         [
    #             ['rpm', 'precision{abort criterion}','ma{flow Mach number}'
    #              'fdv{flow direction}'], 
    #             '|[Rotating Array + Grid]'
    #         ]
    #     )

    @cached_property
    def _get_digest( self ):
        return digest( self )
        

    def _r( self, gpos, mpos=0.0):
        """
        Calculates the virtual distances between grid point locations and
        microphone locations or the origin. These virtual distances correspond
        to travel times of the sound.

        Parameters
        ----------
        c : float
            The speed of sound to use for the calculation.
        gpos : array of floats of shape (3, N)
            The locations of points in the beamforming map grid in 3D cartesian
            co-ordinates.
        mpos : array of floats of shape (3, M), optional
            The locations of microphones in 3D cartesian co-ordinates. If not
            given, then only one microphone at the origin (0, 0, 0) is
            considered.

        Returns
        -------
        array of floats
            The distances in a twodimensional (N, M) array of floats. If M==1, 
            then only a onedimensional array is returned.
        """
        c=self.c
        #print('Calculating distances...')
        prec = abs(self.precision) # ensure positive values for abort criterion
        
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]            

        # ensure fdv being n unit vector
        fdv = self.fdv / sqrt((self.fdv*self.fdv).sum())
        
        mpos = mpos[:, newaxis, :]
        rmv = gpos[:, :, newaxis]-mpos
        rm = sqrt(sum(rmv*rmv, 0))
        
        macostheta = (self.ma*sum(rmv.reshape((3, -1))*fdv[:, newaxis], 0)\
                     / rm.reshape(-1)).reshape(rm.shape)
        rm *= 1/(-macostheta + sqrt(macostheta*macostheta-self.ma*self.ma+1))

        """
        # direction unit vector from mic points to grid points
        rmv_unit = rmv / rm
        # relative Mach number (scalar product)
        ma_rel = (rmv_unit * mav[:,newaxis,newaxis]).sum(0)
        rm /= (1 - ma_rel)
        """
        if self.rpm != 0.0:
            omega = self.rpm/60.0*2*pi # angular velocity
            while True:
                rm_last = rm
                
                # sound travel time from each gridpt to each mic
                t_mf = rm / c  
                phi = -omega * t_mf # where was gridpt when sound was emitted
                
                # rotation of grid positions...
                gpos_r = array((  gpos[0,:,newaxis]*cos(phi) 
                                    - gpos[1,:,newaxis]*sin(phi),
                                  gpos[0,:,newaxis]*sin(phi) 
                                    + gpos[1,:,newaxis]*cos(phi),
                                  gpos[2,:,newaxis]*ones(phi.shape)   ))
                
                rmv = gpos_r - mpos
                rm = sqrt(sum(rmv*rmv, 0))
                
                macostheta = (self.ma*sum(rmv.reshape((3, -1))*fdv[:, newaxis], 0)\
                             / rm.reshape(-1)).reshape(rm.shape)
                rm *= 1/(-macostheta + sqrt(macostheta*macostheta-self.ma*self.ma+1))
                
                rel_change = sum(abs(rm-rm_last)) / sum(rm)
                if  rel_change < prec :
                    break
        
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        #print 'done!'
        return rm

