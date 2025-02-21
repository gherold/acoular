# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, C0103, R0901, R0902, R0903, R0904, W0232
"""
Implements support for circular focus grids 

.. autosummary::
    :toctree: generated/
    
    CircGrid
    CircMesh
    GridMesh
    EqCircGrid
    EqCircGrid3D

"""

# imports from other packages
from acoular.grids import Grid
from acoular.internal import digest

from numpy import mgrid, s_, array, arange, linspace, modf, argmin, amin, \
amax, meshgrid, ones, empty, sin, cos, pi

from scipy.interpolate import griddata
from matplotlib.path import Path
from traits.api import HasPrivateTraits, Trait, Float, Property, \
CArray, property_depends_on, cached_property, on_trait_change




class CircGrid (Grid):
    """
    Provides a circular 2D grid for the beamforming results.
    
    The grid has circular-sector-like cells with four corners and 
    is on a plane perpendicular to the z-axis. 
    It is defined by lower and upper r- and  phi-limits and the 
    z co-ordinate, and the increments in r and phi direction respectively.
    """
    
    #: Inner radius of the grid; defaults to 0.1.
    r_min = Float(0.1, 
        desc="inner radius")

    #: Outer radius of the grid; defaults to 1.
    r_max = Float(1.0, 
        desc="outer radius")

    #: Minimum/starting angle of the grid; defaults to 0.0.
    phi_min = Float(0.0, 
        desc="minimum  phi-value")

    #: Maximum angle of the grid; defaults to 360.
    phi_max = Float(360.0, 
        desc="maximum  phi-value")

    #: Position on z-axis; defaults to 1.0.
    z = Float(1.0, 
        desc="position on z-axis")

    #: Increment in r-direction, defaults to 0.1.
    dr = Float(0.1, 
        desc="step size in r direction")

    #: Increment in phi-direction, defaults to 4.0.
    dphi = Float(4.0, 
        desc="step size in phi direction")
        
    #: Number of grid points along r; is set automatically.
    nrsteps = Property( 
        desc="number of grid points along r")

    #: Number of grid points along phi; is set automatically.
    nphisteps = Property( 
        desc="number of grid points along phi")

    # internal identifier
    digest = Property( 
        depends_on = ['r_min', 'r_max', 'phi_min', 'phi_max', 
                      'z', 'dr', 'dphi']
        )


    @property_depends_on('nrsteps, nphisteps')
    def _get_size ( self ):
        return self.nrsteps*self.nphisteps

    @property_depends_on('nrsteps, nphisteps')
    def _get_shape ( self ):
        return (self.nrsteps, self.nphisteps)

    @property_depends_on('r_min, r_max, dr')
    def _get_nrsteps ( self ):
        i = abs(self.dr)
        if i != 0:
            return int(round((abs(self.r_max-self.r_min)+i)/i))
        return 1

    @property_depends_on('phi_min, phi_max, dphi')
    def _get_nphisteps ( self ):
        diff = self.phi_max-self.phi_min
        i = abs(self.dphi)
        if diff == 360.0:
            diff -= i
        else:
            diff = diff%360
            
        if i != 0:
            return int(round((diff%360.0+i)/i))
        return 1

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def _get_pos( self, polar = False ):
        """
        Calculates grid co-ordinates.
        
        Parameters
        ----------
        polar : boolean
            Flag to trigger the cartesian or polar coordinate output.
            Default is 'False' (=cartesian).
            
        Returns
        -------
        array of floats of shape (3, [gridsize])
            The grid point (x, y, z)- or (r, phi, z)-coordinates in one array.
        """        
        ir = self.dr
        ri = 1j*round((self.r_max-self.r_min+ir)/ir)
        
        iphi = self.dphi
        diff = self.phi_max-self.phi_min
        if diff == 360.0:
            diff -= 360.0/round(diff/iphi)   
        else:
            diff = diff%360
        
        nphi = 1j*round(diff/iphi+1)
        #mgrid[{start} : {stop} : {nsteps}j ]
        bpos = mgrid[self.r_min : self.r_max : ri, 
                     self.phi_min : self.phi_min+diff : nphi,
                     self.z : self.z+0.1]
        bpos.resize((3, self.size))
        if polar:
            return bpos
        else:
            xpos = empty((3, self.size))
            xpos[0,:] = bpos[0,:]*cos(bpos[1,:]/180.0*pi)
            xpos[1,:] = bpos[0,:]*sin(bpos[1,:]/180.0*pi)
            xpos[2,:] = bpos[2,:]
            
            return xpos

    def index ( self, r, phi ):
        """
        Queries the indices for a grid point near a certain co-ordinate.

        This can be used to query results or co-ordinates at/near a certain
        co-ordinate.
        
        Parameters
        ----------
        r, phi : float
            The polar co-ordinates for which the indices are queried.

        Returns
        -------
        2-tuple of integers
            The indices that give the grid point nearest to the given r, phi
            co-ordinates from an array with the same shape as the grid.            
        """
        if r < self.r_min or r > self.r_max:
            raise(ValueError, "r-value out of range")
        if phi  <  self.phi_min or phi > self.phi_max:
            raise(ValueError, "phi-value out of range")
        ri = int((r-self.r_min)/self.dr+0.5)
        phii = int((phi-self.phi_min)/self.dphi+0.5)%self.nphisteps
        return ri, phii

    def indices ( self, r1, phi1, r2, phi2=None ):
        """
        Queries the indices for a subdomain in the grid.
        
        Allows either circle segment or circular subdomains. This can be used to
        mask or to query results from a certain sector or subdomain.
        
        Parameters
        ----------
        r1, phi1, r2, phi2 : float
            If all four paramters are given, then a circle segment sector is
            assumed that is given by two corners (r1, phi1) and (r2, phi2). If
            only three parameters are given, then a circular sector is assumed
            that is given by its center (r1, phi1) and the radius r2.

        Returns
        -------
        2-tuple of arrays of integers or of numpy slice objects
            The indices that can be used to mask/select the grid subdomain from 
            an array with the same shape as the grid.            
        """
        if phi2 is None:
            xpos = self.pos
            pos = self._get_pos(polar = True)
            x = r1*cos(phi1/180.0*pi)
            y = r1*sin(phi1/180.0*pi)
            ris = []
            phiis = []
            
            dr2 = (xpos[0,:]-x)**2 + (xpos[1,:]-y)**2
            inds = dr2 <= r2**2
            for np in arange(self.size)[inds]:
                ri, phii = self.index(pos[0,np], pos[1,np])
                ris += [ri]
                phiis += [phii]
            if not (ris and phiis):
                return self.index(r1, phi1)
            else:
                return array(ris), array(phiis)
                
        else:
            ri1, phii1 = self.index(min(r1, r2), phi1 )
            ri2, phii2 = self.index(max(r1, r2), phi2 )
            
            if phii2 < phii1:
                # sector passes grid border, split the indices
                return s_[ri1:ri2+1], array(range(phii1,self.nphisteps)+\
                                            range(0,phii2+1))
            else:
                return s_[ri1:ri2+1], s_[phii1:phii2+1]




class CircMesh( HasPrivateTraits ):
    """
    Provides the functionality to create a meshed grid based on 
    a :class:`CircGrid` object.
    
    The meshed grid positions and mapped values created can easily be used 
    for 2D plotting, e.g. using matplotlib's pcolor.
    The grid will be closed when full 360 degrees are used.
    
    Usage:  
    
    >>> cm = CircMesh(map_in=L_p(map1), grid=g)
    >>> im = pcolormesh(cm.X, cm.Y, cm.MAP, edgecolors='None')
    """
    
    #: :class:`CircGrid` object that defines the grid geometry.
    grid = Trait(CircGrid, 
                 desc="beamforming grid")
    
    #: The map containing the beamformer output
    map_in = CArray( 
                     desc="beamformer output map")
    
    #: The mesh for the x values.
    X = Property(
                desc="X mesh")
    
    #: The mesh for y values.
    Y = Property(
                desc="Y mesh")

    #: The mesh for the soundmap.
    MAP = Property(
                  desc="Map mesh")
    
    def _get_X ( self ):
        gsh=self.grid.shape
        pos0 = self.grid.pos[0].reshape(gsh)
        if (self.grid.phi_max-self.grid.phi_min == 360.0):
            posx = empty((gsh[0],gsh[1]+1))
            posx[:,0:-1] = pos0[:,:]
            posx[:,-1] = pos0[:,0]
        else:
            posx = pos0
        return posx

    def _get_Y ( self ):
        gsh=self.grid.shape
        pos1 = self.grid.pos[1].reshape(gsh)
        if (self.grid.phi_max-self.grid.phi_min == 360.0):
            posy = empty((gsh[0],gsh[1]+1))
            posy[:,0:-1] = pos1[:,:]
            posy[:,-1] = pos1[:,0]
        else:
            posy = pos1
        return posy

    def _get_MAP ( self ):
        gsh=self.grid.shape
        mapin=self.map_in.reshape(gsh)
        if (self.grid.phi_max-self.grid.phi_min == 360.0):
            mapfull = empty((gsh[0],gsh[1]+1))
            mapfull[:,0:-1] = mapin[:,:]
            mapfull[:,-1] = mapin[:,0]
        else:
            mapfull = mapin 
        return mapfull


class GridMesh( HasPrivateTraits ):
    """
    Provides the functionality to create a meshed grid based on 
    a arbitrary 2D :class:`~acoular.grids.Grid`-derived object.
    
    The grid will be projected onto a regular-spaced rectangular grid and the
    mapped values will be interpolated accordingly using Delaunay 
    triangularization.
    The meshed grid positions and mapped values created can easily be used 
    for 2D plotting, e.g. using matplotlib's pcolor.
    
    Usage:     
    
    >>> gm = GridMesh(map_in=L_p(map1), grid=g)
    >>> im = pcolormesh(gm.XYMAP, edgecolors='None')
    """

    #: :class:`~acoular.grids.Grid`-derived object that defines the grid geometry.
    grid = Trait(Grid, 
                 desc="beamforming grid")
    
    #: The map containing the beamformer output.
    map_in = CArray( 
                     desc="beamformer output map")
    #: Meshed grid and data, returns (X, Y, MAP).
    XYMAP = Property(
                  desc="Map mesh")
                  
    #: The step size for interpolated grid resolution, defaults to 0.1.
    increment = Float(0.1, 
        desc="step size") 
    

    def _get_XYMAP ( self ):
        # Interpolate using delaunay triangularization 
        xpos = self.grid.pos        
        print(xpos[0].shape)
        print(xpos[1].shape)
        print(self.map_in.shape)
        xmin = amin(xpos[0])
        xmax = amax(xpos[0])
        ymin = amin(xpos[0])
        ymax = amax(xpos[0])

        xi = linspace(xmin, xmax, (xmax-xmin)/self.increment)
        yi = linspace(ymin, ymax, (ymax-ymin)/self.increment)

        X, Y = meshgrid(xi, yi)

        return X, Y, griddata( xpos[0],
                               xpos[1],
                               self.map_in.reshape(self.grid.size),
                               X,Y )


class EqCircGrid (Grid):
    """
    Provides a circular (non-regular but fairly equidistant) 2D grid for
    the beamforming results.
    
    The grid is on a plane perpendicular to the z-axis. 
    It is defined by lower and upper r-limits and the 
    z co-ordinate, and the increments in r direction.
    """

    #: Inner radius, defaults to 0.0.
    r_min = Float(0.0, 
        desc="inner radius")
    
    #: Outer radius, defaults to 1.0.   
    r_max = Float(1.0, 
        desc="outer radius")

    #: The increment in r- and phi- direction (both in m), defaults to 0.1.
    increment = Float(0.1, 
        desc="step size")
    
    #: Position on z-axis, defaults to 1.0. 
    z = Float(1.0, 
        desc="position on z-axis")
    
    #: minx/max values of the grid
    extent = Property(desc= "Extent of the grid")    
    
    #: Number of grid points, will be calculated automatically.
    nsteps = Property( 
        desc="number of grid points")

    # internal identifier
    digest = Property( 
        depends_on = ['r_min', 'r_max', 'z', 'increment']
        )

    # traits_view = View(
    #         [
    #             ['r_min', '|'], 
    #             ['r_max', 'z', 'increment', 
    #              'size~{grid size}', '|'], 
    #             '-[Map extension]'
    #         ]
    #     )
        
    @property_depends_on('r_min, r_max, increment')
    def _get_nsteps( self ):
        if self.r_min <= self.increment/pi:
            self.r_min = 0
        i = abs(self.increment)
        if i != 0:
            n = 0
            for rv in arange(self.r_min,self.r_max+i*0.1,i):
                n += max(1,int(2*pi*rv/i))# points per circumference
            return n
        return 1

    @property_depends_on('nsteps')
    def _get_size ( self ):
        return self.nsteps

    @property_depends_on('nsteps')
    def _get_shape ( self ):
        return (self.nsteps,)


    @cached_property
    def _get_digest( self ):
        if self.r_min <= self.increment/pi:
            self.r_min = 0
        return digest( self )

   
    @property_depends_on('rmax')
    def _get_extent(self):
        return (-self.r_max, self.r_max, -self.r_max, self.r_max)
        
    def _get_pos( self, polar = False ):
        """
        Calculates grid co-ordinates.
        
        Parameters
        ----------
        polar : boolean
            Flag to trigger the cartesian or polar coordinate output.
            Default is 'False' (=cartesian).     
            
        Returns
        -------
        array of floats of shape (3, :attr:`~EqCircGrid.nsteps`)
            The grid point (x, y, z) or (r, phi, z)-coordinates in one array.
        """

        if self.r_min <= self.increment/pi:
            self.r_min = 0
        
        i = self.increment
        z = self.z        
        
        xpos = empty((3, self.size))
        ind = 0
        if polar:
            for rv in arange(self.r_min,self.r_max+i*0.1,i):
                s,n = modf(2*pi*rv/i) # points per circumference
                n = max(1,int(n))# point in the middle
                r = rv*ones(n)
                phi = linspace(0,2*pi,n,endpoint=False)+2*pi*s
                xpos[0,ind:ind+n] = r
                xpos[1,ind:ind+n] = (phi / pi * 180.0) % 360.0
                xpos[2,ind:ind+n] = z
                ind += n
        else:
            for rv in arange(self.r_min,self.r_max+i*0.1,i):
                s,n = modf(2*pi*rv/i) # points per circumference
                n = max(1,int(n))# point in the middle
                r = rv*ones(n)
                phi = linspace(0,2*pi,n,endpoint=False)+2*pi*s
                xpos[0,ind:ind+n] = r * cos(phi)
                xpos[1,ind:ind+n] = r * sin(phi)
                xpos[2,ind:ind+n] = z
                ind += n

        return xpos


    def index ( self, r, phi ):
        """
        Queries the index for a grid point next to a certain co-ordinate.

        This can be used to query results or co-ordinates at/near a certain
        co-ordinate.
        
        Parameters
        ----------
        r, phi : float
            The polar co-ordinates for which the indices are queried.

        Returns
        -------
        integer
            The index that gives the grid point nearest to the given r, phi
            co-ordinate from an array with the same shape as the grid.            
        """
        if self.r_min <= self.increment/pi:
            self.r_min = 0
        if r < self.r_min or r > self.r_max:
            raise(ValueError, "r-value out of range")
        if phi  <  self.phi_min or phi > self.phi_max:
            raise(ValueError, "phi-value out of range")
        xpos = self.pos
        x = r*cos(phi/180.0*pi)
        y = r*sin(phi/180.0*pi)
        dr2 = (xpos[0,:]-x)**2 + (xpos[1,:]-y)**2
        return argmin(dr2)

    def indices ( self, *r):
        """
        Queries the indices for a subdomain in the grid.
        
        Allows either circle segment or circular subdomains. This can be used to
        mask or to query results from a certain sector or subdomain.
        
        Parameters
        ----------
        r1, phi1, r2, phi2 : float
            If all four paramters are given, then a circle segment sector is
            assumed that is given by two corners (r1, phi1) and (r2, phi2). If
            only three parameters are given, then a circular sector is assumed
            that is given by its center (r1, phi1) and the radius r2.

        Returns
        -------
        2-tuple of arrays of integers or of numpy slice objects
            The indices that can be used to mask/select the grid subdomain from 
            an array with the same shape as the grid.            
        """
        
        if len(r) == 3:
            xpos = self.pos
    
            x = r[0]*cos(r[1]/180.0*pi)
            y = r[0]*sin(r[1]/180.0*pi)
            
            dr2 = (xpos[0,:]-x)**2 + (xpos[1,:]-y)**2
            inds = dr2 <= r[2]**2
            if not any(inds): # if there is none in radius, return nearest
                return argmin(dr2)
            else: # else return indices of points in radius
                return arange(self.size)[inds]
                
        elif len(r) == 4:
            xpos = self._get_pos(polar=True)
            inds0 = (xpos[0] >= r[0]) * (xpos[0] <= r[2]) # '*' = 'and'
            
            
            if r[3] < r[1]:
                inds1 = (xpos[1] >= r[1]) + (xpos[1] <= r[3]) # '+' = 'or'
            else:
                inds1 = (xpos[1] >= r[1]) * (xpos[1] <= r[3])
            
            inds = inds0 * inds1
            return arange(self.size)[inds]
        else:
            xpos = self.pos
            p = Path(array(r).reshape(-1,2))
            inds = p.contains_points(xpos[:2,:].T)
            return arange(self.size)[inds]
           

class EqCircGrid3D (EqCircGrid):
    """
    Provides a cylindrical (non-regular but fairly equidistant) 3D grid for
    the beamforming results.
    
    The grid cylinder is orientated parallel to the z-axis. 
    It is defined by lower and upper r and z-limit and the increments in 
    r and z direction respectively.
    """
    
    #: Minimum z value, defaults to -1.
    z_min = Float(-1.0, 
        desc="minimum z value")
    
    #: Maximum z value, defaults to 1.
    z_max = Float(1.0, 
        desc="maximum z value")
    
    border = Trait('sloppy', 'strict',
                  desc="whether rmax really should be the limit")

    #: Position on z-axis, defaults to center of grid
    z = Float(desc="position on z-axis")
    def _z_default(self): return (self.z_min + self.z_max)/2.

    #: Increment in z-direction (in m), defaults 
    #: to :attr:`~EqCircGrid.increment` for r. Whichever of the two
    #: increment parameters is set last replaces the other.
    #: If dz is to differ from increment it has to be set separately from it
    #: afterwards.
    dz = Float(desc="step size")
    def _dz_default(self): return self.increment
        
    @on_trait_change('increment')
    def reset_dz(self): self.dz = self.increment
    
    #: Number of grid points in r-phi plane, will be calculated automatically.
    nsteps = Property( 
        desc="number of grid points in r-phi plane")
        
    #: Number of grid points along z, will be calculated automatically.
    nzsteps = Property( 
        desc="number of grid points along z")



    # internal identifier
    digest = Property( 
        depends_on = ['r_min', 'r_max', 
                      'z_min', 'z_max', 
                      'increment', 'dz']
        )
    
    # traits_view = View(
    #         [
    #             ['r_min','r_max', '|'], 
    #             ['z_min','z_max', 'increment', 'dz', \
    #              'size~{grid size}', '|'], 
    #             '-[Map extension]'
    #         ]
    #     )

    @property_depends_on('r_min, r_max, increment')
    def _get_nsteps( self ):
        if self.r_min <= self.increment/pi:
            self.r_min = 0
        i = abs(self.increment)
        if i != 0:
            n = 0
            
            (r1,r2) = (self.r_min,self.r_max)
            if self.border == 'sloppy':
                radii = arange(r1,r2+i,i)
            else: #self.border =='strict':
                ni = int(round((r2-r1)/i))+1
                radii = linspace(r1,r2,ni)
            
            for rv in radii:
                n += max(1,int(2*pi*rv/i))# points per circumference
            return n
        return 1
    
    @property_depends_on('z_min, z_max, dz')
    def _get_nzsteps( self ):
        i = abs(self.dz)
        if i != 0:
            return int(round((abs(self.z_max-self.z_min)+i)/i))
        return 1
    
    @property_depends_on('nzsteps, nsteps')
    def _get_size ( self ):
        return self.nzsteps * self.nsteps

    @property_depends_on('nzsteps, nsteps')
    def _get_shape ( self ):
        return (self.nzsteps,self.nsteps)


    @cached_property
    def _get_digest( self ):
        if self.r_min <= self.increment/pi:
            self.r_min = 0
        return digest( self )

    def _get_gpos ( self, polar = False ):
        """
        Calculates grid co-ordinates.
        
        Parameters
        ----------
        polar : boolean
            Flag to trigger the cartesian or polar coordinate output.
            Default is 'False' (=cartesian).     
            
        Returns
        -------
        array of floats of shape (3, :attr:`~EqCircGrid.nsteps`)
            The grid point (x, y, z) or (r, phi, z)-coordinates in one array.
        """

        if self.r_min <= self.increment/pi:
            self.r_min = 0
        
        i = self.increment
        r1 = self.r_min
        r2 = self.r_max
        
        if self.border == 'sloppy':
            radii = arange(r1,r2+i,i)
        else: #self.border =='strict':
            ni = int(round((r2-r1)/i))+1
            radii = linspace(r1,r2,ni)
        
        xpos = empty((3, self.size))
        ind = 0
        if polar:
            for z in linspace(self.z_min, self.z_max, self.nzsteps):
                for rv in radii:
                    s,n = modf(2*pi*rv/i) # points per circumference
                    n = max(1,int(n))# point in the middle
                    r = rv*ones(n)
                    phi = linspace(0,2*pi,n,endpoint=False)+2*pi*s
                    xpos[0,ind:ind+n] = r
                    xpos[1,ind:ind+n] = (phi / pi * 180.0) % 360.0
                    xpos[2,ind:ind+n] = z
                    ind += n
        else:
            for z in linspace(self.z_min, self.z_max, self.nzsteps):
                for rv in radii:
                    s,n = modf(2*pi*rv/i) # points per circumference
                    n = max(1,int(n))# point in the middle
                    r = rv*ones(n)
                    phi = linspace(0,2*pi,n,endpoint=False)+2*pi*s
                    xpos[0,ind:ind+n] = r * cos(phi)
                    xpos[1,ind:ind+n] = r * sin(phi)
                    xpos[2,ind:ind+n] = z
                    ind += n

        return xpos


    def index ( self, r, phi, z ):
        """
        Queries the index for a grid point next to a certain co-ordinate.

        This can be used to query results or co-ordinates at/near a certain
        co-ordinate.
        
        Parameters
        ----------
        r, phi, z : float
            The cylindrical co-ordinates for which the indices are queried.

        Returns
        -------
        integer
            The index that gives the grid point nearest to the given r, phi, z
            co-ordinate from an array with the same shape as the grid.            
        """

        if self.r_min <= self.increment/pi:
            self.r_min = 0
        if r < self.r_min or r > self.r_max:
            raise(ValueError, "r-value out of range")
        if phi  <  self.phi_min or phi > self.phi_max:
            raise(ValueError, "phi-value out of range")
        if z  <  self.z_min or z > self.z_max:
            raise(ValueError, "z-value out of range")
        xpos = self.pos
        x = r*cos(phi/180.0*pi)
        y = r*sin(phi/180.0*pi)
        dr2 = (xpos[0,:]-x)**2 + (xpos[1,:]-y)**2 + (xpos[2,:]-z)**2
        return argmin(dr2)
        
    def indices ( self, *r):
        """
        Queries the indices for a subdomain in the grid.
        
        Allows either cylinder segment or cylindrical subdomains. This can be used to
        mask or to query results from a certain sector or subdomain.
        
        Parameters
        ----------
        r1, phi1, r2, phi2 : float
            If all four paramters are given, then a cylinder segment sector is
            assumed that is given by two corners (r1, phi1) and (r2, phi2). If
            only three parameters are given, then a cylindrical sector is assumed
            that is given by its center (r1, phi1) and the radius r2.

        Returns
        -------
        2-tuple of arrays of integers or of numpy slice objects
            The indices that can be used to mask/select the grid subdomain from 
            an array with the same shape as the grid.            
        """
        
        if len(r) == 3:
            xpos = self.pos
    
            x = r[0]*cos(r[1]/180.0*pi)
            y = r[0]*sin(r[1]/180.0*pi)
            
            dr2 = (xpos[0,:]-x)**2 + (xpos[1,:]-y)**2
            inds = dr2 <= r[2]**2
            if not any(inds): # if there is none in radius, return nearest
                return argmin(dr2)
            else: # else return indices of points in radius
                #TODO: the following doesn't return the actual indices but 
                #a True/False array which can also be used for indexing
                # -> better change this to real indices later
                return inds.reshape(self.shape) 
                #arange(self.size)[inds]
                
        elif len(r) == 4:
            np0 = self.shape[0] # number of EqCircGrid planes
            np1 = self.shape[1] # number of points in one plane
            xpos = self._get_pos(polar=True).reshape(3,np0,np1)[:,0,:]
            inds0 = (xpos[0] >= r[0]) * (xpos[0] <= r[2]) # '*' = 'and'
            
            
            if r[3] < r[1]:
                inds1 = (xpos[1] >= r[1]) + (xpos[1] <= r[3]) # '+' = 'or'
            else:
                inds1 = (xpos[1] >= r[1]) * (xpos[1] <= r[3])
            
            inds = inds0 * inds1
            # return r-phi slices (..[inds]) of all z-planes (s_..)
            return s_[0:np0], arange(np1)[inds]
        else:
            xpos = self.pos
            np0 = self.shape[0] # number of EqCircGrid planes
            np1 = self.shape[1] # number of points in one plane
            p = Path(array(r).reshape(-1,2))
            inds = p.contains_points(xpos[:2,:].T)
            return inds.reshape(self.shape) 
    