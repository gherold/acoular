# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Classes to deal with rotating stuff.

.. autosummary::
    :toctree: generated/

    grids
    environments
    microphones
    fbeamform
    tprocess    
"""

from .environments import (
    AxialRotatingFlowEnvironment,
    EnvironmentRot,
    EnvironmentRotFlow,    
)

from .fbeamform import (
    SteeringVectorInduct,
    SteeringVectorModeTransformer,
)

from .grids import (
    CircGrid, 
    CircMesh, 
    EqCircGrid, 
    EqCircGrid3D, 
    GridMesh,
)


from .microphones import (
    MicGeomCirc, 
    MicRing,
)

from .spectra import PowerSpectraDR

from .tprocess import (
    AngleTrajectory,
    Trigger, 
    FeatureTrigger,
    SpaceModesTransformer,
    VirtualRotator,
    VirtualRotatorAngle,
    VirtualRotatorModal,
    VirtualRotatorSpatial,
    RotationalSpeedDetector,
    RotationalSpeedDetector2,
    RotationalSpeedDetector3,
    RotationalSpeed,
)

from.trajectory import (
    AngleTrajectory,
    TrajectoryAnglesFromTrigger,
)
