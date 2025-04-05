from .mean_vfe import MeanVFE, RADAR_MeanVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE, DynamicPillarVFESimple2D, Radar_DynamicPillarVFESimple2D, Radar_DynamicPillarVFESimple2D_Test
from .dynamic_voxel_vfe import DynamicVoxelVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'RADAR_MeanVFE':RADAR_MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'DynamicPillarVFESimple2D': DynamicPillarVFESimple2D,
    'DynamicVoxelVFE': DynamicVoxelVFE,
    'Radar_DynamicPillarVFESimple2D' : Radar_DynamicPillarVFESimple2D,
    'Radar_DynamicPillarVFESimple2D_Test':Radar_DynamicPillarVFESimple2D_Test
}
