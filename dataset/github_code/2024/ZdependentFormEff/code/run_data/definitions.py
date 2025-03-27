import os
import h5py as h5
from astropy.table import Table


sim_flags_dict = {
      # Fiducial 
    "OldWinds_RemFryer2012": "--mass-loss-prescription BELCZYNSKI2010 --OB-mass-loss-prescription VINK2001 --VMS-mass-loss-prescription VINK2011 --RSG-mass-loss-prescription NJ90 --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN ",
      # Old winds with Muller mandel rem
    "OldWinds_RemMullerMandel": "--mass-loss-prescription BELCZYNSKI2010 --OB-mass-loss-prescription VINK2001 --VMS-mass-loss-prescription VINK2011 --RSG-mass-loss-prescription NJ90 --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription MULLERMANDEL --kick-magnitude-distribution MULLERMANDEL ",
      # no BH kick
    "OldWinds_RemFryer2012_noBHkick": "--mass-loss-prescription BELCZYNSKI2010 --OB-mass-loss-prescription VINK2001 --VMS-mass-loss-prescription VINK2011 --RSG-mass-loss-prescription NJ90 --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN \
 --black-hole-kicks ZERO --kick-magnitude-sigma-CCSN-BH 0 --natal-kick-for-PPISN FALSE ",
      # no NS or BH kick
    "OldWinds_RemFryer2012_noNSBHkick": "--mass-loss-prescription BELCZYNSKI2010 --OB-mass-loss-prescription VINK2001 --VMS-mass-loss-prescription VINK2011 --RSG-mass-loss-prescription NJ90 --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --black-hole-kicks ZERO --kick-magnitude-distribution ZERO --kick-magnitude-sigma-ECSN 0.0 ",
      # NO main sequence mass loss
    "OldWinds_RemFryer2012_noMSwinds": "--mass-loss-prescription BELCZYNSKI2010 --RSG-mass-loss-prescription NJ90 --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN \
 --OB-mass-loss-prescription NONE --VMS-mass-loss-prescription NONE ",
      # NO WR winds
    "OldWinds_RemFryer2012_noWRwinds": "--mass-loss-prescription BELCZYNSKI2010 --OB-mass-loss-prescription VINK2001 --VMS-mass-loss-prescription VINK2011 --RSG-mass-loss-prescription NJ90 --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN --wolf-rayet-multiplier 0 ",
      # NO winds
    "OldWinds_RemFryer2012_NOwinds": "--mass-loss-prescription BELCZYNSKI2010 --overall-wind-mass-loss-multiplier 0 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN ",
      # NOTE New winds with Fryer remnants NOTE 
    "NewWinds_RemFryer2012": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN ",
      # New winds with Muller Mandel remnants
    "NewWinds_RemMullerMandel": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription MULLERMANDEL --kick-magnitude-distribution MULLERMANDEL ",
       # New winds with Fryer remnants and no BH kick
    "NewWinds_RemFryer2012_noBHkick": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
  --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN \
 --black-hole-kicks ZERO --kick-magnitude-sigma-CCSN-BH 0 --natal-kick-for-PPISN FALSE ",
       # New winds with Fryer remnants and no NS or BH kick
    "NewWinds_RemFryer2012_noNSBHkick": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
  --remnant-mass-prescription FRYER2012 --black-hole-kicks ZERO --kick-magnitude-distribution ZERO --kick-magnitude-sigma-ECSN 0.0 ",
      # New winds NO WR winds
    "NewWinds_RemFryer2012_noWRwinds": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN --wolf-rayet-multiplier 0 ",
      # New winds strong WR winds
    "NewWinds_RemFryer2012_strongWRwinds": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN --wolf-rayet-multiplier 5 ",
       # New winds extreme WR winds
    "NewWinds_RemFryer2012_extremeWRwinds": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN --wolf-rayet-multiplier 100 ",
       # New winds NO main sequence mass loss
    "NewWinds_RemFryer2012_noMSwinds":  "--OB-mass-loss-prescription NONE --VMS-mass-loss-prescription NONE --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN ",
       # NO winds
    "RemFryer2012_NOwinds": " --overall-wind-mass-loss-multiplier 0 --OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN ",
       # New winds with Fryer remnants - NO CHE
    "NewWinds_RemFryer2012_noCHE": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN --chemically-homogeneous-evolution-mode NONE ",
        # New winds with Fryer remnants Bel
    "NewWinds_RemFryer2012_WRBELCZYNSKI2010": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN",
      # New winds but old WR & MS winds !! NOTE the VMS prescription does not allow VINK2001, instead, up VMS_MASS_THRESHOLD >150 in constants.h
    "NewWinds_RemFryer2012_oldMS_oldWR": "--OB-mass-loss-prescription VINK2001 --VMS-mass-loss-prescription VINK2011 --RSG-mass-loss-prescription DECIN2023  --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN",
      # old WR & MS & RSG winds !! NOTE the VMS prescription does not allow VINK2001, instead, up VMS_MASS_THRESHOLD >150 in constants.h
    "NewWinds_RemFryer2012_oldMS_oldWR_oldRSG": "--OB-mass-loss-prescription VINK2001 --VMS-mass-loss-prescription VINK2011 --RSG-mass-loss-prescription NJ90  --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN",
       # New winds with Fryer remnants - old RSG 
    "NewWinds_RemFryer2012_oldRSG": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription NJ90  --WR-mass-loss-prescription SANDERVINK2023 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN ",
        # New winds with Fryer remnants - old RSG & old WR
    "NewWinds_RemFryer2012_oldWR_oldRSG": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription NJ90  --WR-mass-loss-prescription BELCZYNSKI2010 \
 --remnant-mass-prescription FRYER2012 --kick-magnitude-distribution MAXWELLIAN ",
        # New winds with Fryer remnants - old RSG 
    "NewWinds_RemMullerMandel_oldRSG": "--OB-mass-loss-prescription VINK2021 --VMS-mass-loss-prescription SABHAHIT2023 --RSG-mass-loss-prescription NJ90  --WR-mass-loss-prescription SANDERVINK2023 \
       --remnant-mass-prescription MULLERMANDEL --kick-magnitude-distribution MULLERMANDEL ",
}