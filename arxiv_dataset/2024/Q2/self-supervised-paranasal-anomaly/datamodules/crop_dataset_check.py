import pandas as pd 
import nibabel 
import numpy as np
import uuid
import glob

cfg = {
        "smax_l": {"coronal":  { "min": {"mean":151,"std":1.899835519},
                                "max": {"mean":198.5,"std":1.414213562}
                                },

                 "sagittal":  { "min": {"mean":39.5,"std":1.322875656},
                                 "max": {"mean": 75.75,"std":1.785357107}
                                },

                 "axial":      { "min": {"mean":68.875,"std":1.964529206},
                                 "max": {"mean": 113.5,"std":1.802775638}
                                }
                 },

        "smax_r": {"coronal":  { "min": {"mean":151,"std":2.175861898},
                                "max": {"mean":198.375,"std":1.316956719}
                                },

                 "sagittal":  { "min": {"mean":95.25,"std":1.71391365},
                                 "max": {"mean": 128.875,"std":2.315032397}
                                },

                 "axial":      { "min": {"mean":66.375,"std":6.479535091},
                                 "max": {"mean": 111.5,"std":7.465145348}
                                }
                 },

        "sphen": {"coronal":  { "min": {"mean":123.75,"std":7.066647013},
                                "max": {"mean":158.375,"std":4.370036867}
                                },

                 "sagittal":  { "min": {"mean":63.625,"std":3.533323506},
                                 "max": {"mean": 103.875,"std":4.0754601}
                                },

                 "axial":      { "min": {"mean":99.625,"std":2.446298224},
                                 "max": {"mean": 127.625,"std":2.287875652}
                                }
                 },

        "sfront": {"coronal":  { "min": {"mean":185,"std":2.618614683},
                                "max": {"mean":208.2857143,"std":1.829464068}
                                },

                 "sagittal":  { "min": {"mean":54.14285714,"std":8.773801447},
                                 "max": {"mean": 109.4285714,"std":10.18201696}
                                },

                 "axial":      { "min": {"mean":126,"std":4.035556255},
                                 "max": {"mean": 156.8571429,"std":6.685347975}
                                }
                 },


        "seth": {"coronal":  { "min": {"mean":152.5714286,"std":2.258769757},
                                "max": {"mean":197.7142857,"std":4.025429372}
                                },

                 "sagittal":  { "min": {"mean":71.57142857,"std":9.897433186},
                                 "max": {"mean":101.8571429,"std":1.456862718}
                                },

                 "axial":      { "min": {"mean":104.5714286,"std":1.916629695},
                                 "max": {"mean": 129.8571429,"std":3.090472522}
                                }
                 },


        "nose": {"coronal":  { "min": {"mean":147.3333333,"std":4.229525847},
                                "max": {"mean":201.6666667,"std":2.924988129}
                                },

                 "sagittal":  { "min": {"mean":68.5,"std":1.802775638},
                                 "max": {"mean":99.33333333,"std":1.885618083}
                                },

                 "axial":      { "min": {"mean":73.16666667,"std":3.89087251},
                                 "max": {"mean": 123.8333333,"std":2.477678125}
                                }
                 },
        
      }


import numpy as np



location = cfg["nose"]
#Function to crop out sub volume
cmin,cmax =  (location["coronal"]["min"]["mean"]),(location["coronal"]["max"]["mean"])
cmin_std,cmax_std =  (location["coronal"]["min"]["std"]),(location["coronal"]["max"]["std"])
smin,smax =  (location["sagittal"]["min"]["mean"]),(location["sagittal"]["max"]["mean"])
smin_std,smax_std =  (location["sagittal"]["min"]["std"]),(location["sagittal"]["max"]["std"])
amin,amax =  (location["axial"]["min"]["mean"]),(location["axial"]["max"]["mean"])
amin_std,amax_std =  (location["axial"]["min"]["std"]),(location["axial"]["max"]["std"])

positions  = {"coronal":(cmin,cmax,cmin_std,cmax_std),"saggital":(smin,smax,smin_std,smax_std),"axial":(amin,amax,amin_std,amax_std)}

num_samples = 10000

#setting seed
desired_std_dev = 1
np.random.seed(10)

coordinates = {}

for key in positions.keys():

    min_pos,max_pos,std_min,std_max = positions[key]
    

    arr_min = np.random.normal(min_pos, std_min, num_samples)
    arr_max = np.random.normal(max_pos, std_max, num_samples)

    coordinate  = (arr_min + arr_max)/2


    if key == "saggital":

        print("X", np.mean(coordinate),np.std(coordinate))
    
    elif key == "coronal":
        print("Y", np.mean(coordinate),np.std(coordinate))
    
    elif key == "axial":
        print("Z", np.mean(coordinate),np.std(coordinate))




    



"""

samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=num_samples)

actual_mean = np.mean(samples)
actual_std = np.std(samples)
print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))

zero_mean_samples = samples - (actual_mean)

zero_mean_mean = np.mean(zero_mean_samples)
zero_mean_std = np.std(zero_mean_samples)
print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))

scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std)
scaled_mean = np.mean(scaled_samples)
scaled_std = np.std(scaled_samples)
print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))

final_samples = scaled_samples + desired_mean
final_mean = np.mean(final_samples)
final_std = np.std(final_samples)
print("Final samples stats     : mean = {:.4f} stdv = {:.4f}".format(final_mean, final_std))
"""