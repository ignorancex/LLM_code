[DIRECTORIES]
SAVE_DIR   = results/mnist/amd/    #add / at the end please
MODEL_SAVE_DIR   = models/mnist/     #add / at the end please

[NETWORK]
MODEL      = AE
NET        = mnist
BATCH_SIZE = 1000                                   
INPUT_DIM  = 20                                        
OUTPUT_DIM = 784
NET_LAYERS = 3                #No of layers for generative network

[OPTIMIZATION]
EPOCHS      = 20                                       
OPT_LR_RATE = 0.0005   
OPT_EPSILON = 1e-6
OPT_ALPHA   = 0.9

[GFLOW]
ODE_LOG_BASE = 4
ODE_TAU    = 1.e4                                  
ODE_ALPHA  = 1.0                                 
ODE_METHOD = amd
ODE_RTOL   = 1e-5
ODE_ATOL   = 1e-9   #used as amd threshold
ODE_TSTEPS = 2 
ODE_ADTAU  = false
ODE_ADTAU_MIN = 1e-5          #Decrese threshold, for less change: lower D  
ODE_ADTAU_MAX = 2e-4          #Increase threshold, for less change: increase I 
ODE_ERRDT = 1

[AMD]
AMD_DTSTEP = 1
AMD_DTMAXSTEP = 10                                  
AMD_DTFACTOR = 0.5                                 
AMD_DTITER = 20
AMD_CONV_GRADDT = 1e-4
AMD_CONV_PERCTAU = 0.95  

[ACTORS]
TENSORBOARD = false
TNSE = true   
VALIDATION= false
TEST= false
TRAIN= true
SAVE_MODEL = false
RUNNAME=None

