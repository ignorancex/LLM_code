[DIRECTORIES]
SAVE_DIR   = results/mnist/        #add / at the end please

[NETWORK]
MODEL      = GFE
NET        = mnist
BATCH_SIZE = 20                                   
INPUT_DIM  = 20                                        
OUTPUT_DIM = 784
NET_LAYERS = 3                #No of layers for generative network

[OPTIMIZATION]
EPOCHS      = 1000                                       
OPT_LR_RATE = 0.0005   
OPT_EPSILON = 1e-6
OPT_ALPHA   = 0.9

[GFLOW]
ODE_LOG_BASE = 4
ODE_TAU    = 1.5e4                                  
ODE_METHOD = rk4
ODE_RTOL   = 1e-7
ODE_ATOL   = 1e-9
ODE_TSTEPS = 100 
ODE_ADTAU  = false
ODE_ADTAU_MIN = 1e-5          #Decrese threshold, for less change: lower D  
ODE_ADTAU_MAX = 2e-4          #Increase threshold, for less change: increase I 
ODE_ERRDT = 4

[ACTORS]
TENSORBOARD = true  
TNSE = false   