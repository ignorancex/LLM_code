# Image config
height = 32
width = 32
channels = 3
horizontal_fov = 1.3963

# Training config
latent_size = 8
learning_rate = 1e-3
variance = 1
n_epochs = 100
n_batch = 512
beta = 2 # for betaVAE

# Agent config
package_name = "aif_model"
vae_path = "vae_lat8_b2_e5_nb256.pt" # vae-disentangled_state_dict_scaled.pt
n_orders = 2 # orders of belief
num_intentions = 2 # number of possible objects
prop_len = 2 # size of proprioceptive belief
needs_len = 3 # size of needs/cueing belief
focus_len = 3 # size of focus belief: amplitude, x_position, y_position
k = 0.06 # intention error gain
pi_prop = 0.5 # proprioceptive precision baseline
pi_need = 0.5 # sensory cue precision baseline
pi_vis =  2 # visual precision baseline
foveation_sigma = 2
attn_damper1 = 1e-2
attn_damper2 = 1e-3
dt = 0.4 
a_max = 2.0

printing = False

limits = [[90,90],[-90,-90]] # [[pitch min, yaw min],[pitch max, yaw max]]
noise = 5e-5 # action noise

update_frequency = 20

def set_pi_vis(value):
    global pi_vis
    pi_vis = value