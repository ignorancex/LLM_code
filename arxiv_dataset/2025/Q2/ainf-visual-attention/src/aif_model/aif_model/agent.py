import aif_model.vae as vae
import aif_model.config as c
import aif_model.networks as networks
import aif_model.utils as utils
import torch
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os
from scipy.special import softmax

def printf(*args, **kwargs):
    '''
    Debugging print functions
    '''

    if c.printing: print(*args, **kwargs)

class Agent:
    '''
    Active Inference agent
    '''

    def __init__(self):

        # Load networks
        package_share_directory = get_package_share_directory(c.package_name)
        vae_path = os.path.join(package_share_directory, 'resource', c.vae_path)

        self.vectors = np.zeros((3,2)) # vectors of target locations from center, and focus point, used in displaying

        self.vae  = vae.VAE( # VAE network
        latent_dim=c.latent_size,
        encoder=networks.Encoder(in_chan=c.channels, latent_dim=c.latent_size),
        decoder=networks.Decoder(out_chan=c.channels, latent_dim=c.latent_size))
        self.vae.load(vae_path)

        self.belief_dim = c.needs_len + c.prop_len + c.latent_size + c.focus_len # symbolic_cue(need), proprioceptive belief, visual belief, visual focus 

        # Initialization of variables
        self.mu = np.zeros((c.n_orders, self.belief_dim), dtype="float32") 
        self.mu_dot = np.zeros_like(self.mu)

        self.a = np.zeros(c.prop_len)
        self.a_dot = np.zeros_like(self.a)

        self.E_i = np.zeros((c.num_intentions, self.belief_dim))

        self.beta_index = 0
        weights = [] 
        for i in range(c.num_intentions):
            builder = np.array([1]*c.needs_len+[1]*c.prop_len+[1e-1]*c.latent_size+[1]*c.focus_len)
            weights.append(builder)

        self.beta_weights = weights

        # Generative models (trivial/simple)
        self.G_p = utils.shift_rows(np.eye(self.belief_dim, c.prop_len),c.needs_len)

        self.G_n = np.eye(self.belief_dim, c.needs_len)

    def get_p(self):
        '''
        Get predicitons
        '''
        # Visual
        input_, output = self.vae.predict_visual(torch.tensor(self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.latent_size]).unsqueeze(0))
        # Proprioceptive
        p_prop = self.mu[0].dot(self.G_p)
        # Needs
        p_needs = self.mu[0].dot(self.G_n)

        P = [p_needs, p_prop, output.detach().squeeze().cpu().numpy()]

        return P, [input_, output]
    
    def get_prop_intentions(self):
        '''
        Get proprioceptive intentions
        '''
        targets = np.zeros((c.num_intentions,c.prop_len))
        
        targets = self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len*c.num_intentions] # grab visual positions of objects
        targets = np.reshape(targets,(c.num_intentions,c.prop_len)) # reshape
        targets = utils.denormalize(targets) # convert from range [-1,1] to [0,width]
        printf("Target in pixels:",targets)
        self.vectors[:2,:] = np.array(utils.normalize(targets))
        targets = utils.pixels_to_angles(targets) # convert to angles

        result = np.zeros_like(targets) + self.mu[0,c.needs_len:c.needs_len+c.prop_len]
        if self.mu[0, c.needs_len+c.prop_len+c.prop_len]>0: # if object visible in image
            result += targets # add relative target angle to global camera angle

        return result
    
    def get_vis_intentions(self):
        '''
        Get visual intentions
        '''
        red_cue = self.mu[0,2] # grab cue strength
        red_exist = self.mu[0, c.needs_len+c.prop_len+c.prop_len] # red sphere existence
        sm = softmax((red_cue,red_exist))
        
        old = self.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+2]
        cue = self.mu[0,:2]
        ending = self.mu[0,c.needs_len+c.prop_len+2:c.needs_len+c.prop_len+c.latent_size]
        mix = sm[0]*old + sm[1]*cue # mix visual belief and cue 

        result = np.zeros((2,c.latent_size))
        result[0]=np.concatenate((mix,ending))

        return result

    def get_focus_intentions(self):
        '''
        Get focus intentions
        '''
        result = np.zeros((c.num_intentions,c.focus_len))
        result[:,1:] = self.mu[0,-2:] # previous focus belief
        if self.mu[0,2]>0.1:
            result[:,1:] = self.mu[0,:2]

        amp = self.mu[0,c.needs_len+c.prop_len+c.latent_size]
        result[:,0] = 0.1*self.mu[0,2] - 0.05*amp # decay

        printf("Focus intentions:", result)

        return result

    
    def get_i(self):
        """
        Get intentions
        """
        targets_prop = self.get_prop_intentions()
        targets_vis = self.get_vis_intentions()
        targets_needs = np.tile(self.mu[0,:c.needs_len],(c.num_intentions,1))
        targets_focus = self.get_focus_intentions()

        targets = np.concatenate((targets_needs,targets_prop,targets_vis,targets_focus),axis=1) # concatenate to get final matrix of shape NUM_INTENTIONS x (NEEDS_LEN + PROP_LEN + LATENT_SIZE + FOCUS_LEN)

        return targets
    
    def get_e_s(self, S, P):
        """
        Get sensory prediction errors
        """
        return [s - p for s, p in zip(S, P)]
    
    def get_e_mu(self, I):
        """
        Get dynamics prediction errors
        """
        self.E_i = (I - self.mu[0]) * c.k

        return self.mu[1] - self.E_i
    
    def get_sensory_precisions(self, S):
        '''
        Get sensory precisions
        '''
        
        pi_vis, dPi_dmu0_vis, dPi_dmu1_vis = utils.pi_foveate(np.ones((c.height,c.width))*c.pi_vis, self.mu[0]) # covert attention
        pi_vis_s, dPi_dS0, dPi_dS1 = utils.pi_presence(np.ones((c.height,c.width))*c.pi_vis, S[2]) # bottom-up object presence attention

        dim = c.needs_len+c.prop_len+c.latent_size+c.focus_len

        Pi = [np.ones(dim) * c.pi_need,
              np.ones(dim) * c.pi_prop, 
              (pi_vis + pi_vis_s)/2]  # /2 if pi_presence is active
        
        dPi_dmu0 = [np.zeros((dim,dim)), 
                    np.zeros((dim,dim)), 
                    dPi_dmu0_vis/2]
        
        dPi_dmu1 = [np.zeros((dim,dim)), 
                    np.zeros((dim,dim)), 
                    dPi_dmu1_vis/2]

        dPi_dS = [np.zeros((dim,dim)), 
                    np.zeros((dim,dim)), 
                    dPi_dS0/2]

        return Pi, dPi_dmu0, dPi_dmu1, dPi_dS
    
    def get_intention_precisions(self):
        '''
        Get intention precisions
        '''
        self.beta_index = np.argmax(self.mu[0,2:c.needs_len])
        self.beta = [np.ones(self.belief_dim)*1e-10] * c.num_intentions
        self.beta[self.beta_index] = self.beta_weights[self.beta_index]

        dGamma_dmu0 = [np.zeros((self.belief_dim, self.belief_dim))] * c.num_intentions 

        dGamma_dmu1 = [np.zeros((self.belief_dim, self.belief_dim))] * c.num_intentions

        return self.beta, dGamma_dmu0, dGamma_dmu1
    
    
    def get_likelihood(self, E_s, grad_v, Pi):
        """
        Get likelihood components
        """
        lkh = {}
        lkh['need'] = Pi[0] * E_s[0].dot(self.G_n.T)

        lkh['prop'] = Pi[1] * E_s[1].dot(self.G_p.T)

        lkh['vis'] = self.vae.get_grad(*grad_v, torch.from_numpy(Pi[2])*E_s[2])
        lkh['vis'] = np.concatenate((np.zeros((c.needs_len+c.prop_len)),lkh['vis'],np.zeros(c.focus_len))) 

        return lkh


    def attention(self, precision, derivative, error):
        '''
        Attention components in free-energy update
        '''
        total = np.zeros(self.belief_dim)
        for i in range(len(precision)):
            component1 = 0.5 * np.mean(np.expand_dims(1/precision[i], axis=-1) * derivative[i], axis=tuple(range(derivative[i].ndim - 1)))
            component1[-3] = 0.1 * component1[-3]
            component1[-2:] = c.attn_damper1 * component1[-2:]
            component2 = -0.5 * np.sum(np.expand_dims(error[i]**2, axis=-1) * derivative[i], axis=tuple(range(derivative[i].ndim - 1)))
            component2[-3] = c.attn_damper2 * component2[-3]
            
            total += component1 + component2

        return total, component1, component2

    def get_mu_dot(self, lkh, E_s, E_mu, Pi, Gamma, dPi_dmu0, dGamma_dmu0, dPi_dmu1, dGamma_dmu1):
        """
        Get belief update
        """
        self.mu_dot = np.zeros_like(self.mu)

        # Pad needs and proprioceptive error to size of mu
        e_s = [np.concatenate([E_s[0],np.zeros(self.belief_dim - c.needs_len)]),np.concatenate([E_s[1],np.zeros(self.belief_dim - c.prop_len)]), torch.mean(E_s[2],dim=(0,1))]

        # Intention components
        forward_i = np.zeros((self.belief_dim)) 
        for g, e in zip(Gamma, np.array(E_mu)):
            forward_i += g * e

        generative = lkh['prop'] + lkh['need'] + lkh['vis']
        backward = - c.k * forward_i

        # Attention calculation
        bottom_up0, _, _ = self.attention(Pi,dPi_dmu0,e_s)
        top_down0, _, _ = self.attention(Gamma,dGamma_dmu0,E_mu)

        bottom_up1, _, _ = self.attention(Pi, dPi_dmu1,[0]*3) # No sensory error for second order
        top_down1, _, _ = self.attention(Gamma,dGamma_dmu1,[0]*c.num_intentions) # No intention error for second order

        printf("self.mu[1]", self.mu[1], np.linalg.norm(self.mu[1]))

        self.mu_dot[0] = self.mu[1] + generative + backward + bottom_up0 + top_down0 #
        self.mu_dot[1] = -forward_i + bottom_up1 + top_down1
        printf("mu_dot0 before clip:",self.mu_dot[0], np.linalg.norm(self.mu_dot[0]))
        self.mu_dot = np.clip(self.mu_dot,-0.25,0.25) # clip mu update

    def get_a_dot(self, likelihood, Pi, dPi_dS, E_s):
        """
        Get action update
        """
        e_s = [np.concatenate([E_s[0],np.zeros(self.belief_dim - c.needs_len)]),np.concatenate([E_s[1],np.zeros(self.belief_dim - c.prop_len)]), torch.mean(E_s[2],dim=(0,1))]
        e_prop = likelihood["prop"].dot(self.G_p)
        printf("eprop",e_prop)

        d_mu_lkh_prop = -c.dt * e_prop

        attn, c1, c2 = self.attention(Pi, dPi_dS, e_s)
        focus = np.zeros((1,2))
        focus[0] = attn[-2:]
        focus = utils.denormalize(focus)
        attn_comp = utils.pixels_to_angles(focus)[0]

        self.a_dot = d_mu_lkh_prop + attn_comp # both components 
        printf("d_mu_lkh_prop", d_mu_lkh_prop)
        printf("attn_comp", attn_comp)
        printf("a_dot",self.a_dot)

    def integrate(self):
        """
        Integrate with gradient descent
        """
        
        # Update belief
        self.mu[0] += c.dt * self.mu_dot[0]
        self.mu[1] += c.dt * self.mu_dot[1]

        self.mu[:,c.needs_len+c.prop_len+c.latent_size] = np.clip(self.mu[:,c.needs_len+c.prop_len+c.latent_size],0.1,1) # clip mu_amp
        printf("self.mu[0]",self.mu[0])
        self.vectors[2,:] = self.mu[0,-2:]

        # Update action
        self.a += c.dt * self.a_dot
        self.a = np.clip(self.a, -c.a_max, c.a_max)

    def init_belief(self, needs, prop, visual):
        """
        Initialize belief
        """
        visual_state =self.vae.predict_latent(visual.squeeze()).detach().squeeze().numpy() # use VAE encoder to encode initial visual belief
        focus = np.array([c.pi_vis,0,0])

        self.mu[0] = np.concatenate((needs, prop, visual_state,focus)) # initialize with beliefs about needs, proprioceptive and visual state
        printf("mu initialized to:",self.mu[0])

        self.beta_index = np.argmax(needs[2:c.needs_len])
        self.beta = [np.ones(self.belief_dim)*1e-10] * c.num_intentions
        self.beta[self.beta_index] = self.beta_weights[self.beta_index]

    def inference_step(self, S):
        """
        Run an inference step
        """
        
        printf("mu:",self.mu[0])
        
        # Get predictions
        P, grad_v = self.get_p()

        # Get intentions
        I = self.get_i()

        # Get sensory prediction errors
        E_s = self.get_e_s(S, P)

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)

        # Get sensory precisions
        Pi, dPi_dmu0, dPi_dmu1, dPi_dS = self.get_sensory_precisions(S)

        # Get intention precisions
        Gamma, dGamma_dmu0, dGamma_dmu1 = self.get_intention_precisions()

        # Get likelihood components
        likelihood = self.get_likelihood(E_s, grad_v, Pi)

        # Get belief update
        self.get_mu_dot(likelihood, E_s, E_mu, Pi, Gamma, dPi_dmu0, dGamma_dmu0, dPi_dmu1, dGamma_dmu1)

        # Get action update
        self.get_a_dot(likelihood, Pi, dPi_dS, E_s) # E_s[0] * self.pi_s[0] 

        # Update
        self.integrate()

        # Show visual sensory and predicted data
        utils.show_SP(S, P, self.vectors)

        return self.a
