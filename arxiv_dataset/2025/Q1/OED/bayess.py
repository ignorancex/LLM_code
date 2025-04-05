import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

plt.rcParams.update({'font.size': 16})

# Set up the x-axis
x = np.linspace(30, 100, 1000)

# Prior distribution (blue)
prior_mean, prior_std = 60, 10
prior = norm.pdf(x, prior_mean, prior_std)

# Single observation (green dot)
observation = 75

# Likelihood (green)
likelihood_mean = observation
likelihood_std = 5  # Adjust this value to change the width of the likelihood curve
likelihood = norm.pdf(x, likelihood_mean, likelihood_std)

# Posterior distribution (purple)
posterior_mean = (prior_mean/prior_std**2 + observation/likelihood_std**2) / (1/prior_std**2 + 1/likelihood_std**2)
posterior_std = np.sqrt(1 / (1/prior_std**2 + 1/likelihood_std**2))
posterior = norm.pdf(x, posterior_mean, posterior_std)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, prior, 'b-', label='Prior (m)')
plt.plot(x, likelihood, 'g-', label='Likelihood (d|m)')
plt.plot(x, posterior, 'red', label='Posterior (m|d)')

# Plot single observation as green dot on x-axis
plt.plot(observation, 0, 'go', markersize=10, label='Observation (d)')

# Customize the plot
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Set axis limits
plt.ylim(-0.005, 0.12)
plt.xlim(30, 100)

plt.xticks([])
plt.yticks([])

# Remove x and y axis labels
plt.xlabel('')
plt.ylabel('')

plt.box(False)

# Calculate the y-values for the modes
prior_mode_y = norm.pdf(prior_mean, prior_mean, prior_std)
posterior_mode_y = norm.pdf(posterior_mean, posterior_mean, posterior_std)



# Show the plot
plt.tight_layout()
plt.savefig('bayes.png')




# Add black arrow from prior mode to posterior mode
arrow = plt.annotate('', xy=(posterior_mean, posterior_mode_y), xytext=(prior_mean, prior_mode_y),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

# Calculate angle of the arrow
dx = posterior_mean - prior_mean
dy = posterior_mode_y - prior_mode_y
angle = math.pi/4 +0.15#math.atan2(dy, dx)

# Calculate the perpendicular offset for the text
offset = 0.01  # Adjust this value to move text up/down
perp_dx = -math.sin(angle) * offset
perp_dy = math.cos(angle) * offset


# Add label for the arrow, parallel to it and slightly above
text_x = (prior_mean + posterior_mean) / 2 + perp_dx
text_y = (prior_mode_y + posterior_mode_y) / 2 + perp_dy
plt.text(text_x,
         text_y,
         r'$KL(\mu_{\text{post}}||\mu_{\text{pr}})$',
         ha='center',
         va='center', 
         rotation=math.degrees(angle))#, rotation_mode='anchor')
plt.savefig('kl.png')

plt.show()




































# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import math

# plt.rcParams.update({'font.size': 16})

# # Set up the x-axis
# x = np.linspace(30, 100, 1000)

# # Prior distribution (blue)
# prior_mean, prior_std = 60, 10
# prior = norm.pdf(x, prior_mean, prior_std)

# # Single observation (green dot)
# observation = 75

# # Likelihood (green)
# likelihood_mean = observation
# likelihood_std = 5  # Adjust this value to change the width of the likelihood curve
# likelihood = norm.pdf(x, likelihood_mean, likelihood_std)

# # Posterior distribution (purple)
# posterior_mean = (prior_mean/prior_std**2 + observation/likelihood_std**2) / (1/prior_std**2 + 1/likelihood_std**2)
# posterior_std = np.sqrt(1 / (1/prior_std**2 + 1/likelihood_std**2))
# posterior = norm.pdf(x, posterior_mean, posterior_std)

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(x, prior, 'b-', label='Prior (m)')
# plt.plot(x, likelihood, 'g-', label='Likelihood (d|m)')
# plt.plot(x, posterior, 'red', label='Posterior (m|d)')

# # Plot single observation as green dot on x-axis
# plt.plot(observation, 0, 'go', markersize=10, label='Observation (d)')



# # Calculate the y-values for the modes
# prior_mode_y = norm.pdf(prior_mean, prior_mean, prior_std)
# posterior_mode_y = norm.pdf(posterior_mean, posterior_mean, posterior_std)

# # Add black arrow from prior mode to posterior mode
# arrow = plt.annotate('', xy=(posterior_mean, posterior_mode_y), xytext=(prior_mean, prior_mode_y),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

# # Calculate angle of the arrow
# dx = posterior_mean - prior_mean
# dy = posterior_mode_y - prior_mode_y
# angle = math.atan2(dy, dx) * 180 / math.pi

# # Add label for the arrow, parallel to it and slightly above
# text_x = (prior_mean + posterior_mean) / 2
# text_y = (prior_mode_y + posterior_mode_y) / 2 + 0.005  # Adjust this value to move text up/down
# plt.text(text_x, text_y, r'$KL(\mu_{post}||\mu_{pr})$', ha='center', va='center', 
#          rotation=angle, rotation_mode='anchor')



# # Customize the plot
# # plt.xlabel('μ')
# # plt.ylabel('Probability Density')
# # plt.title('Bayesian Inference with Single Observation')
# plt.legend()
# plt.grid(True, linestyle=':', alpha=0.7)

# # Set axis limits
# plt.ylim(-0.005, 0.12)
# plt.xlim(30, 100)


# plt.xticks([])
# plt.yticks([])

# # Remove x and y axis labels
# plt.xlabel('')
# plt.ylabel('')

# plt.box(False)
    
# # Show the plot
# plt.tight_layout()
# plt.savefig('bayes.png')
# plt.show()
