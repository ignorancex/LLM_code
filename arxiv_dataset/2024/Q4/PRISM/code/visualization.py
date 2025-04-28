### ====== PRISM VISUALIZATION ======

import seaborn as sns
import torch
import pickle
import matplotlib.pyplot as plt

dir = '/kaggle/working'

with open(f'{dir}/style_codes_guys.pkl', 'rb') as f:
    style_codes_guys = pickle.load(f)
with open(f'{dir}/style_codes_hh.pkl', 'rb') as f:
    style_codes_hh = pickle.load(f)
with open(f'{dir}/style_codes_iop.pkl', 'rb') as f:
    style_codes_iop = pickle.load(f)
with open(f'{dir}/style_codes_adni.pkl', 'rb') as f:
    style_codes_adni = pickle.load(f)

with open(f'{dir}/style_codes_hh_og.pkl', 'rb') as f:
    style_codes_hh_og = pickle.load(f)
with open(f'{dir}/style_codes_iop_og.pkl', 'rb') as f:
    style_codes_iop_og = pickle.load(f)
with open(f'{dir}/style_codes_adni_og.pkl', 'rb') as f:
    style_codes_adni_og = pickle.load(f)

style_codes_guys = torch.stack(style_codes_guys) # Target site has a fixed style
style_codes_hh = torch.stack(style_codes_hh)
style_codes_iop = torch.stack(style_codes_iop)
style_codes_adni = torch.stack(style_codes_adni)

style_codes_hh_og = torch.stack(style_codes_hh_og)
style_codes_iop_og = torch.stack(style_codes_iop_og)
style_codes_adni_og = torch.stack(style_codes_adni_og)


# Pre-harmonization latent style visualization
plt.figure(figsize=(12, 6))
marker = 'P'
sns.scatterplot(x=style_codes_guys[:, 0], y=style_codes_guys[:, 1], color='darkcyan', marker=marker, label='Site Guys', s=50)
sns.scatterplot(x=style_codes_hh_og[:, 0], y=style_codes_hh_og[:, 1], color='crimson', marker=marker, label='Site HH', s=50)
sns.scatterplot(x=style_codes_iop_og[:, 0], y=style_codes_iop_og[:, 1], color='springgreen', marker=marker, label='Site IOP', s=50)
sns.scatterplot(x=style_codes_adni_og[:, 0], y=style_codes_adni_og[:, 1], color='orange', marker=marker, label='Site ADNI', s=50)
# plt.xlim(-23, 17)
# plt.ylim(-27, 23)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Style distributions (pre-harmonization)')
plt.legend(fontsize=15)
plt.show()


# Post-harmonization latent style visualization
plt.figure(figsize=(12, 6))
sns.scatterplot(x=style_codes_guys[:, 0], y=style_codes_guys[:, 1], color='darkcyan', marker=marker, label='Site Guys', s=50)
sns.scatterplot(x=style_codes_hh[:, 0], y=style_codes_hh[:, 1], color='crimson', marker=marker, label='Site HH', s=50)
sns.scatterplot(x=style_codes_iop[:, 0], y=style_codes_iop[:, 1], color='springgreen', marker=marker, label='Site IOP', s=50)
sns.scatterplot(x=style_codes_adni[:, 0], y=style_codes_adni[:, 1], color='orange', marker=marker, label='Site ADNI', s=50)
# plt.xlim(-23, 17)
# plt.ylim(-27, 23)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Style distributions (post-harmonization)')
plt.legend(fontsize=15)
plt.show()