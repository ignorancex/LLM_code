### ====== PRISM HARMONIZATION ======

import torch
import os
import matplotlib.pyplot as plt
import pickle
from model import PRISM
from dataset import PRISM_MRI_Dataset

## Model and data initialization
'''
Harmonization method 1 (preferred): Uniform harmonization - reduces inter-site AND intra-site variations
Harmonization method 2: Image-wise harmonization
'''
harmonization_method = 1
data_dir = '/kaggle/input'
model_dir = '/kaggle/working'
out_dir = f'/kaggle/working/harmonized_{harmonization_method}'
modality = 'T2'
target='Guys'

prism_g = PRISM(intensity_levels=5, latent_dim=2, num_sites=4, gpu_id=0, modality=modality, modalities = ['T2'])
prism_h = PRISM(intensity_levels=5, latent_dim=2, num_sites=4, gpu_id=0, modality=modality, modalities = ['T2'])
prism_i = PRISM(intensity_levels=5, latent_dim=2, num_sites=4, gpu_id=0, modality=modality, modalities = ['T2'])
prism_a = PRISM(intensity_levels=5, latent_dim=2, num_sites=4, gpu_id=0, modality=modality, modalities = ['T2'])

# Target site: Guys
prism_g.anatomy_encoder.load_state_dict(torch.load(f'{model_dir}/prism-anatomy-encoder_guys.pth', map_location='cpu'))
prism_g.style_encoder.load_state_dict(torch.load(f'{model_dir}/prism-style-encoder_guys.pth', map_location='cpu'))
prism_g.decoder.load_state_dict(torch.load(f'{model_dir}/prism-decoder_guys.pth', map_location='cpu'))

# Source sites: HH, IOP and ADNI
prism_h.anatomy_encoder.load_state_dict(torch.load(f'{model_dir}/prism-anatomy-encoder_hh.pth', map_location='cpu'))
prism_i.anatomy_encoder.load_state_dict(torch.load(f'{model_dir}/prism-anatomy-encoder_iop.pth', map_location='cpu'))
prism_a.anatomy_encoder.load_state_dict(torch.load(f'{model_dir}/prism-anatomy-encoder_adni1.pth', map_location='cpu'))

# guys_ds = torch.load(f'{data_dir}/IXI-Guys-{data_mode}.pt', map_location='cpu')
# hh_ds = torch.load(f'{data_dir}/IXI-HH-{data_mode}.pt', map_location='cpu')
# iop_ds = torch.load(f'{data_dir}/IXI-IOP-{data_mode}.pt', map_location='cpu')
# adni_ds = torch.load(f'{data_dir}/ADNI1-{data_mode}.pt', map_location='cpu')

# Load both train and test datasets for each site
guys_ds_train = torch.load(f'{data_dir}/IXI-Guys-train.pt', map_location='cpu')
guys_ds_test = torch.load(f'{data_dir}/IXI-Guys-test.pt', map_location='cpu')
hh_ds_train = torch.load(f'{data_dir}/IXI-HH-train.pt', map_location='cpu')
hh_ds_test = torch.load(f'{data_dir}/IXI-HH-test.pt', map_location='cpu')
iop_ds_train = torch.load(f'{data_dir}/IXI-IOP-train.pt', map_location='cpu')
iop_ds_test = torch.load(f'{data_dir}/IXI-IOP-test.pt', map_location='cpu')
adni_ds_train = torch.load(f'{data_dir}/ADNI1-train.pt', map_location='cpu')
adni_ds_test = torch.load(f'{data_dir}/ADNI1-test.pt', map_location='cpu')

datasets = {
    'Guys': {'model': prism_g, 'data': [(guys_ds_train, 'train'), (guys_ds_test, 'test')]},
    'HH': {'model': prism_h, 'data': [(hh_ds_train, 'train'), (hh_ds_test, 'test')]},
    'IOP': {'model': prism_i, 'data': [(iop_ds_train, 'train'), (iop_ds_test, 'test')]},
    'ADNI1': {'model': prism_a, 'data': [(adni_ds_train, 'train'), (adni_ds_test, 'test')]}
}

## Harmonization
'''
Harmonization method 1 (preferred): Uniform harmonization - reduces inter-site AND intra-site variations
Harmonization method 2: Image-wise harmonization
'''

style_codes = {site: [] for site in datasets.keys()}
style_codes_og = {site: [] for site in ['HH', 'IOP', 'ADNI1']}
anatomies = {site: [] for site in datasets.keys()}
masks = {site: [] for site in datasets.keys()}
names = {site: [] for site in datasets.keys()}

if harmonization_method == 1:
    with torch.set_grad_enabled(False):
        # Set all models to eval mode
        for site_info in datasets.values():
            site_info['model'].anatomy_encoder.eval()
            site_info['model'].style_encoder.eval()

        # Process each site
        for site, site_info in datasets.items():
            if not os.path.exists(f'{out_dir}/{site}'):
                os.makedirs(f'{out_dir}/{site}')
            
            for dataset, data_mode in site_info['data']:
                for subject in dataset:
                    # ... rest of your processing code, using site_info['model'] instead of individual models ...
                    image = subject[site_info['model'].modality]['image'].to(site_info['model'].device).unsqueeze(1)
                    mask = subject[site_info['model'].modality]['mask'].to(site_info['model'].device).unsqueeze(1)
                    _, anatomy = site_info['model'].get_anatomy_representations(image, mask)
                    style_code, _, _ = prism_g.get_style_code(image)

                    anatomies[site].append(anatomy.detach().squeeze())
                    style_codes[site].append(style_code.detach().cpu().squeeze())
                    masks[site].append(mask.detach().squeeze())
                    names[site].append(subject[site_info['model'].modality]['subject_id'])

                    if site != target:
                        style_code_og, _, _ = site_info['model'].get_style_code(image)
                        style_codes_og[site].append(style_code_og.detach().cpu().squeeze())

    # Uniform Harmonization
    for site, site_info in datasets.values():
        style_code = torch.mean(style_codes[site], dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(site_info['model'].device)
        with torch.set_grad_enabled(False):
            prism_g.decoder.eval()

            for i in range(len(anatomies[site])):
                anatomy = anatomies[site][i].unsqueeze(0).unsqueeze(0)
                mask = masks[site][i].unsqueeze(0)
                harmonized = prism_g.decode(anatomy, style_code, mask)
                subid = names[site][i]
                if not os.path.exists(f'{out_dir}/{site}/{data_mode}/{subid}'):
                    os.makedirs(f'{out_dir}/{site}/{data_mode}/{subid}')
                
                plt.imsave(f'{out_dir}/{site}/{data_mode}/{subid}/IXI-{site}-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')

elif harmonization_method==2:
    with torch.set_grad_enabled(False):
        # Set all models to eval mode
        for site_info in datasets.values():
            site_info['model'].anatomy_encoder.eval()
        prism_g.style_encoder.eval()
        prism_g.decoder.eval()

        # Process each site
        for site, site_info in datasets.items():
            if not os.path.exists(f'{out_dir}/{site}'):
                os.makedirs(f'{out_dir}/{site}')
            
            for dataset, data_mode in site_info['data']:
                # Image-level harmonization
                for subject in dataset:
                    image = subject[site_info['model'].modality]['image'].to(site_info['model'].device).unsqueeze(1)
                    mask = subject[site_info['model'].modality]['mask'].to(site_info['model'].device).unsqueeze(1)
                    _, anatomy = site_info['model'].get_anatomy_representations(image, mask)
                    style_code, _, _ = prism_g.get_style_code(image)
                    harmonized = prism_g.decode(anatomy, style_code, mask)
                    subid = subject[site_info['model'].modality]['subject_id']
                    if not os.path.exists(f'{out_dir}/{site}/{data_mode}/{subid}'):
                        os.makedirs(f'{out_dir}/{site}/{data_mode}/{subid}')
                    
                    plt.imsave(f'{out_dir}/{site}/{data_mode}/{subid}/IXI-{site}-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')

# Save original and translated latent style codes for each site - for visualization.py
if len(style_codes[target])!=0:
    for site in datasets.keys():
        with open(f'{model_dir}/style_codes_{site.lower()}.pkl', 'wb') as f:
            pickle.dump(style_codes[site], f)
        if site!=target:
            with open(f'{model_dir}/style_codes_{site.lower()}_og.pkl', 'wb') as f:
                pickle.dump(style_codes_og[site], f)
            

# Old version - ignore
'''
if harmonization_method==1:
    style_codes_guys = []
    style_codes_hh = []
    style_codes_iop = []
    style_codes_adni = []

    style_codes_hh_og = []
    style_codes_iop_og = []
    style_codes_adni_og = []

    anatomies_guys = []
    anatomies_hh = []
    anatomies_iop = []
    anatomies_adni = []

    mask_guys = []
    mask_hh = []
    mask_iop = []
    mask_adni = []

    names_guys = []
    names_hh = []
    names_iop = []
    names_adni = []

    with torch.set_grad_enabled(False):
        prism_g.anatomy_encoder.eval()
        prism_g.style_encoder.eval()
        prism_h.anatomy_encoder.eval()
        prism_h.style_encoder.eval()
        prism_i.anatomy_encoder.eval()
        prism_i.style_encoder.eval()
        prism_a.anatomy_encoder.eval()
        prism_a.style_encoder.eval()
        # Guys
        if not os.path.exists(f'{out_dir}/Guys'):
            os.makedirs(f'{out_dir}/Guys')
        for subject in guys_ds:
            image = subject[prism_g.modality]['image'].to(prism_g.device).unsqueeze(1)
            mask = subject[prism_g.modality]['mask'].to(prism_g.device).unsqueeze(1)
            _, anatomy = prism_g.get_anatomy_representations(image, mask)
            style_code, _, _ = prism_g.get_style_code(image)

            anatomies_guys.append(anatomy.detach().squeeze())
            style_codes_guys.append(style_code.detach().cpu().squeeze())
            mask_guys.append(mask.detach().squeeze())
            names_guys.append(subject[prism_g.modality]['subject_id'])


        # HH
        if not os.path.exists(f'{out_dir}/HH'):
            os.makedirs(f'{out_dir}/HH')
        for subject in hh_ds:
            image = subject[prism_h.modality]['image'].to(prism_h.device).unsqueeze(1)
            mask = subject[prism_h.modality]['mask'].to(prism_h.device).unsqueeze(1)
            _, anatomy = prism_h.get_anatomy_representations(image, mask)
            style_code, _, _ = prism_g.get_style_code(image)  ###
            
            anatomies_hh.append(anatomy.detach().squeeze())
            style_codes_hh.append(style_code.detach().cpu().squeeze())
            mask_hh.append(mask.detach().squeeze())
            names_hh.append(subject[prism_h.modality]['subject_id'])

            style_code_og, _, _ = prism_h.get_style_code(image)
            style_codes_hh_og.append(style_code_og.detach().cpu().squeeze())


        # IOP
        if not os.path.exists(f'{out_dir}/IOP'):
            os.makedirs(f'{out_dir}/IOP')
        for subject in iop_ds:
            image = subject[prism_i.modality]['image'].to(prism_i.device).unsqueeze(1)
            mask = subject[prism_i.modality]['mask'].to(prism_i.device).unsqueeze(1)
            _, anatomy = prism_i.get_anatomy_representations(image, mask)
            style_code, _, _ = prism_g.get_style_code(image)
            
            anatomies_iop.append(anatomy.detach().squeeze())
            style_codes_iop.append(style_code.detach().cpu().squeeze())
            mask_iop.append(mask.detach().squeeze())
            names_iop.append(subject[prism_i.modality]['subject_id'])

            style_code_og, _, _ = prism_i.get_style_code(image)
            style_codes_iop_og.append(style_code_og.detach().cpu().squeeze())


        # ADNI
        if not os.path.exists(f'{out_dir}/ADNI1'):
            os.makedirs(f'{out_dir}/ADNI1')
        for subject in adni_ds:
            image = subject[prism_a.modality]['image'].to(prism_a.device).unsqueeze(1)
            mask = subject[prism_a.modality]['mask'].to(prism_a.device).unsqueeze(1)
            _, anatomy = prism_a.get_anatomy_representations(image, mask)
            style_code, _, _ = prism_g.get_style_code(image)
            
            anatomies_adni.append(anatomy.detach().squeeze())
            style_codes_adni.append(style_code.detach().cpu().squeeze())
            mask_adni.append(mask.detach().squeeze())
            names_adni.append(subject[prism_a.modality]['subject_id'])

            style_code_og, _, _ = prism_a.get_style_code(image)
            style_codes_adni_og.append(style_code_og.detach().cpu().squeeze())

    # Save original and translated latent style codes for each site - for visualization.py

    with open(f'{model_dir}/style_codes_guys.pkl', 'wb') as f:
        pickle.dump(style_codes_guys, f)
    with open(f'{model_dir}/style_codes_hh.pkl', 'wb') as f:
        pickle.dump(style_codes_hh, f)
    with open(f'{model_dir}/style_codes_iop.pkl', 'wb') as f:
        pickle.dump(style_codes_iop, f)
    with open(f'{model_dir}/style_codes_adni.pkl', 'wb') as f:
        pickle.dump(style_codes_adni, f)

    with open(f'{model_dir}/style_codes_hh_og.pkl', 'wb') as f:
        pickle.dump(style_codes_hh_og, f)
    with open(f'{model_dir}/style_codes_iop_og.pkl', 'wb') as f:
        pickle.dump(style_codes_iop_og, f)
    with open(f'{model_dir}/style_codes_adni_og.pkl', 'wb') as f:
        pickle.dump(style_codes_adni_og, f)

    # Uniform Harmonization

    style_code_guys = torch.mean(style_codes_guys, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(prism_g.device)
    style_code_hh = torch.mean(style_codes_hh, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(prism_h.device)
    style_code_iop = torch.mean(style_codes_iop, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(prism_i.device)
    style_code_adni = torch.mean(style_codes_adni, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(prism_a.device)

    with torch.set_grad_enabled(False):
        prism_g.decoder.eval()

        # Guys
        for i in range(len(anatomies_guys)):
            anatomy = anatomies_guys[i].unsqueeze(0).unsqueeze(0)
            mask = mask_guys[i].unsqueeze(0)
            harmonized = prism_g.decode(anatomy, style_code_guys, mask)
            subid = names_guys[i]
            if not os.path.exists(f'{out_dir}/Guys/{data_mode}/{subid}'):
                os.makedirs(f'{out_dir}/Guys/{data_mode}/{subid}')
            
            plt.imsave(f'{out_dir}/Guys/{data_mode}/{subid}/IXI-Guys-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')

        # HH
        for i in range(len(anatomies_hh)):
            anatomy = anatomies_hh[i].unsqueeze(0).unsqueeze(0)
            mask = mask_hh[i].unsqueeze(0)
            harmonized = prism_g.decode(anatomy, style_code_hh, mask)
            subid = names_hh[i]
            if not os.path.exists(f'{out_dir}/HH/{data_mode}/{subid}'):
                os.makedirs(f'{out_dir}/HH/{data_mode}/{subid}')
            
            plt.imsave(f'{out_dir}/HH/{data_mode}/{subid}/IXI-HH-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')

        # IOP
        for i in range(len(anatomies_iop)):
            anatomy = anatomies_iop[i].unsqueeze(0).unsqueeze(0)
            mask = mask_iop[i].unsqueeze(0)
            harmonized = prism_g.decode(anatomy, style_code_iop, mask)
            subid = names_iop[i]
            if not os.path.exists(f'{out_dir}/IOP/{data_mode}/{subid}'):
                os.makedirs(f'{out_dir}/IOP/{data_mode}/{subid}')
            
            plt.imsave(f'{out_dir}/IOP/{data_mode}/{subid}/IXI-IOP-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')

        # ADNI
        for i in range(len(anatomies_adni)):
            anatomy = anatomies_adni[i].unsqueeze(0).unsqueeze(0)
            mask = mask_adni[i].unsqueeze(0)
            harmonized = prism_g.decode(anatomy, style_code_adni, mask)
            subid = names_adni[i]
            if not os.path.exists(f'{out_dir}/ADNI1/{data_mode}/{subid}'):
                os.makedirs(f'{out_dir}/ADNI1/{data_mode}/{subid}')
            
            plt.imsave(f'{out_dir}/ADNI1/{data_mode}/{subid}/ADNI1-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')

elif harmonization_method==2:
    with torch.set_grad_enabled(False):
        prism_g.anatomy_encoder.eval()
        prism_h.anatomy_encoder.eval()
        prism_i.anatomy_encoder.eval()
        prism_a.anatomy_encoder.eval()
        prism_g.style_encoder.eval()
        prism_g.decoder.eval()

        # Guys
        if not os.path.exists(f'{out_dir}/Guys'):
            os.makedirs(f'{out_dir}/Guys')
        for subject in guys_ds:
            image = subject[prism_g.modality]['image'].to(prism_g.device).unsqueeze(1)
            mask = subject[prism_g.modality]['mask'].to(prism_g.device).unsqueeze(1)
            _, anatomy = prism_g.get_anatomy_representations(image, mask)
            style_code, _, _ = prism_g.get_style_code(image)
            harmonized = prism_g.decode(anatomy, style_code, mask)
            subid = subject[prism_g.modality]['subject_id']
            if not os.path.exists(f'{out_dir}/Guys/{data_mode}/{subid}'):
                os.makedirs(f'{out_dir}/Guys/{data_mode}/{subid}')
            
            plt.imsave(f'{out_dir}/Guys/{data_mode}/{subid}/IXI-Guys-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')
            

        # HH
        if not os.path.exists(f'{out_dir}/HH'):
            os.makedirs(f'{out_dir}/HH')
        for subject in hh_ds:
            image = subject[prism_h.modality]['image'].to(prism_h.device).unsqueeze(1)
            mask = subject[prism_h.modality]['mask'].to(prism_h.device).unsqueeze(1)
            _, anatomy = prism_h.get_anatomy_representations(image, mask)
            style_code, _, _ = prism_g.get_style_code(image)  ###
            harmonized = prism_g.decode(anatomy, style_code, mask)
            subid = subject[prism_g.modality]['subject_id']
            if not os.path.exists(f'{out_dir}/HH/{data_mode}/{subid}'):
                os.makedirs(f'{out_dir}/HH/{data_mode}/{subid}')
            
            plt.imsave(f'{out_dir}/HH/{data_mode}/{subid}/IXI-HH-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')
            

        # IOP
        if not os.path.exists(f'{out_dir}/IOP'):
            os.makedirs(f'{out_dir}/IOP')
        for subject in iop_ds:
            image = subject[prism_i.modality]['image'].to(prism_i.device).unsqueeze(1)
            mask = subject[prism_i.modality]['mask'].to(prism_i.device).unsqueeze(1)
            _, anatomy = prism_i.get_anatomy_representations(image, mask)
            style_code, _, _ = prism_g.get_style_code(image)
            harmonized = prism_g.decode(anatomy, style_code, mask)
            subid = subject[prism_g.modality]['subject_id']
            if not os.path.exists(f'{out_dir}/IOP/{data_mode}/{subid}'):
                os.makedirs(f'{out_dir}/IOP/{data_mode}/{subid}')
            
            plt.imsave(f'{out_dir}/IOP/{data_mode}/{subid}/IXI-IOP-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')


        # ADNI
        if not os.path.exists(f'{out_dir}/ADNI1'):
            os.makedirs(f'{out_dir}/ADNI1')
        for subject in adni_ds:
            image = subject[prism_a.modality]['image'].to(prism_a.device).unsqueeze(1)
            mask = subject[prism_a.modality]['mask'].to(prism_a.device).unsqueeze(1)
            _, anatomy = prism_a.get_anatomy_representations(image, mask)
            style_code, _, _ = prism_g.get_style_code(image)
            harmonized = prism_g.decode(anatomy, style_code, mask)
            subid = subject[prism_g.modality]['subject_id']
            if not os.path.exists(f'{out_dir}/ADNI1/{data_mode}/{subid}'):
                os.makedirs(f'{out_dir}/ADNI1/{data_mode}/{subid}')
            
            plt.imsave(f'{out_dir}/ADNI1/{data_mode}/{subid}/ADNI1-{subid}-{modality}_harmonized.png', harmonized.squeeze().cpu().numpy(), cmap='gray')
'''