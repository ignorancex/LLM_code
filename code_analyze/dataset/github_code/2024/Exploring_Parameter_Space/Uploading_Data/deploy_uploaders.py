import os
from subprocess import Popen

# GWL = GWLandscape(token='1446a1b3c8be718461913e2f6397e5bda92bf4959b65dd284374381a7f89cedc')
# PUB = GWL.create_publication(
#     author='Adam Boesky',
#     title='The Binary Black Hole Merger Rate Deviates from a Simple Delayed Cosmic Star Formation Rate: The Impact of Metallicity and Delay Time distributions',
#     arxiv_id='000000',
#     year=2023
# )
# PATH_TO_DATA = '/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta'
PATH_TO_DATA = '/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_sigma_remnant_prescription'


def deploy_uploaders():
    # alpha_vals = ['0.1', '0.5', '2.0', '10.0']  # All the alpha values
    # beta_vals = ['0.25', '0.5', '0.75']         # All the beta values
    # alpha_beta_datasets = {}
    sigma_vals = ['30', '265', '750']
    rmp_vals = ['D', 'R', 'M']
    sigma_rmp_datasets = {}

    ps = []
    # for alpha in alpha_vals:
    for sig in sigma_vals:
        # alpha_beta_datasets[alpha] = {}
        sigma_rmp_datasets[sig] = {}
        # for beta in beta_vals:
        for rmp in rmp_vals:
            # print(f'Creating dataset for alpha = {alpha}, beta = {beta}')
            print(f'Creating dataset for sigma = {sig}, RMP = {rmp}')

            # Set up env vars
            # fpath = os.path.join(PATH_TO_DATA, f'output_alpha_CE_{alpha}_beta_{beta}/COMPAS_Output_Weighted.h5')
            fpath = os.path.join(PATH_TO_DATA, F'output_sigma_{sig}_remnant_prescription_{rmp}/COMPAS_Output_Weighted.h5')
            # os.environ['ALPHA'] = alpha
            # os.environ['BETA'] = beta
            os.environ['SIGMA'] = sig
            os.environ['RMP'] = rmp
            os.environ['FPATH'] = fpath

            # Run uploader
            # sbatch_command = f'sbatch --wait /n/home04/aboesky/berger/Exploring_Parameter_Space/Uploading_Data/run_uploader.sh {alpha} {beta} {fpath}'
            sbatch_command = f'sbatch --wait /n/home04/aboesky/berger/Exploring_Parameter_Space/Uploading_Data/run_uploader.sh {sig} {rmp} {fpath}'
            proc = Popen(sbatch_command, shell=True)
            ps.append(proc)

    exit_codes = [p.wait() for p in ps]  # wait for processes to finish
    print(f'exit_codes = {exit_codes}')
    return exit_codes 


if __name__ == '__main__':
    deploy_uploaders()
