import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from deep_decoder import deep_decoder_recover
from forward_model import get_forward_model
from iagan import iagan_recover
from mgan import mgan_recover
from model.began import Generator128
from model.biggan import BigGanSkip
from model.dcgan import Generator as dcgan_generator
from model.vae import VAE
from recover import recover, recover_dct
from settings import baseline_settings, forward_models, recovery_settings
from utils import (dict_to_str, get_baseline_results_folder, get_images_folder,
                   get_results_folder, load_target_image, load_trained_net,
                   psnr)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

BASE_DIR = './runs'


def lasso_cs_images(args):
    if args.set_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(BASE_DIR, exist_ok=True)
    if args.model in ['lasso-dct-64', 'lasso-dct-128']:
        recover_fn = recover_dct
    else:
        raise NotImplementedError()

    metadata = baseline_settings[args.model]
    assert len(metadata['n_measure']) == len(metadata['lasso_coeff'])

    data_split = Path(args.img_dir).name
    for img_name in tqdm(sorted(os.listdir(args.img_dir)),
                         desc='Images',
                         leave=True,
                         disable=args.disable_tqdm):
        # Load image and get filename without extension
        orig_img = load_target_image(os.path.join(args.img_dir, img_name),
                                     metadata['img_size']).numpy().transpose(
                                         [1, 2, 0])
        img_basename, _ = os.path.splitext(img_name)

        for n_measure, lasso_coeff in zip(
                tqdm(metadata['n_measure'],
                     desc='N_measure',
                     leave=False,
                     disable=args.disable_tqdm), metadata['lasso_coeff']):

            # Before doing recovery, check if results already exist
            # and possibly skip
            recovered_name = 'recovered.npy'
            results_folder = get_baseline_results_folder(
                image_name=img_basename,
                model=args.model,
                split=data_split,
                n_measure=n_measure,
                lasso_coeff=lasso_coeff,
                base_dir=BASE_DIR)

            os.makedirs(results_folder, exist_ok=True)

            recovered_path = results_folder / recovered_name
            if os.path.exists(recovered_path) and not args.overwrite:
                print(f'{recovered_path} already exists, skipping...')
                continue

            recovered_img = recover_fn(orig_img, n_measure, lasso_coeff,
                                       metadata['img_size'])

            # Make images folder
            img_folder = get_images_folder(split=data_split,
                                           image_name=img_basename,
                                           img_size=metadata['img_size'],
                                           base_dir=BASE_DIR)
            os.makedirs(img_folder, exist_ok=True)

            # Save original image if needed
            original_img_path = img_folder / 'original.npy'
            if not os.path.exists(original_img_path):
                np.save(original_img_path, orig_img)

            # Save recovered image and metadata
            np.save(recovered_path, recovered_img)
            pickle.dump(metadata, open(results_folder / 'metadata.pkl', 'wb'))
            pickle.dump(psnr(recovered_img, orig_img),
                        open(results_folder / 'psnr.pkl', 'wb'))


def gan_images(args):
    if args.set_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(BASE_DIR, exist_ok=True)

    def reset_gen():
        if args.model.startswith('began'):
            gen = Generator128(64)
            if 'untrained' not in args.model:
                gen = load_trained_net(gen, (
                    './checkpoints/celeba_began.withskips.bs32.cosine.min=0.25'
                    '.n_cuts=0/gen_ckpt.49.pt'))
            gen = gen.eval().to(DEVICE)
            img_size = 128
        elif args.model.startswith('beta_vae'):
            gen = VAE()
            if 'untrained' not in args.model:
                t = torch.load(
                    './vae_checkpoints/vae_bs=128_beta=0.1/epoch_19.pt')
                gen.load_state_dict(t)
            gen = gen.eval().to(DEVICE)
            gen = gen.decoder
            img_size = 128
        elif args.model.startswith('biggan'):
            gen = BigGanSkip().to(DEVICE)
            img_size = 512
        elif args.model.startswith('dcgan'):
            gen = dcgan_generator()
            if 'untrained' not in args.model:
                t = torch.load(
                    ('./dcgan_checkpoints/netG.epoch_24.n_cuts_0.bs_64'
                     '.b1_0.5.lr_0.0002.pt'))
                gen.load_state_dict(t)
            gen = gen.eval().to(DEVICE)
            img_size = 64

        elif args.model.startswith('vanilla_vae'):
            gen = VAE()
            if 'untrained' not in args.model:
                t = torch.load(
                    './vae_checkpoints/vae_bs=128_beta=1.0/epoch_19.pt')
                gen.load_state_dict(t)
            gen = gen.eval().to(DEVICE)
            gen = gen.decoder
            img_size = 128
        else:
            raise NotImplementedError()
        return gen, img_size

    gen, img_size = reset_gen()
    img_shape = (3, img_size, img_size)
    metadata = recovery_settings[args.model]
    n_cuts_list = metadata['n_cuts_list']
    del (metadata['n_cuts_list'])

    z_init_mode_list = metadata['z_init_mode']
    limit_list = metadata['limit']
    assert len(z_init_mode_list) == len(limit_list)
    del (metadata['z_init_mode'])
    del (metadata['limit'])

    forwards = forward_models[args.model]

    data_split = Path(args.img_dir).name
    for img_name in tqdm(sorted(os.listdir(args.img_dir)),
                         desc='Images',
                         leave=True,
                         disable=args.disable_tqdm):
        # Load image and get filename without extension
        # If untrained, reset generator for every image
        if "untrained" in args.model:
            gen, _ = reset_gen()
        orig_img = load_target_image(os.path.join(args.img_dir, img_name),
                                     img_size).to(DEVICE)
        img_basename, _ = os.path.splitext(img_name)

        for n_cuts in tqdm(n_cuts_list,
                           desc='N_cuts',
                           leave=False,
                           disable=args.disable_tqdm):
            metadata['n_cuts'] = n_cuts
            for i, (f, f_args_list) in enumerate(
                    tqdm(forwards.items(),
                         desc='Forwards',
                         leave=False,
                         disable=args.disable_tqdm)):
                for f_args in tqdm(f_args_list,
                                   desc=f'{f} Args',
                                   leave=False,
                                   disable=args.disable_tqdm):

                    f_args['img_shape'] = img_shape
                    forward_model = get_forward_model(f, **f_args)

                    for z_init_mode, limit in zip(
                            tqdm(z_init_mode_list,
                                 desc='z_init_mode',
                                 leave=False), limit_list):
                        metadata['z_init_mode'] = z_init_mode
                        metadata['limit'] = limit

                        # Before doing recovery, check if results already exist
                        # and possibly skip
                        recovered_name = 'recovered.pt'
                        results_folder = get_results_folder(
                            image_name=img_basename,
                            model=args.model,
                            n_cuts=n_cuts,
                            split=data_split,
                            forward_model=forward_model,
                            recovery_params=dict_to_str(metadata),
                            base_dir=BASE_DIR)

                        os.makedirs(results_folder, exist_ok=True)

                        recovered_path = results_folder / recovered_name
                        if os.path.exists(
                                recovered_path) and not args.overwrite:
                            print(
                                f'{recovered_path} already exists, skipping...'
                            )
                            continue

                        if args.run_name is not None:
                            current_run_name = (
                                f'{img_basename}.n_cuts={n_cuts}'
                                f'.{forward_model}.z_lr={metadata["z_lr"]}'
                                f'.z_init={z_init_mode}.limit={limit}'
                                f'.{args.run_name}')
                        else:
                            current_run_name = None

                        recovered_img, distorted_img, _ = recover(
                            orig_img, gen, metadata['optimizer'], n_cuts,
                            forward_model, z_init_mode, limit,
                            metadata['z_lr'], metadata['n_steps'],
                            metadata['restarts'], args.run_dir,
                            current_run_name, args.disable_tqdm)

                        # Make images folder
                        img_folder = get_images_folder(split=data_split,
                                                       image_name=img_basename,
                                                       img_size=img_size,
                                                       base_dir=BASE_DIR)
                        os.makedirs(img_folder, exist_ok=True)

                        # Save original image if needed
                        original_img_path = img_folder / 'original.pt'
                        if not os.path.exists(original_img_path):
                            torch.save(orig_img, original_img_path)

                        # Save distorted image if needed
                        if forward_model.viewable:
                            distorted_img_path = img_folder / f'{forward_model}.pt'
                            if not os.path.exists(distorted_img_path):
                                torch.save(distorted_img, distorted_img_path)

                        # Save recovered image and metadata
                        torch.save(recovered_img, recovered_path)
                        pickle.dump(
                            metadata,
                            open(results_folder / 'metadata.pkl', 'wb'))
                        p = psnr(recovered_img, orig_img)
                        pickle.dump(p, open(results_folder / 'psnr.pkl', 'wb'))


def iagan_images(args):
    if args.set_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(BASE_DIR, exist_ok=True)

    def reset_gen():
        if args.model in ['iagan_began_cs']:
            gen = Generator128(64)
            gen = load_trained_net(
                gen,
                ('./checkpoints/celeba_began.withskips.bs32.cosine.min=0.25'
                 '.n_cuts=0/gen_ckpt.49.pt'))
            gen = gen.eval().to(DEVICE)
            img_size = 128
        elif args.model in ['iagan_dcgan_cs']:
            gen = dcgan_generator()
            t = torch.load(('./dcgan_checkpoints/netG.epoch_24.n_cuts_0.bs_64'
                            '.b1_0.5.lr_0.0002.pt'))
            gen.load_state_dict(t)
            gen = gen.eval().to(DEVICE)
            img_size = 64

        elif args.model in ['iagan_vanilla_vae_cs']:
            gen = VAE()
            t = torch.load('./vae_checkpoints/vae_bs=128_beta=1.0/epoch_19.pt')
            gen.load_state_dict(t)
            gen = gen.eval().to(DEVICE)
            gen = gen.decoder
            img_size = 128
        else:
            raise NotImplementedError()
        return gen, img_size

    metadata = recovery_settings[args.model]

    z_init_mode_list = metadata['z_init_mode']
    limit_list = metadata['limit']
    assert len(z_init_mode_list) == len(limit_list)
    del (metadata['z_init_mode'])
    del (metadata['limit'])

    forwards = forward_models[args.model]

    data_split = Path(args.img_dir).name
    for img_name in tqdm(sorted(os.listdir(args.img_dir)),
                         desc='Images',
                         leave=True,
                         disable=args.disable_tqdm):
        # Reset generator weights between each image
        gen, img_size = reset_gen()
        img_shape = (3, img_size, img_size)
        # Load image and get filename without extension
        orig_img = load_target_image(os.path.join(args.img_dir, img_name),
                                     img_size).to(DEVICE)
        img_basename, _ = os.path.splitext(img_name)

        for i, (f, f_args_list) in enumerate(
                tqdm(forwards.items(),
                     desc='Forwards',
                     leave=False,
                     disable=args.disable_tqdm)):
            for f_args in tqdm(f_args_list,
                               desc=f'{f} Args',
                               leave=False,
                               disable=args.disable_tqdm):

                f_args['img_shape'] = img_shape
                forward_model = get_forward_model(f, **f_args)

                for z_init_mode, limit in zip(
                        tqdm(z_init_mode_list, desc='z_init_mode',
                             leave=False), limit_list):
                    metadata['z_init_mode'] = z_init_mode
                    metadata['limit'] = limit

                    # Before doing recovery, check if results already exist
                    # and possibly skip
                    recovered_name = 'recovered.pt'
                    results_folder = get_results_folder(
                        image_name=img_basename,
                        model=args.model,
                        n_cuts=0,  # NOTE - this field is unused for iagan
                        split=data_split,
                        forward_model=forward_model,
                        recovery_params=dict_to_str(metadata),
                        base_dir=BASE_DIR)

                    os.makedirs(results_folder, exist_ok=True)

                    recovered_path = results_folder / recovered_name
                    if os.path.exists(recovered_path) and not args.overwrite:
                        print(f'{recovered_path} already exists, skipping...')
                        continue

                    if args.run_name is not None:
                        current_run_name = (
                            f'{img_basename}'
                            f'.{forward_model}'
                            f'.z_steps1={metadata["z_steps1"]}'
                            f'.z_steps2={metadata["z_steps2"]}'
                            f'.z_lr1={metadata["z_lr1"]}'
                            f'.z_lr2={metadata["z_lr2"]}'
                            f'.model_lr={metadata["model_lr"]}'
                            f'.z_init={z_init_mode}.limit={limit}'
                            f'.{args.run_name}')
                    else:
                        current_run_name = None

                    recovered_img, distorted_img, _ = iagan_recover(
                        orig_img, gen, forward_model, metadata['optimizer'],
                        z_init_mode, limit, metadata['z_lr1'],
                        metadata['z_lr2'], metadata['model_lr'],
                        metadata['z_steps1'], metadata['z_steps2'],
                        metadata['restarts'], args.run_dir, current_run_name,
                        args.disable_tqdm)

                    # Make images folder
                    img_folder = get_images_folder(split=data_split,
                                                   image_name=img_basename,
                                                   img_size=img_size,
                                                   base_dir=BASE_DIR)
                    os.makedirs(img_folder, exist_ok=True)

                    # Save original image if needed
                    original_img_path = img_folder / 'original.pt'
                    if not os.path.exists(original_img_path):
                        torch.save(orig_img, original_img_path)

                    # Save distorted image if needed
                    if forward_model.viewable:
                        distorted_img_path = img_folder / f'{forward_model}.pt'
                        if not os.path.exists(distorted_img_path):
                            torch.save(distorted_img, distorted_img_path)

                    # Save recovered image and metadata
                    torch.save(recovered_img, recovered_path)
                    pickle.dump(metadata,
                                open(results_folder / 'metadata.pkl', 'wb'))
                    p = psnr(recovered_img, orig_img)
                    pickle.dump(p, open(results_folder / 'psnr.pkl', 'wb'))


def mgan_images(args):
    if args.set_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(BASE_DIR, exist_ok=True)

    if args.model in ['mgan_began_cs']:
        gen = Generator128(64)
        gen = load_trained_net(
            gen, ('./checkpoints/celeba_began.withskips.bs32.cosine.min=0.25'
                  '.n_cuts=0/gen_ckpt.49.pt'))
        gen = gen.eval().to(DEVICE)
        img_size = 128
    elif args.model in ['mgan_vanilla_vae_cs']:
        gen = VAE()
        t = torch.load('./vae_checkpoints/vae_bs=128_beta=1.0/epoch_19.pt')
        gen.load_state_dict(t)
        gen = gen.eval().to(DEVICE)
        gen = gen.decoder
        img_size = 128
    elif args.model in ['mgan_dcgan_cs']:
        gen = dcgan_generator()
        t = torch.load(('./dcgan_checkpoints/netG.epoch_24.n_cuts_0.bs_64'
                        '.b1_0.5.lr_0.0002.pt'))
        gen.load_state_dict(t)
        gen = gen.eval().to(DEVICE)
        img_size = 64
    else:
        raise NotImplementedError()

    img_shape = (3, img_size, img_size)
    metadata = recovery_settings[args.model]
    n_cuts_list = metadata['n_cuts_list']
    del (metadata['n_cuts_list'])

    z_init_mode_list = metadata['z_init_mode']
    limit_list = metadata['limit']
    assert len(z_init_mode_list) == len(limit_list)
    del (metadata['z_init_mode'])
    del (metadata['limit'])

    forwards = forward_models[args.model]

    data_split = Path(args.img_dir).name
    for img_name in tqdm(sorted(os.listdir(args.img_dir)),
                         desc='Images',
                         leave=True,
                         disable=args.disable_tqdm):
        # Load image and get filename without extension
        orig_img = load_target_image(os.path.join(args.img_dir, img_name),
                                     img_size).to(DEVICE)
        img_basename, _ = os.path.splitext(img_name)

        for n_cuts in tqdm(n_cuts_list,
                           desc='N_cuts',
                           leave=False,
                           disable=args.disable_tqdm):
            metadata['n_cuts'] = n_cuts
            for i, (f, f_args_list) in enumerate(
                    tqdm(forwards.items(),
                         desc='Forwards',
                         leave=False,
                         disable=args.disable_tqdm)):
                for f_args in tqdm(f_args_list,
                                   desc=f'{f} Args',
                                   leave=False,
                                   disable=args.disable_tqdm):

                    f_args['img_shape'] = img_shape
                    forward_model = get_forward_model(f, **f_args)

                    for z_init_mode, limit in zip(
                            tqdm(z_init_mode_list,
                                 desc='z_init_mode',
                                 leave=False), limit_list):
                        metadata['z_init_mode'] = z_init_mode
                        metadata['limit'] = limit

                        # Before doing recovery, check if results already exist
                        # and possibly skip
                        recovered_name = 'recovered.pt'
                        results_folder = get_results_folder(
                            image_name=img_basename,
                            model=args.model,
                            n_cuts=n_cuts,
                            split=data_split,
                            forward_model=forward_model,
                            recovery_params=dict_to_str(metadata),
                            base_dir=BASE_DIR)

                        os.makedirs(results_folder, exist_ok=True)

                        recovered_path = results_folder / recovered_name
                        if os.path.exists(
                                recovered_path) and not args.overwrite:
                            print(
                                f'{recovered_path} already exists, skipping...'
                            )
                            continue

                        if args.run_name is not None:
                            current_run_name = (
                                f'{img_basename}.{forward_model}'
                                f'.{dict_to_str(metadata)}'
                                f'.{args.run_name}')
                        else:
                            current_run_name = None

                        recovered_img, distorted_img, _ = mgan_recover(
                            orig_img, gen, n_cuts, forward_model,
                            metadata['optimizer'], z_init_mode, limit,
                            metadata['z_lr'], metadata['n_steps'],
                            metadata['z_number'], metadata['restarts'],
                            args.run_dir, current_run_name, args.disable_tqdm)

                        # Make images folder
                        img_folder = get_images_folder(split=data_split,
                                                       image_name=img_basename,
                                                       img_size=img_size,
                                                       base_dir=BASE_DIR)
                        os.makedirs(img_folder, exist_ok=True)

                        # Save original image if needed
                        original_img_path = img_folder / 'original.pt'
                        if not os.path.exists(original_img_path):
                            torch.save(orig_img, original_img_path)

                        # Save distorted image if needed
                        if forward_model.viewable:
                            distorted_img_path = img_folder / f'{forward_model}.pt'
                            if not os.path.exists(distorted_img_path):
                                torch.save(distorted_img, distorted_img_path)

                        # Save recovered image and metadata
                        torch.save(recovered_img, recovered_path)
                        pickle.dump(
                            metadata,
                            open(results_folder / 'metadata.pkl', 'wb'))
                        p = psnr(recovered_img, orig_img)
                        pickle.dump(p, open(results_folder / 'psnr.pkl', 'wb'))


def deep_decoder_images(args):
    if args.set_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(BASE_DIR, exist_ok=True)

    metadata = recovery_settings[args.model]
    forwards = forward_models[args.model]

    data_split = Path(args.img_dir).name
    for img_name in tqdm(sorted(os.listdir(args.img_dir)),
                         desc='Images',
                         leave=True,
                         disable=args.disable_tqdm):
        orig_img = load_target_image(os.path.join(args.img_dir, img_name),
                                     metadata['img_size']).to(DEVICE)
        img_basename, _ = os.path.splitext(img_name)

        for f, f_args_list in tqdm(forwards.items(),
                                   desc='Forwards',
                                   leave=False,
                                   disable=args.disable_tqdm):
            for f_args in tqdm(f_args_list,
                               desc=f'{f} Args',
                               leave=False,
                               disable=args.disable_tqdm):
                f_args['img_shape'] = (3, metadata['img_size'],
                                       metadata['img_size'])
                forward_model = get_forward_model(f, **f_args)

                recovered_name = 'recovered.pt'
                results_folder = get_results_folder(
                    image_name=img_basename,
                    model=args.model,
                    n_cuts=0,  # NOTE - this field is unused for iagan
                    split=data_split,
                    forward_model=forward_model,
                    recovery_params=dict_to_str(metadata),
                    base_dir=BASE_DIR)

                os.makedirs(results_folder, exist_ok=True)

                recovered_path = results_folder / recovered_name
                if os.path.exists(recovered_path) and not args.overwrite:
                    print(f'{recovered_path} already exists, skipping...')
                    continue

                if args.run_name is not None:
                    current_run_name = (f'{img_basename}' +
                                        f'.{forward_model}' +
                                        dict_to_str(metadata) +
                                        f'.{args.run_name}')
                else:
                    current_run_name = None

                recovered_img, distorted_img, _ = deep_decoder_recover(
                    orig_img,
                    forward_model=forward_model,
                    optimizer=metadata['optimizer'],
                    num_filters=metadata['num_filters'],
                    depth=metadata['depth'],
                    lr=metadata['lr'],
                    img_size=metadata['img_size'],
                    steps=metadata['steps'],
                    restarts=metadata['restarts'],
                    run_dir=args.run_dir,
                    run_name=current_run_name,
                    disable_tqdm=args.disable_tqdm)

                # Make images folder
                img_folder = get_images_folder(split=data_split,
                                               image_name=img_basename,
                                               img_size=metadata['img_size'],
                                               base_dir=BASE_DIR)
                os.makedirs(img_folder, exist_ok=True)

                # Save original image if needed
                original_img_path = img_folder / 'original.pt'
                if not os.path.exists(original_img_path):
                    torch.save(orig_img, original_img_path)

                # Save distorted image if needed
                if forward_model.viewable:
                    distorted_img_path = img_folder / f'{forward_model}.pt'
                    if not os.path.exists(distorted_img_path):
                        torch.save(distorted_img, distorted_img_path)

                # Save recovered image and metadata
                torch.save(recovered_img, recovered_path)
                pickle.dump(metadata,
                            open(results_folder / 'metadata.pkl', 'wb'))
                p = psnr(recovered_img, orig_img)
                pickle.dump(p, open(results_folder / 'psnr.pkl', 'wb'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir', required=True, help='')
    p.add_argument('--model', required=True)
    p.add_argument('--run_dir', default=None)
    p.add_argument('--run_name', default=None)
    p.add_argument('--disable_tqdm', action='store_true')
    p.add_argument('--overwrite',
                   action='store_true',
                   help='Set flag to overwrite pre-existing files')
    p.add_argument('--set_seed', action='store_true')
    args = p.parse_args()

    if args.model in [
            'began_cs',
            'began_cs_n_cuts',
            'began_cs_other_init',
            'began_inv',
            'began_noop',
            'began_opt_error_fake_imgs',
            'began_untrained_cs',
            'began_restarts_cs',
            'beta_vae_cs',
            'beta_vae_inv',
            'beta_vae_noop',
            'biggan_inv',
            'biggan_noop',
            'dcgan_cs',
            'dcgan_cs_n_cuts',
            'dcgan_restarts_cs',
            'dcgan_inv',
            'dcgan_noop',
            'dcgan_untrained_cs',
            'vanilla_vae_cs',
            'vanilla_vae_cs_n_cuts',
            'vanilla_vae_inv',
            'vanilla_vae_noop',
            'vanilla_vae_untrained_cs',
    ]:
        gan_images(args)
    elif args.model in [
            'lasso-dct-64',
            'lasso-dct-128',
    ]:
        lasso_cs_images(args)
    elif args.model in [
            'iagan_dcgan_cs',
            'iagan_began_cs',
            'iagan_vanilla_vae_cs',
    ]:
        iagan_images(args)
    elif args.model in [
            'mgan_began_cs',
            'mgan_vanilla_vae_cs',
            'mgan_dcgan_cs',
    ]:
        mgan_images(args)
    elif args.model in [
            'deep_decoder_64_cs',
            'deep_decoder_128_cs',
    ]:
        deep_decoder_images(args)
    else:
        raise NotImplementedError()
