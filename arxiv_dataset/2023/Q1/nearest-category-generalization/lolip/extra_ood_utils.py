from os.path import join

def get_ood_data_paths(dataset_name, base_dir):
    if dataset_name in ['cifar10', 'cifar100']:
        if dataset_name == 'cifar10':
            folder_name = "CIFAR-10-C"
        elif dataset_name == 'cifar100':
            folder_name = "CIFAR-100-C"

        image_files = [
            join(base_dir, folder_name, 'gaussian_noise.npy'),
            join(base_dir, folder_name, 'impulse_noise.npy'),
            join(base_dir, folder_name, 'shot_noise.npy'),
            join(base_dir, folder_name, 'defocus_blur.npy'),
            join(base_dir, folder_name, 'motion_blur.npy'),
            join(base_dir, folder_name, 'zoom_blur.npy'),
            join(base_dir, folder_name, 'glass_blur.npy'),
            join(base_dir, folder_name, 'snow.npy'),
            join(base_dir, folder_name, 'fog.npy'),
            join(base_dir, folder_name, 'contrast.npy'),
            join(base_dir, folder_name, 'pixelate.npy'),
            join(base_dir, folder_name, 'brightness.npy'),
            join(base_dir, folder_name, 'elastic_transform.npy'),
            join(base_dir, folder_name, 'gaussian_blur.npy'),
            join(base_dir, folder_name, 'jpeg_compression.npy'),
            join(base_dir, folder_name, 'saturate.npy'),
            join(base_dir, folder_name, 'spatter.npy'),
            join(base_dir, folder_name, 'speckle_noise.npy'),
        ]
        ood_names = ['gaussian', 'impulse', 'shot', 'defocus', 'motion', 'zoom', 'glass', 'snow', 'fog', 'contrast', 'pixelate',
                    'brightness', 'elastic_transform', 'gaussian_blur', 'jpeg_compression', 'saturate', 'spatter', 'speckle_noise']

    elif dataset_name in ['imgnet']:
        folder_name = "ImageNet-c"

        image_files = []
        ood_names = []
        for i in range(1, 6):
            image_files += [
                join(base_dir, folder_name, f'noise/gaussian_noise/{i}/'),
                join(base_dir, folder_name, f'noise/impulse_noise/{i}/'),
                join(base_dir, folder_name, f'noise/shot_noise/{i}/'),
                join(base_dir, folder_name, f'blur/defocus_blur/{i}/'),
                join(base_dir, folder_name, f'blur/motion_blur/{i}/'),
                join(base_dir, folder_name, f'blur/zoom_blur/{i}/'),
                join(base_dir, folder_name, f'blur/glass_blur/{i}/'),

                join(base_dir, folder_name, f'weather/snow/{i}/'),
                join(base_dir, folder_name, f'weather/fog/{i}/'),
                join(base_dir, folder_name, f'weather/frost/{i}/'),
                join(base_dir, folder_name, f'weather/brightness/{i}/'),
                join(base_dir, folder_name, f'digital/contrast/{i}/'),
                join(base_dir, folder_name, f'digital/pixelate/{i}/'),
                join(base_dir, folder_name, f'digital/jpeg_compression/{i}/'),
                join(base_dir, folder_name, f'digital/elastic_transform/{i}/'),
            ]
            ood_names += [f'gaussian_{i}', f'impulse_{i}', f'shot_{i}',
                          f'defocus_{i}', f'motion_{i}', f'zoom_{i}', f'glass_{i}',
                          f'snow_{i}', f'fog_{i}', f'frost_{i}', f'brightness_{i}',
                          f'contrast_{i}', f'pixelate_{i}', f'jpeg_{i}', f'elastic_{i}']

    else:
        raise ValueError(f"dataset_name {dataset_name} not supported")

    return image_files, ood_names
