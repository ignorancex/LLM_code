import argparse
from random import choice
import torch
from torchvision import models
import numpy as np

from imnet_val import validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valdir', type=str)
    parser.add_argument('--curvedir', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    scale_inv_2_1 = [
        108,  21,  25, 127,  56,  47,  18,  52,  35,  67,  48,  72, 110, 106,
        105,  13,  15,  88,  30, 124,  31,  36,  68
    ]

    scale_inv_3_1 = [
        113, 198,  67,  69, 204, 188, 164, 137, 144, 149, 147, 132, 180,  88,
        126, 141,  48, 135, 187, 250, 158,  29, 178, 161, 163,  13, 153,  14,
        80, 206, 220, 179, 215,  16, 192,   9,  59, 200,  63,   1, 100,  58,
        168, 117, 226,  22
    ]

    mean_inv_acts = []
    for c in scale_inv_3_1:
        curve = torch.tensor(np.load(f'{args.curvedir}/layer3.1.prerelu_out/layer3.1.prerelu_out_unit{c}_all_act_list.npy'))
        curve_mean = torch.mean(torch.clamp(curve, min=0))
        mean_inv_acts.append(curve_mean)

    mean_inv_acts = torch.tensor(mean_inv_acts, dtype=torch.float32).cuda()
    all_mean_inv_acts = [
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': mean_inv_acts}],
        [{'block': 0, 'channels': None},  {'block': 1, 'channels': None}]
    ]

    scale_lesions = [
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
        [{'block': 0, 'channels': None}, {'block': 1, 'channels': scale_inv_3_1}],
        [{'block': 0, 'channels': None},  {'block': 1, 'channels': None}]
    ]

    #   2.1 post-relu mean ablate accs (clamped to zero BEFORE mean is taken)
    # vanilla_acc = 63.75
    # up_accs = [61.394, 59.39, 56.222, 51.686, 44.292]

    #   2.1 ZERO ABLATE
    # vanilla_acc = 
    # up_accs = []

    #   3.1 post-relu mean ablate accs (clamped to zero BEFORE mean is taken)
    vanilla_acc = 63.493
    up_accs = [61.712, 60.194, 57.424, 53.358, 46.704]

    model = models.resnet18(True, lesion=True).cuda()
    model.eval()

    trials = 0
    no_scale_results = []
    scales = np.arange(1.1, 1.6, 0.1)
    while trials < 10:
        print(f"\n\nTRIAL {trials+1}")

        # rand_2_1 = []
        # while len(rand_2_1) < len(scale_inv_2_1):
        #     rand_2_1.append(choice([i for i in range(128) if i not in scale_inv_2_1 and i not in rand_2_1]))

        rand_3_1 = []
        while len(rand_3_1) < len(scale_inv_3_1):
            rand_3_1.append(choice([i for i in range(256) if i not in scale_inv_3_1 and i not in rand_3_1]))

        mean_rand_acts = []
        for c in rand_3_1:
            curve = torch.tensor(np.load(f'{args.curvedir}/layer3.1.prerelu_out/layer3.1.prerelu_out_unit{c}_all_act_list.npy'))
            curve_mean = torch.mean(torch.clamp(curve, min=0))
            mean_rand_acts.append(curve_mean)

        mean_rand_acts = torch.tensor(mean_rand_acts, dtype=torch.float32).cuda()
        all_mean_rand_acts = [
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': mean_rand_acts}],
            [{'block': 0, 'channels': None},  {'block': 1, 'channels': None}]
        ]

        rand_lesions = [
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': None}],
            [{'block': 0, 'channels': None}, {'block': 1, 'channels': rand_3_1}],
            [{'block': 0, 'channels': None},  {'block': 1, 'channels': None}]
        ]

        print('No scale')
        # scale_lesion_acc = validate(model, args.valdir, lesions=scale_lesions, mean_ablates=all_mean_inv_acts)
        
        rand_lesion_acc = validate(model, args.valdir, lesions=rand_lesions, mean_ablates=all_mean_rand_acts)
        print(f"No Scale lesion delta: {vanilla_acc - rand_lesion_acc}\n")

        #   We only want to test scale robustness on random channels which are MORE important for general object recognition.
        if vanilla_acc - rand_lesion_acc < -1:
            print('Rand lesion is less damaging without scale.  Skipping')
            continue

        no_scale_results.append(rand_lesion_acc)
        
        up_results = []
        down_black_results = []
        down_gray_results = []
        for i in range(scales.shape[0]):
            scale = np.round(scales[i], 1)
            print(f"Scale-Up {scale}")

            # scale_acc = validate(model, args.valdir, scale=scale, lesions=scale_lesions, mean_ablates=all_mean_inv_acts)

            rand_lesion_acc = validate(model, args.valdir, scale=scale, lesions=rand_lesions, mean_ablates=all_mean_rand_acts)
            print(f"Scale-Up {scale} lesion delta: {up_accs[i] - rand_lesion_acc}\n")
            up_results.append(rand_lesion_acc)
            
            # print(f"Scale-Down Black {1-frac}")
            # scale_acc = validate(model, args.valdir, scale=1-frac, lesions=scale_lesions, mean_ablates=None, fill=0)

            # rand_lesion_acc = validate(model, args.valdir, scale=1-frac, lesions=rand_lesions, mean_ablates=all_mean_rand_acts, fill=0)
            # print(f"Scale-Down {1-frac} Black lesion delta: {down_black_accs[i-1] - rand_lesion_acc}\n")
            # down_black_results.append(rand_lesion_acc)

            # print(f"Scale-Down Gray {1-frac}")
            # scale_acc = validate(model, args.valdir, scale=1-frac, lesions=scale_lesions, mean_ablates=None, fill=127)

            # rand_lesion_acc = validate(model, args.valdir, scale=1-frac, lesions=rand_lesions, mean_ablates=all_mean_rand_acts, fill=127)
            # print(f"Scale-Down {1-frac} Gray lesion delta: {down_gray_accs[i-1] - rand_lesion_acc}\n")
            # down_gray_results.append(rand_lesion_acc)
        # exit()

        np.save(f'{args.savedir}/layer3.1/up_rand_mean_trial_{trials}.npy', np.array(up_results))
        # np.save(f'{args.savedir}/layer2.1/down_black_rand_zero_trial_{trials}.npy', np.array(down_black_results))
        # np.save(f'{args.savedir}/layer2.1/down_gray_rand_zero_trial_{trials}.npy', np.array(down_gray_results))
        trials += 1

    np.save(f'{args.savedir}/layer3.1/no_scale_rand_mean_all_trials.npy', np.array(no_scale_results))