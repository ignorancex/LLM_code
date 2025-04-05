import os
import glob
import argparse
import random
import fnmatch

tot = 13200
category_male_male = 4400
category_female_female = 4400
category_male_female = 4400
category_random = 0


if __name__ == '__main__':

    random.seed(1234) # 保证每次运行该文件生成的数据集是相同的

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='path/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='path/frame_files',
                        help="root for extracted video frames")
    parser.add_argument('--train_csv_name', default='train',
                        help="root for extracted audio files")
    parser.add_argument('--test_csv_name', default='test',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    args = parser.parse_args()

    # The dataset should be classified by person names, with names of people of the same gender placed in a list.
    male = [] # need for specific dataset
    female = [] # need for specific dataset

    audio_files_male = []
    audio_files_female = []
    infos_male = []
    infos_female = []

    for host in os.listdir(args.root_audio):
        if host in male:
            audio_files_male += glob.glob(os.path.join(args.root_audio, host, '*.wav'))
        elif host in female:
            audio_files_female += glob.glob(os.path.join(args.root_audio, host, '*.wav'))

    print('{} male audios found'.format(len(audio_files_male)))
    print('{} female audios found'.format(len(audio_files_female)))

    for audio_path in audio_files_male:
        frame_path = audio_path.replace(args.root_audio, args.root_frame).replace('.wav', '')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) <= 20:
            continue
        infos_male.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} male audio/frames pairs found.'.format(len(infos_male)))

    for audio_path in audio_files_female:
        frame_path = audio_path.replace(args.root_audio, args.root_frame).replace('.wav', '')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) == 0:
            continue
        infos_female.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} female audio/frames pairs found.'.format(len(infos_female)))

    infos_tot = infos_male + infos_female
    print('total {} audio/frames pairs found.'.format(len(infos_tot)))

    infos = []

    for index, value in enumerate(range(category_male_male)):
        index_male1 = random.randint(0, len(infos_male) - 1)
        index_male2 = random.randint(0, len(infos_male) - 1)

        aa = infos_male[index_male1].split('/')[-2]
        bb = infos_male[index_male2].split('/')[-2]

        cc = int(infos_male[index_male1].split(',')[-1])
        dd = int(infos_male[index_male2].split(',')[-1])

        flag = aa == bb or cc <= 20 or dd <= 20

        while flag: # 同一个人 或者 有人的时长太短
            index_male1 = random.randint(0, len(infos_male) - 1)
            index_male2 = random.randint(0, len(infos_male) - 1)

            aa = infos_male[index_male1].split('/')[-2]
            bb = infos_male[index_male2].split('/')[-2]

            cc = int(infos_male[index_male1].split(',')[-1])
            dd = int(infos_male[index_male2].split(',')[-1])

            flag = aa == bb or cc <= 20 or dd <= 20

        tmp = infos_male[index_male1] + '|' + infos_male[index_male2]
        infos.append(tmp)

    print("after select audios from male2male, the length of infos is", len(infos))

    for index, value in enumerate(range(category_female_female)):
        index_female1 = random.randint(0, len(infos_female) - 1)
        index_female2 = random.randint(0, len(infos_female) - 1)

        aa = infos_female[index_female1].split('/')[-2]
        bb = infos_female[index_female2].split('/')[-2]

        cc = int(infos_female[index_female1].split(',')[-1])
        dd = int(infos_female[index_female2].split(',')[-1])

        flag = aa == bb or cc <= 20 or dd <= 20

        while flag:
            index_female1 = random.randint(0, len(infos_female) - 1)
            index_female2 = random.randint(0, len(infos_female) - 1)

            aa = infos_female[index_female1].split('/')[-2]
            bb = infos_female[index_female2].split('/')[-2]

            cc = int(infos_female[index_female1].split(',')[-1])
            dd = int(infos_female[index_female2].split(',')[-1])

            flag = aa == bb or cc <= 20 or dd <= 20

        tmp = infos_female[index_female1] + '|' + infos_female[index_female2]
        infos.append(tmp)

    print("after select audios from female2female, the length of infos is", len(infos))

    for index, value in enumerate(range(category_male_female)):
        index_male_female1 = random.randint(0, len(infos_male) - 1)
        index_male_female2 = random.randint(0, len(infos_female) - 1)

        aa = infos_male[index_male_female1].split('/')[-2]
        bb = infos_female[index_male_female2].split('/')[-2]

        cc = int(infos_male[index_male_female1].split(',')[-1])
        dd = int(infos_female[index_male_female2].split(',')[-1])

        flag = aa == bb or cc <= 20 or dd <= 20

        while flag:
            index_male_female1 = random.randint(0, len(infos_male) - 1)
            index_male_female2 = random.randint(0, len(infos_female) - 1)

            aa = infos_male[index_male_female1].split('/')[-2]
            bb = infos_female[index_male_female2].split('/')[-2]

            cc = int(infos_male[index_male_female1].split(',')[-1])
            dd = int(infos_female[index_male_female2].split(',')[-1])

            flag = aa == bb or cc <= 20 or dd <= 20

        tmp = infos_male[index_male_female1] + '|' + infos_female[index_male_female2]
        infos.append(tmp)

    print("after select audios from male2female, the length of infos is", len(infos))

    for index, value in enumerate(range(category_random)):
        index_random1 = random.randint(0, len(infos_tot) - 1)
        index_random2 = random.randint(0, len(infos_tot) - 1)

        aa = infos_tot[index_random1].split('/')[-2]
        bb = infos_tot[index_random2].split('/')[-2]

        cc = int(infos_tot[index_random1].split(',')[-1])
        dd = int(infos_tot[index_random2].split(',')[-1])

        flag = aa == bb or cc <= 20 or dd <= 20

        while flag:
            index_random1 = random.randint(0, len(infos_tot) - 1)
            index_random2 = random.randint(0, len(infos_tot) - 1)

            aa = infos_tot[index_random1].split('/')[-2]
            bb = infos_tot[index_random2].split('/')[-2]

            cc = int(infos_tot[index_random1].split(',')[-1])
            dd = int(infos_tot[index_random2].split(',')[-1])

            flag = aa == bb or cc <= 20 or dd <= 20

        tmp = infos_tot[index_random1] + '|' + infos_tot[index_random2]
        infos.append(tmp)

    print("after select audios from total, the length of infos is", len(infos))

    # split train/val
    n_train = int(len(infos) * (10. / 11))
    random.shuffle(infos)
    trainset = infos[0:n_train]
    valset = infos[n_train:]
    for name, subset in zip([args.train_csv_name, args.test_csv_name], [trainset, valset]):
        if not os.path.exists(args.path_output):
            os.makedirs(args.path_output)
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
