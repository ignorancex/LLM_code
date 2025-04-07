
import os


header_names = ['Tracker','Detector', 'Dataset name', 'Set id', 'mAP', 'Precision', 'Recall', 'TP', 'FP', 'GT detections', 'FN',
                'HOTA(0)', 'LocA(0)', 'HOTALocA(0)', 'HOTA___5', 'HOTA___10', 'HOTA___15', 'HOTA___20', 'HOTA___25', 'HOTA___30', 'HOTA___35', 'HOTA___40', 'HOTA___45', 'HOTA___50', 'HOTA___55', 'HOTA___60', 'HOTA___65', 'HOTA___70', 'HOTA___75', 'HOTA___80', 'HOTA___85', 'HOTA___90', 'HOTA___95', 'HOTA___AUC', 'DetA___5', 'DetA___10', 'DetA___15', 'DetA___20', 'DetA___25', 'DetA___30', 'DetA___35', 'DetA___40', 'DetA___45', 'DetA___50', 'DetA___55', 'DetA___60', 'DetA___65', 'DetA___70', 'DetA___75', 'DetA___80', 'DetA___85', 'DetA___90', 'DetA___95', 'DetA___AUC', 'AssA___5', 'AssA___10', 'AssA___15', 'AssA___20', 'AssA___25', 'AssA___30', 'AssA___35', 'AssA___40', 'AssA___45', 'AssA___50', 'AssA___55', 'AssA___60', 'AssA___65', 'AssA___70', 'AssA___75', 'AssA___80', 'AssA___85', 'AssA___90', 'AssA___95', 'AssA___AUC', 'DetRe___5', 'DetRe___10', 'DetRe___15', 'DetRe___20', 'DetRe___25', 'DetRe___30', 'DetRe___35', 'DetRe___40', 'DetRe___45', 'DetRe___50', 'DetRe___55', 'DetRe___60', 'DetRe___65', 'DetRe___70', 'DetRe___75', 'DetRe___80', 'DetRe___85', 'DetRe___90', 'DetRe___95', 'DetRe___AUC', 'DetPr___5', 'DetPr___10', 'DetPr___15', 'DetPr___20', 'DetPr___25', 'DetPr___30', 'DetPr___35', 'DetPr___40', 'DetPr___45', 'DetPr___50', 'DetPr___55', 'DetPr___60', 'DetPr___65', 'DetPr___70', 'DetPr___75', 'DetPr___80', 'DetPr___85', 'DetPr___90', 'DetPr___95', 'DetPr___AUC', 'AssRe___5', 'AssRe___10', 'AssRe___15', 'AssRe___20', 'AssRe___25', 'AssRe___30', 'AssRe___35', 'AssRe___40', 'AssRe___45', 'AssRe___50', 'AssRe___55', 'AssRe___60', 'AssRe___65', 'AssRe___70', 'AssRe___75', 'AssRe___80', 'AssRe___85', 'AssRe___90', 'AssRe___95', 'AssRe___AUC', 'AssPr___5', 'AssPr___10', 'AssPr___15', 'AssPr___20', 'AssPr___25', 'AssPr___30', 'AssPr___35', 'AssPr___40', 'AssPr___45', 'AssPr___50', 'AssPr___55', 'AssPr___60', 'AssPr___65', 'AssPr___70', 'AssPr___75', 'AssPr___80', 'AssPr___85', 'AssPr___90', 'AssPr___95', 'AssPr___AUC', 'LocA___5', 'LocA___10', 'LocA___15', 'LocA___20', 'LocA___25', 'LocA___30', 'LocA___35', 'LocA___40', 'LocA___45', 'LocA___50', 'LocA___55', 'LocA___60', 'LocA___65', 'LocA___70', 'LocA___75', 'LocA___80', 'LocA___85', 'LocA___90', 'LocA___95', 'LocA___AUC', 'RHOTA___5', 'RHOTA___10', 'RHOTA___15', 'RHOTA___20', 'RHOTA___25', 'RHOTA___30', 'RHOTA___35', 'RHOTA___40', 'RHOTA___45', 'RHOTA___50', 'RHOTA___55', 'RHOTA___60', 'RHOTA___65', 'RHOTA___70', 'RHOTA___75', 'RHOTA___80', 'RHOTA___85', 'RHOTA___90', 'RHOTA___95', 'RHOTA___AUC', 'HOTA_TP___5', 'HOTA_TP___10', 'HOTA_TP___15', 'HOTA_TP___20', 'HOTA_TP___25', 'HOTA_TP___30', 'HOTA_TP___35', 'HOTA_TP___40', 'HOTA_TP___45', 'HOTA_TP___50', 'HOTA_TP___55', 'HOTA_TP___60', 'HOTA_TP___65', 'HOTA_TP___70', 'HOTA_TP___75', 'HOTA_TP___80', 'HOTA_TP___85', 'HOTA_TP___90', 'HOTA_TP___95', 'HOTA_TP___AUC', 'HOTA_FN___5', 'HOTA_FN___10', 'HOTA_FN___15', 'HOTA_FN___20', 'HOTA_FN___25', 'HOTA_FN___30', 'HOTA_FN___35', 'HOTA_FN___40', 'HOTA_FN___45', 'HOTA_FN___50', 'HOTA_FN___55', 'HOTA_FN___60', 'HOTA_FN___65', 'HOTA_FN___70', 'HOTA_FN___75', 'HOTA_FN___80', 'HOTA_FN___85', 'HOTA_FN___90', 'HOTA_FN___95', 'HOTA_FN___AUC', 'HOTA_FP___5', 'HOTA_FP___10', 'HOTA_FP___15', 'HOTA_FP___20', 'HOTA_FP___25', 'HOTA_FP___30', 'HOTA_FP___35', 'HOTA_FP___40', 'HOTA_FP___45', 'HOTA_FP___50', 'HOTA_FP___55', 'HOTA_FP___60', 'HOTA_FP___65', 'HOTA_FP___70', 'HOTA_FP___75', 'HOTA_FP___80', 'HOTA_FP___85', 'HOTA_FP___90', 'HOTA_FP___95', 'HOTA_FP___AUC', 'MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA', 'CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum', 'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag', 'CLR_Frames', 'IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP', 'STDA', 'ATA', 'FDA', 'SFDA', 'VACE_IDs', 'VACE_GT_IDs', 'num_non_empty_timesteps', 'Dets', 'GT_Dets', 'IDs', 'GT_IDs',
                'E', 'Eintra', 'Qd', 'Qt', 'Nd', 'Nt', 'Id', 'It', 'Einter', 'Id', 'It', 'Y', 'C', 'IDSW_score']


def get_detection_metrics(name):

    file_name = os.path.join('outputs/evaluation/detection', name, 'metrics.txt')

    with open(file_name, 'r') as file:

        data = file.read()

        data = data.split('\n')[:-1]

        header = data[0].split(',')
        header = [header[0]] + [header[1]] + header[3:]

        # data = [line.split(',') for line in data[1:]]

        info = {}

        for line in data[1:]:

            aux = line.split(',')

            info[aux[2]] = [aux[0]] + [aux[1]] + aux[3:]

        # print(header)
        # print(info)


    return header, info



def get_tracking_metrics(path, file_name):

    file_name = os.path.join(path, file_name)

    with open(file_name, 'r') as file:

        data = file.read()

        data = data.split('\n')[:-1]

        header = data[0].split(',')[1:]

        # data = [line.split(',') for line in data[1:]]

        info = {}

        for line in data[1:]:

            aux = line.split(',')

            info[aux[0]] = aux[1:]

        # print(header)
        # print(info)

    return header, info


def get_own_metrics(path, file_name):

    file_name = os.path.join(path, file_name)

    info = {}

    with open(file_name, 'r') as file:

        data = file.read()

        data = data.split('\n')[:-1]

        header = data[0].split(',')[1:]

        # print(header)
        for line in data[1:]:

            aux = line.split(',')

            detector = aux[0]
            tracker  = aux[1]
            seq_name = aux[2]

            if detector not in info: info[detector] = {}
            if tracker not in info[detector]: info[detector][tracker] = {}
            if seq_name not in info[detector][tracker]: info[detector][tracker][seq_name] = aux[3:]

            # data[detector]
            # data[detector][tracker]
            # data[detector][tracker][seq_name]

            # print(detector, tracker, seq_name, aux[3:])

    return header, info






def print_list(file, lista):

    for i, elem in enumerate(lista):

        if i == 0: file.write(str(elem))
        else:      file.write(',' + str(elem))

    file.write('\n')



if __name__ == '__main__':


    path = 'outputs/evaluation/tracking'

    file = open('outputs/evaluation/all_metrics.csv', 'w')

    print_list(file, header_names)


    datasets = os.listdir(path)
    _, own_metrics = get_own_metrics("outputs/evaluation", "own.csv")


    for set_name in datasets:

        path_set = os.path.join(path, set_name)
        trackers = os.listdir(path_set)

        for tracker in trackers:

            path_track = os.path.join(path_set, tracker)
            detectors = os.listdir(path_track)
            detectors.sort()

            for detector in detectors:

                path_metrics = os.path.join(path_track, detector)

                header_t, info_t = get_tracking_metrics(path_metrics, 'pedestrian_detailed.csv')
                header_d, info_d = get_detection_metrics(detector)


                for subset_name in info_t.keys():

                    if subset_name == 'COMBINED':

                        # seq_metrics = [tracker, detector, set_name, subset_name] + ['-1', '-1', '-1', '-1', '-1', '-1', '-1'] + info_t[subset_name][:] + ['-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1']
                        continue


                    elif not detector in own_metrics:

                        seq_metrics = [tracker, detector, set_name, subset_name] + info_d[subset_name][2:] + info_t[subset_name][:] + ['-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1']


                    elif not tracker in own_metrics[detector]:

                        seq_metrics = [tracker, detector, set_name, subset_name] + info_d[subset_name][2:] + info_t[subset_name][:] + ['-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1']


                    elif not subset_name in own_metrics[detector][tracker]:

                        seq_metrics = [tracker, detector, set_name, subset_name] + info_d[subset_name][2:] + info_t[subset_name][:] + ['-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1', '-1']

                    else:

                        seq_metrics = [tracker, detector, set_name, subset_name] + info_d[subset_name][2:] + info_t[subset_name][:] + own_metrics[detector][tracker][subset_name]


                    print_list(file, seq_metrics)
