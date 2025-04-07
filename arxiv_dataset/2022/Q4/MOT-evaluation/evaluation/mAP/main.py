
import os
import numpy as np



classes_dict = {1 : 'Pedestrian',
                2 : 'Person_on_vehicle',
                3 : 'Car',
                4 : 'Bicycle',
                5 : 'Motorbike',
                6 : 'Non_motorized_vehicle',
                7 : 'Static_person',
                8 : 'Distractor',
                9 : 'Occluder',
                10 : 'Occluder_on_the_ground',
                11 : 'Occluder_full',
                12 : 'Reflection',
                13 : 'Crowd'}



def readFile(path):
    '''Read a GT or predictions file in the indicated
    path. Return a dict were the keys are frames.
    '''

    file = np.loadtxt(path, delimiter=',')

    unique = np.unique(file[:, 0])


    frame = {}

    for u in unique:

        a = np.where(file[:, 0] == u)

        frame[u] = file[a][:, 1:]


    return frame



def int2class(frame, classes_dict, permited=[1, 2, 3, 4, 5, 6, 7], det_pub=False):

    out_array = []

    # permited=[1, 2, 7]


    # If public detections
    if (frame.shape[-1] == 6) or det_pub:
        # print('public')

        for i, _ in enumerate(frame[:, 0]):

            aux_frame = np.full((6), "", dtype=object)

            # Label string
            aux_frame[0] = 'object'

            # Confidence
            aux_frame[1] = frame[i, 5]

            # Bounding box
            aux_frame[2:6] = frame[i, 1:5]


            # width, height -> x2, y2
            aux_frame[4] = aux_frame[2] + aux_frame[4]
            aux_frame[5] = aux_frame[3] + aux_frame[5]


            out_array.append( aux_frame )


    else:

        for i, label in enumerate(frame[:, 6]):

            if not label in permited: continue

            # out_list.append( classes_dict[label] )

            aux_frame = np.full((6), "", dtype=object)

            # Label string
            # aux_frame[0] = classes_dict[label]
            aux_frame[0] = 'object'

            # Confidence
            aux_frame[1] = frame[i, 5]

            # Bounding box
            aux_frame[2:6] = frame[i, 1:5]


            # width, height -> x2, y2
            aux_frame[4] = aux_frame[2] + aux_frame[4]
            aux_frame[5] = aux_frame[3] + aux_frame[5]


            out_array.append( aux_frame )


    return np.asarray(out_array)



def processFrame(frame, filename, gt=False, pub=False):

    if frame is None:
        f = open(filename, 'w')
        f.close()

        return

    # <class_name> <left> <top> <right> <bottom>
    final_frame = np.zeros( (frame.shape[0], 6))

    final_frame = int2class(frame, classes_dict, det_pub=pub)


    # print(final_frame.shape)


    if gt: np.savetxt(filename, final_frame[:, [0, 2, 3, 4, 5]], delimiter=' ', fmt=['%s', '%.0f', '%.0f', '%.0f', '%.0f'])
    else:  np.savetxt(filename, final_frame, delimiter=' ', fmt=['%s', '%.2f', '%.0f', '%.0f', '%.0f', '%.0f'])


def cleanFile(path):

    os.system("rm -r %s/*" %(path))



def processSequence(gt_path, det_path, img_path, gt_auxiliar='evaluation/mAP/auxiliar/GT', det_auxiliar='evaluation/mAP/auxiliar/DET', pub=False):

    # Clean axiliar folder
    cleanFile(gt_auxiliar)
    cleanFile(det_auxiliar)


    # Read files with data
    gt_file  = readFile(gt_path)
    det_file = readFile(det_path)


    # For each frame (create aux file)
    for key, gt_frame in gt_file.items():

        gt_filename  = os.path.join(gt_auxiliar, '%.6d.txt'%key)
        det_filename = os.path.join(det_auxiliar, '%.6d.txt'%key)


        processFrame(gt_frame, gt_filename, gt=True)

        try:
            processFrame(det_file[key], det_filename, pub=pub)

        except:
            processFrame(None, det_filename)


    # https://github.com/Cartucho/mAP
    os.system("python evaluation/mAP/mAP.py --img_path " + img_path)



if __name__ == '__main__':

    list_detectors = os.listdir('outputs/detections')
    list_detectors = ['gt', 'public', 'faster_rcnn', 'faster_rcnn-mod-1', 'faster_rcnn-mod-2', 'faster_rcnn-mod-3', 'faster_rcnn-mod-4', 'faster_rcnn-fine-tune', 'keypoint_rcnn', 'mask_rcnn', 'retinanet', 'yolo3', 'yolo4']

    list_sets = ['MOT17', 'MOT20', 'VisDrone2019-MOT-val']
    list_not_detectors = []

    output_file = os.path.join('outputs/evaluation/detection', 'summary.txt')


    # Check folders
    if not os.path.exists('outputs/evaluation'):
        os.makedirs('outputs/evaluation')

    if not os.path.exists('outputs/evaluation/detection'):
        os.makedirs('outputs/evaluation/detection')



    verbose = True

    with open(output_file, "w") as file:

        if verbose: print('File open')

        # file.write('| Detector | mAP | Precision | Recall | TP | FP | GT detections |\n')
        # file.write('|----------|-----|-----------|--------|----|----|---------------|\n')
        file.write('Detector,mAP,Precision,Recall,TP,FP,FN,GT detections\n')


        for detector in list_detectors:

            if detector in list_not_detectors: continue
            if verbose: print('DETECTOR:  ', detector)


            pub = True if detector == 'public' else False

            # list_sets = os.listdir('dataset')

            average = {'mAP':[],
                       'precision':[],
                       'recall':[],
                       'TP':[],
                       'FP':[],
                       'GT_detections':[]}

            output_file_detector = os.path.join('outputs/evaluation/detection', detector)

            # Check folders
            if not os.path.exists(output_file_detector):
                os.makedirs(output_file_detector)

            output_file_detector = os.path.join(output_file_detector, 'metrics.txt')



            file_detec = open(output_file_detector, "w")

            # file_detec.write('| Detector | Subset name | mAP | Precision | Recall | TP | FP | GT detections |\n')
            # file_detec.write('|----------|-------------|-----|-----------|--------|----|----|---------------|\n')
            file_detec.write('Detector,Set,Subset,mAP,Precision,Recall,TP,FP,GT detections,FN\n')


            for set_name in list_sets:

                if not os.path.exists( os.path.join('outputs/detections', detector, set_name)):
                    print('! WARNING :', detector, set_name, '- do not exist.')
                    continue

                list_subsets = os.listdir( os.path.join('dataset', set_name) )
                list_subsets.sort()

                for subset in list_subsets:

                    if verbose: print('   >', set_name, subset)


                    gt_path  = os.path.join('dataset/', set_name, subset, 'gt/gt.txt')
                    img_path = os.path.join('dataset/', set_name, subset, 'img1')
                    det_path = os.path.join('outputs/detections', detector, set_name, subset, 'det/det.txt')




                    processSequence(gt_path, det_path, img_path, pub=pub)


                    with open('evaluation/mAP/auxiliar.txt', 'r') as f:

                        data = f.read().split('\n')

                        names = data[0].split(',')
                        values = data[1].split(',')

                        # print('names:', names, len(names))
                        # print('values:', values, len(values))


                    #     if verbose: print(mAP)



                    text_w = detector + ',' + set_name + ',' + subset + ','

                    for name, value in zip(names, values):

                        average[name].append( float(value) )
                        text_w += value + ','


                    # FN
                    # Predicted detections
                    FN = int(average['GT_detections'][-1] - average['TP'][-1])
                    text_w += str(FN) + ','

                    file_detec.write(text_w[:-1] + '\n')

                    # continue


            mAP           = sum(average['mAP']) / len(average['mAP'])
            precision     = sum(average['precision']) / len(average['precision'])
            recall        = sum(average['recall']) / len(average['recall'])
            TP            = sum(average['TP'])
            FP            = sum(average['FP'])
            GT_detections = sum(average['GT_detections'])

            FN            = GT_detections - TP
            # Pred_detect   =

            # file.write('| ' + detector + ' | AVERAGE | ' + str(sum(average) / len(average)) + ' | \n')
            file.write(detector + ',%f,%f,%f,%d,%d,%d,%d\n' % (mAP, precision, recall, TP, FP, FN, GT_detections))







    # detector = 'yolo3'
    # det_path = os.path.join('outputs/detections', detector, 'MOT17/MOT17-02/det/det.txt')

    # print('DETECTOR:  ', detector)
    # processSequence(gt_path, det_path)
