
import os
import cv2
import argparse
import numpy as np

from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt

# GT, det, track
COLORS = ['green', 'blue', 'red']


def parseInput():
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Visualize tracks')

    # Select an able detector from list.
    parser.add_argument('-d', "--detector", help="Name of detector used.", required=True)
    parser.add_argument('-t', "--tracker", help="Name of tracker used.", required=True)
    parser.add_argument("--set", help="Name of set of data.", required=True)

    # Important arguments, but optional.
    parser.add_argument("--path", help="Path to tracking folder.", default='outputs/tracks')
    parser.add_argument('-v', "--verbose", help="Print some debug info.")

    # Other optional arguments


    return parser.parse_args()


class Visualize:


    def __init__(self, detector, tracker, set_data, data, verbose=0, data_path='../dataset/', out_path='../outputs/', aux=False, save=True):

        data, self.extension = os.path.splitext(data)

        self.gt_path = os.path.join(data_path, set_data, data, 'gt')
        self.gt_frame    = self.readFile(self.gt_path + '/gt.txt')

        self.imgs_path  = os.path.join(data_path, set_data, data, 'img1')
        self.out_data  = os.path.join(out_path, 'videos', data + '.mp4')

        if detector:
            self.detec_path = os.path.join(out_path, 'detections', detector, set_data, data, 'det')
            self.detec_frame = self.readFile(self.detec_path + '/det.txt')
        else:
            self.detec_path = None


        if tracker:
            self.track_path = os.path.join(out_path, 'tracks', tracker, detector, set_data, data)
            self.track_frame = self.readFile(self.track_path + '.txt')
        else:
            self.track_path = None

        if save:
            if not os.path.exists('outputs/videos'):  os.mkdir('outputs/videos')


        self.listColors()

        img  = self.readFrame(1)
        size = (img.shape[1], img.shape[0])

        self.videoWritter(size)


    @staticmethod
    def subsets(detector, tracker, set_data, path='outputs/tracks'):

        path = os.path.join(path, tracker, detector, set_data)

        return os.listdir( path )


    @staticmethod
    def fromXtoWH(x1, y1, x2, y2):

        return x1, y1, x2-x1, y2-y1


    @staticmethod
    def fromWHtoX(x, y, w, h):

        return x, y, x+w, y+h


    @staticmethod
    def centerWH(x, y, w, h):

        return int(x+(w/2)), int(y+(h/2))



    def readFile(self, path):

        tracks = np.loadtxt(path, delimiter=',')

        unique = np.unique(tracks[:, 0])


        frame = {}

        for u in unique:

            a = np.where(tracks[:, 0] == u)

            frame[u] = tracks[a][:, 1:]


        return frame


    def readFrame(self, frame, k=5):
        '''
        '''

        if k == 5:   file_name = os.path.join(self.imgs_path, '%.6d.jpg'%frame)
        elif k == 6: file_name = os.path.join(self.imgs_path, '%.7d.jpg'%frame)

        # print('---->',file_name, '(%d)' % frame)

        return cv2.imread(file_name)


    def listColors(self):

        list_rgb = {}

        for i, c in enumerate(COLORS):


            rgb = to_rgb(c)
            rgb = tuple( [rgb[0] * 255, rgb[1] * 255, rgb[2] * 255])
            # print(i, c, rgb)

            list_rgb[i] = rgb

            # print(rgb)


        self.list_rgb = list_rgb
        # print(list_rgb)
        self.max_colors = len(COLORS)








    def draw_bbox(self, img, frame, thickness=2):


        try:
            img = self.draw_b(img, self.gt_frame[frame]   , self.list_rgb[0], thickness)

            if self.detec_path:
                img = self.draw_b(img, self.detec_frame[frame], self.list_rgb[1], thickness)

            if self.track_path:
                img = self.draw_b(img, self.track_frame[frame], self.list_rgb[2], thickness)

        except:

            print('No detections in frame %d.' % frame)


        return img


    def draw_b(self, img, frame_b, color, thickness):

        for bbox in frame_b:

            if bbox[6] in [9, 10, 11]: continue

            x1, y1, x2, y2 = Visualize.fromWHtoX(int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]))

            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            # img = Visualize.write_text(img, 'ID: %d' % (bbox[0]), color, x1, y1)

        return img




    @staticmethod
    def write_text(img, text, color, x1, y1):

        img = cv2.rectangle(img, (x1, y1), (x1 + 60, y1 + 14), color, -1)
        img = cv2.putText(img, text, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        return img


    def videoWritter(self, size):

        self.writter = cv2.VideoWriter(self.out_data, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, size)
        # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    def writeVideo(self, img):

        self.writter.write(img)

    def saveVideo(self):

        self.writter.release()





def saveSetData(detector, tracker, set_data, data, verbose=0):

    visual_s = Visualize(detector, tracker, set_data, data, verbose=verbose)


    frame = 1

    while True:
    # while frame < 30:

        img = visual_s.readFrame(frame)

        if img is None: break

        visual_s.draw_bbox(img, frame)
        visual_s.writeVideo(img)

        frame += 1


    visual_s.saveVideo()



def plotFrame(detector, tracker, set_data, data, frame_number, verbose=0):

    visual_s = Visualize(detector, tracker, set_data, data, verbose=verbose, save=False)

    img = visual_s.readFrame(frame_number)

    if img is None: return None

    img = visual_s.draw_bbox(img, frame_number)

    return img


def plotImg(img, figsize=(20, 10)):

    fig = plt.figure(figsize=figsize)

    plt.axis('off')

    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(RGB_img)
    plt.show()



if __name__ == '__main__':


    args = parseInput()

    detector = args.detector #'public'
    tracker  = args.tracker  #'sort'
    set_data = args.set      #'MOT20'
    path     = args.path
    verbose  = args.verbose


    # data = Visualize.subsets(detector, tracker, set_data, path)

    if verbose: print('All sets to display:', data)

    # data = ['MOT17-05', 'MOT17-09']
    # data = ['MOT17-09']
    data = ['MOT17-02']

    # For each set
    for d in data:

        if verbose: print(' - Set:', d)

        saveSetData(detector, tracker, set_data, d, verbose=verbose)