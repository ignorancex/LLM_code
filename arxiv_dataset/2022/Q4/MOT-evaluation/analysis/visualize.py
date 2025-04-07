
import os
import cv2
import argparse
import numpy as np

from matplotlib.colors import to_rgb


# python auxiliar/visualize.py --detector . --tracker . --set MOT20 --path dataset/ -v 2


# Need more colors?:
# https://matplotlib.org/2.0.2/examples/color/named_colors.html
COLORS = ['b', 'green', 'r', 'c', 'm', 'y', 'fuchsia', 'lime']




def parseInput():
    '''Parse input of the script.
    '''

    parser = argparse.ArgumentParser(description='Visualize tracks')

    # Select an able detector from list.
    parser.add_argument("--detector", help="Name of detector used.", required=True)
    parser.add_argument("--tracker", help="Name of tracker used.", required=True)
    parser.add_argument("--set", help="Name of set of data.", required=True)

    # Important arguments, but optional.
    parser.add_argument("--path", help="Path to tracking folder.", default='outputs/tracks')
    parser.add_argument('-v', "--verbose", help="Print some debug info.")

    # Other optional arguments


    return parser.parse_args()



class Visualize:



    def __init__(self, detector, tracker, set_data, data, verbose=0, path='outputs/tracks', aux=False):

        data, self.extension = os.path.splitext(data)

        if aux:  self.track_path = os.path.join(path, set_data, data, 'gt')
        else:    self.track_path = os.path.join(path, tracker, detector, set_data, data)

        self.imgs_path  = os.path.join('dataset/', set_data, data, 'img1')
        self.out_data  = os.path.join('outputs/', 'videos', data + '.mp4')
        self.aux = aux

        if not os.path.exists('outputs/videos'):  os.mkdir('outputs/videos')


        self.readTracks()
        self.processTraces()
        self.listColors()

        img  = self.readFrame(1)
        size = (img.shape[1], img.shape[0])

        self.videoWritter(size)





    @staticmethod
    def subsets(detector, tracker, set_data, path='outputs/tracks'):

        path = os.path.join(path, tracker, detector, set_data)

        return os.listdir( path )



    def readTracks(self):

        if self.aux:  tracks = np.loadtxt(self.track_path + '/gt.txt', delimiter=',')
        else:         tracks = np.loadtxt(self.track_path + '.txt', delimiter=',')

        unique = np.unique(tracks[:, 0])


        frame = {}

        for u in unique:

            a = np.where(tracks[:, 0] == u)

            frame[u] = tracks[a][:, 1:]


        self.frames = frame
        


    def processTraces(self):

        ids = {}

        for frame in self.frames.values():

            for obj in frame:

                if obj[6] in [9, 10, 11]: continue

                if not obj[0] in ids:

                    ids[obj[0]] = []


                ids[obj[0]].append( Visualize.centerWH(obj[1], obj[2], obj[3], obj[4]) )


        self.ids = ids



    def listColors(self):

        list_rgb = {}

        for i, c in enumerate(COLORS):

            rgb = to_rgb(c)
            rgb = tuple( [rgb[0] * 255, rgb[1] * 255, rgb[2] * 255])

            list_rgb[i] = rgb

            # print(rgb)


        self.list_rgb = list_rgb
        self.max_colors = len(COLORS)






    def readFrame(self, frame, k=6):
        '''
        '''

        if k == 5:   file_name = os.path.join(self.imgs_path, '%.6d.jpg'%frame)
        elif k == 6: file_name = os.path.join(self.imgs_path, '%.7d.jpg'%frame)

        return cv2.imread(file_name)


            

    @staticmethod
    def fromXtoWH(x1, y1, x2, y2):

        return x1, y1, x2-x1, y2-y1


    @staticmethod
    def fromWHtoX(x, y, w, h):

        return x, y, x+w, y+h


    @staticmethod
    def centerWH(x, y, w, h):

        return int(x+(w/2)), int(y+(h/2))


    def draw_bbox(self, img, frame, thickness=2):

        # color = (0, 0, 255)
        ids = [1]

        for bbox in self.frames[frame]:

            if bbox[6] in [9, 10, 11]: continue
            if not bbox[0] in ids: continue

            x1, y1, x2, y2 = Visualize.fromWHtoX(int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4]))

            path = self.ids[bbox[0]]
            color = self.list_rgb[ bbox[0] % self.max_colors ]

            # print(color, (x1, y1), (x2, y2))

            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            img = Visualize.drawPath(img, path, color, 1)


        return img


    @staticmethod
    def drawPath(img, path, color, thickness=2):

        old = ()

        for idx, coord in enumerate(path):

            if idx == 0:

                old = coord
                continue


            img = cv2.line(img, old, coord, color, thickness=thickness)

            old = coord


        return img


    def videoWritter(self, size):

        self.writter = cv2.VideoWriter(self.out_data, cv2.VideoWriter_fourcc(*'mp4v'), 25.0, size)
        # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    def writeVideo(self, img):

        self.writter.write(img)

    def saveVideo(self):

        self.writter.release()



def saveSetData(detector, tracker, set_data, data, path, verbose=0, aux=False):

    visual_s = Visualize(detector, tracker, set_data, data, path=path, verbose=verbose, aux=aux)


    # PLOT ---------------------------------------------
    frame = 1

    while True:
    # while frame < 30:

        img = visual_s.readFrame(frame)

        if img is None: break

        visual_s.draw_bbox(img, frame)
        visual_s.writeVideo(img)

        frame += 1


    visual_s.saveVideo()





if __name__ == '__main__':


    args = parseInput()

    detector = args.detector #'public'
    tracker  = args.tracker  #'sort'
    set_data = args.set      #'MOT20'
    path     = args.path
    verbose  = args.verbose
    aux      = False
    # aux      = True


    data = Visualize.subsets(detector, tracker, set_data, path)

    if verbose: print('All sets to display:', data)

    # For each set
    for d in data:

        if verbose: print(' - Set:', d)

        saveSetData(detector, tracker, set_data, d, path, verbose=verbose, aux=aux)

    
