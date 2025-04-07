
import torchvision
from torch.utils.data import DataLoader

from Detector import Detector


class mask_rcnn(Detector):

    def __init__(self, batch_size):

        super().__init__('mask_rcnn', batch_size)


        # Load model (pretrained)
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        # Classes:  https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
        self.label_permited = [1, 2, 3, 4, 6, 7, 8]



    def eval_set(self, dataset, loader, device, verbose=0):
        '''Run evaluation over loaded data.
        '''

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        self.model.to(device)


        for i_batch, (images, y) in enumerate(dataloader):

            images = images.to(device)
            output = self.model(images)

            if verbose: print('Frame:', i_batch * self.batch_size)

            for i, frame in enumerate(output):

                i_frame = (i_batch * self.batch_size) + i + 1

                loader.update(i_frame, frame['boxes'].cpu().detach().numpy(), frame['scores'].cpu().detach().numpy(), frame['labels'].cpu(), label_permited=self.label_permited)
