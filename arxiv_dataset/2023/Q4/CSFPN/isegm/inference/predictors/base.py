import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide


class BasePredictor(object):
    def __init__(self, int_model, seg_model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 eval_mode='normal',
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.gt_mask = None

        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(int_model, tuple):
            self.int_net, self.click_models = int_model
        else:
            self.int_net = int_model

        if isinstance(seg_model, tuple):
            self.seg_net, self.click_models = seg_model
        else:
            self.seg_net = seg_model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

        self.eval_mode = 'cascade'
        self.cascade_mode = 'final'

        self.ignore_intention_count = 0
        self.valid_intention_count = 0

        self.ignore_intention_step = -1
        self.ignore_threshold = 0.3

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_gt_mask(self, gt_mask):
        gt_mask = torch.from_numpy(gt_mask)
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
        self.gt_mask = gt_mask.to(self.device)

    def is_need_intention_bbox(self, bbox, click):
        if bbox is None:
            return True
        x, y = click.coords
        min_row, max_row, min_col, max_col = bbox
        height = max_row - min_row + 2
        width = max_col - min_col + 2
        min_row += height // 8
        max_row -= height // 8
        min_col += width // 8
        max_col -= width // 8
        if min_row < x < max_row and min_col < y < max_col:
            return False
        else:
            return True

    def is_need_intention_prob(self, last_prob, click):
        if last_prob is None:
            return True
        x, y = click.coords
        click_prob = last_prob[0][0][x][y]
        if self.ignore_threshold < click_prob < 1 - self.ignore_threshold:
            return False
        else:
            return True

    def is_need_intention(self, *args, **kwargs):
        return True

    def get_prediction(self, clicker, prev_mask=None, step=-1):
        clicks_list = clicker.get_clicks()

        if step == 0:
            need_intention = True
        else:
            need_intention = self.is_need_intention()

        # > Last prediction as previous mask
        if self.eval_mode == 'normal':
            raise NotImplementedError
        # + Cascade prediction as previous mask
        elif self.eval_mode == 'cascade':
            if need_intention:
                prediction = self.infer_intention(prev_mask, clicks_list)
                self.valid_intention_count += 1
            else:
                prediction = prev_mask if prev_mask is not None else self.prev_prediction
                self.ignore_intention_count += 1
        else:
            raise NotImplementedError

        input_image = torch.cat((self.gt_mask, self.original_image, prediction), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )

        zoom_gt_mask = image_nd[:, 0, :, :]
        image_nd = image_nd[:, 1:, :, :]

        out = self._get_prediction(image_nd, clicks_lists, is_image_changed,
                                   zoom_gt_mask=zoom_gt_mask, net_type='seg')

        pred_logits = out['instances']

        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in list(reversed(self.transforms))[:-1]:
            prediction = t.inv_transform(prediction)

        for t in list(reversed(self.transforms))[-1:]:
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = prediction

        return prediction.cpu().numpy()[0, 0], None

    def infer_intention(self, prev_mask, clicks_list):
        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.int_net, 'with_prev_mask') and self.int_net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        input_image = torch.cat((self.gt_mask, input_image), dim=1)

        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list], without_zoomin=True
        )

        zoom_gt_mask = image_nd[:, 0, :, :]
        image_nd = image_nd[:, 1:, :, :]

        out = self._get_prediction(image_nd, clicks_lists, is_image_changed,
                                   zoom_gt_mask=zoom_gt_mask, net_type='int')

        if self.cascade_mode == 'global':
            pred_logits = torch.sigmoid(out['instances'])
        elif self.cascade_mode == 'final':
            pred_logits = out['instances']
        else:
            raise NotImplementedError

        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        for t in list(reversed(self.transforms))[:-1]:
            prediction = t.inv_transform(prediction)

        self.transforms[0].set_probs(prediction.detach().cpu().numpy())
        return prediction

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed, net_type, zoom_gt_mask=None):
        points_nd = self.get_points_nd(clicks_lists)
        if net_type == 'seg':
            return self.seg_net(image_nd, points_nd, gt_mask=zoom_gt_mask)
        elif net_type == 'int':
            return self.int_net(image_nd, points_nd, gt_mask=zoom_gt_mask)
        else:
            raise NotImplementedError

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists, without_zoomin=False):
        is_image_changed = False
        for t in self.transforms:
            if without_zoomin:
                without_zoomin = False
                continue
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']

    def zoom_out(self, out, zoom_pred):
        B, C, H, W = out['instances'].shape
        H = int(H / 2 + 0.5)
        W = int(W / 2 + 0.5)
        pred = torch.zeros((B, C, H, W), device=zoom_pred.device)
        B = zoom_pred.size()[0]
        for b in range(B):
            rmin, rmax, cmin, cmax = out['zoom_in']['bboxes'][b].long().cpu().numpy()
            size = (rmax - rmin + 1, cmax - cmin + 1)
            _zoom_pred = F.interpolate(zoom_pred[b:b+1], size=size, mode='bilinear', align_corners=True)
            pred[b:b+1, 0, rmin:rmax + 1, cmin:cmax + 1] = _zoom_pred
        return pred
