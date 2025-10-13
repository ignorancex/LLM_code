import cv2
import time
import json
import math
import rospy
import torch
import faiss
import random
import datetime
import threading
import numpy as np
from pathlib import Path
from .losses import LRIZZ


# Keycodes for labeling interface
KEY_CONFIRM = 32
KEY_DOWN = 97
KEY_UP = 100


class ChungusBackend:
    def __init__(self,
                 initial_embeddings_file,
                 initial_images_folder,
                 results_folder,
                 controller_paused_param,
                 threshold_stdevs,
                 model_resolution,
                 device,
                 traversability_prediction_node,
                 train_epochs=50,
                 train_decay_gamma=0.25,
                 train_decay_step=50,
                 train_lr=0.005,
                 train_wd=0.0,
                 equality_threshold=0.25,
                 lrizz_L=0.5):
        
        """ Backend for CHUNGUS

        :param initial_embeddings_file: the file to use for initializing embeddings
        :param initial_images_folder: initial folder for images (used for cross image labeling database)
        :param results_folder: root folder to save results to (specific results are saved to a subfolder with date/time)
        :param controller_paused_param: param name for the controller pause parameter (written to by CHUNGUS)
        :param threshold_stdevs: stdev threshold for Faiss (alpha in paper)
        :param model_resolution: resolution for the model (only used for recording purposes)
        :param device: device for inference (must be cuda for FeatUp)
        :parma traversability_prediction_node: chungus traversability node
        :param train_epochs: epochs to train for
        :param train_decay_gamma: learning rate decay gamma
        :param train_decay_step: learning rate decay step
        :param train_lr: train learning rate
        :param train_wd: train weight decay
        :param equality_threshold: threshold for equality
        :param lrizz_L: l hyperparameter for LRIZZ loss
        """
        
        # Setup files/folders 
        self.initial_embeddings_file = Path(initial_embeddings_file)
        self.initial_images_folder = Path(initial_images_folder)
        self.results_folder = Path(results_folder)
        assert(self.initial_embeddings_file.exists() and self.initial_images_folder.exists() and not self.results_folder.exists())
        self.settings_file = self.results_folder / Path('settings.json')
        self.gen_embeddings_file = self.results_folder / Path('generated_embeddings.npy')
        self.gen_images_folder = self.results_folder / Path('images')
        self.label_images_folder = self.results_folder / Path('labeled_images')
        self.model_save_file = self.results_folder / Path('saved_model.pth')
        self.initial_model_save_file = self.results_folder / Path('initial_model.pth')

        # Scale factor for annotation image
        self.annotation_scale_factor = 2

        self.controller_paused_param = controller_paused_param
        self.threshold_stdevs = threshold_stdevs
        self.model_resolution = model_resolution
        self.device = device
        self.traversability_prediction_node = traversability_prediction_node
        self.train_epochs = train_epochs
        self.train_decay_gamma = train_decay_gamma
        self.train_decay_step = train_decay_step
        self.train_lr = train_lr
        self.train_wd = train_wd
        self.equality_threshold = equality_threshold
        self.lrizz = LRIZZ(lrizz_L)

        # Initialization
        self.create_results_directory()
        self.embeddings = None
        self.embeddings_lock = threading.Lock()
        self.read_embeddings(self.initial_embeddings_file)
        self.novelty_threshold = float('-inf')
        self.index = None
        self.index_metadata = None
        self.index_lock = threading.Lock()
        self.create_index()
        self.write_embeddings()

    def save_initial_model(self):
        """ Save the initial model """

        self.traversability_prediction_node.model.save_model(self.initial_model_save_file)
        rospy.loginfo("Saving model to file: {}".format(str(self.initial_model_save_file)))

    def create_results_directory(self):
        """ Create a results folder and save settings """

        self.results_folder.mkdir(parents=True, exist_ok=False)
        self.gen_images_folder.mkdir(exist_ok=False)
        self.label_images_folder.mkdir(exist_ok=False)
        settings = {
            'model': {
                'name': self.traversability_prediction_node.model.get_model_name(),
                'res_x': self.model_resolution[1],
                'res_y': self.model_resolution[0],
            },
            'train': {
                'epochs': self.train_epochs,
                'lr': self.train_lr,
                'decay_step': self.train_decay_step,
                'decay_gamma': self.train_decay_gamma
            },
            'controller_paused_param': self.controller_paused_param,
            'device': str(self.device),
            'initial_embeddings': str(self.initial_embeddings_file),
            'initial_images': str(self.initial_images_folder),
            'gen_embeddings': str(self.gen_embeddings_file),
            'gen_images': str(self.gen_images_folder),
            'label_images': str(self.label_images_folder)
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)

    def read_embeddings(self, file):
        """ Read embeddings from a file """

        with self.embeddings_lock:
            self.embeddings = np.load(file, allow_pickle=True).item()
            rospy.loginfo("Read embeddings file. {} embeddings found (with {} cls tokens)".format(
                self.embeddings['embeddings'].shape[0], self.embeddings['cls_tokens'].shape[0]
            ))

    def write_embeddings(self):
        """ Write embeddings to a file """

        rospy.loginfo("Writing embeddings to {}".format(str(self.gen_embeddings_file)))
        with self.embeddings_lock:
            np.save(self.gen_embeddings_file, self.embeddings)

    def create_index(self):
        """ Create a FAISS index """

        with self.index_lock:
            with self.embeddings_lock:
                self.index = faiss.IndexFlatL2(self.embeddings['cls_tokens'].shape[-1])
                if self.embeddings['cls_tokens'].shape[0] > 0:
                    self.index.add(self.embeddings['cls_tokens'])
                    assert(self.index.is_trained)
                    rospy.loginfo("Updated index (now contains {} items)".format(self.index.ntotal))

                    errors = None
                    if self.embeddings['cls_tokens'].shape[0] > 1:
                        errors = self.index.search(self.embeddings['cls_tokens'], k=2)[0][:,1] # Get second nearest neighbor distance
                    
                    # Populate metadata
                    self.index_metadata = {
                        'mean_error': float(errors.mean()) if errors is not None else 0.0,
                        'stdev_error': float(errors.std()) if errors is not None else 0.0
                    }
                else:
                    rospy.loginfo("No CLS tokens in embeddings file. Creating empty index...")
                    # Populate metadata
                    self.index_metadata = {
                        'mean_error': 0.0,
                        'stdev_error': 0.0
                    }
            
            # Update novelty threshold
            self.novelty_threshold = self.index_metadata['mean_error'] + self.threshold_stdevs * self.index_metadata['stdev_error']

    def update_index(self, inference_results):
        """ Update the FAISS index """

        with self.index_lock:
            embedding = inference_results['cls_token']
            assert(embedding is not None)
            self.index.add(np.expand_dims(embedding, axis=0))
            with self.embeddings_lock:
                self.embeddings['cls_tokens'] = np.concatenate([
                    self.embeddings['cls_tokens'],
                    np.expand_dims(embedding, axis=0)
                ], axis=0)
                errors = None
                if self.embeddings['cls_tokens'].shape[0] > 1:
                    errors = self.index.search(self.embeddings['cls_tokens'], k=2)[0][:,1]

            # Populate metadata
            self.index_metadata = {
                'mean_error': float(errors.mean()) if errors is not None else 0.0,
                'stdev_error': float(errors.std()) if errors is not None else 0.0
            }

            # Update novelty threshold
            self.novelty_threshold = self.index_metadata['mean_error'] + self.threshold_stdevs * self.index_metadata['stdev_error']

    def compute_novelty(self, inference_results):
        """ Compute novelty using FAISS index """

        with self.index_lock:
            embedding = inference_results['cls_token']
            if self.index.ntotal > 0:
                novelty_score = float(self.index.search(np.expand_dims(embedding, axis=0), k=1)[0][0,0])
                is_novel = novelty_score >= self.novelty_threshold
            else:
                novelty_score = float('nan')
                is_novel = True

        return {'is_novel': is_novel, 'novelty_score': novelty_score}
    
    def retrain_model(self):
        """ Retrain the model """

        # Fetch the embeddings to train on
        # self.embedding['embeddings'] will be of shape: (N, 2, 384)
        # self.embedding['labels'] will be of shape (N,)
        elapsed = time.time()
        with self.embeddings_lock:
            # embedding -> (N,2,384) -> (N,2,384,1) -> (N,384,2,1)
            if self.embeddings['embeddings'].shape[0] == 0:
                rospy.logwarn("Train requested but not completed due to no data.")
                return
            else:
                emb = torch.tensor(self.embeddings['embeddings'], dtype=torch.float, device=self.device).unsqueeze(-1).permute(0,2,1,3)
                lab = torch.tensor(self.embeddings['labels'], dtype=torch.long, device=self.device)
        
        # Create new prediction head
        prediction_head = self.traversability_prediction_node.model.get_new_prediction_head()
        prediction_head.train()
        prediction_head.to(self.device)

        # Train
        if self.train_wd is not None and self.train_wd > 0:
            optimizer = torch.optim.AdamW(prediction_head.parameters(), lr=self.train_lr, weight_decay=self.train_wd)
        else:
            optimizer = torch.optim.Adam(prediction_head.parameters(), lr=self.train_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.train_decay_step, gamma=self.train_decay_gamma)

        losses = []
        best_state_dict = None

        for epoch in range(self.train_epochs):
            predictions = prediction_head(emb)
            loss_tensor = self.lrizz(
                {k: v[:,0,0,0] for k,v in predictions.items()},
                {k: v[:,0,1,0] for k,v in predictions.items()},
                lab
            )
            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()
            scheduler.step()
            loss_value = loss_tensor.detach().cpu().item()
            if len(losses) == 0 or loss_value < min(losses): # best
                best_state_dict = prediction_head.state_dict().copy()
            losses.append(loss_value)
        assert(best_state_dict is not None)

        # Set new prediction head
        prediction_head.load_state_dict(best_state_dict)
        prediction_head.eval()
        self.traversability_prediction_node.update_model(best_state_dict)
        del prediction_head
        elapsed = time.time() - elapsed

        # Log info
        rospy.loginfo("Finished training in {:.3f} seconds (best loss = {:.3f} @ epoch = {})".format(
            elapsed,
            min(losses),
            1+np.argmin(losses)
        ))
        rospy.loginfo("Training was performed on embeddings: {} and labels: {}".format(
            emb.shape, lab.shape
        ))

        return elapsed
    
    def pause_controller(self, paused=True):
        """ Pause the controller by setting the paused parameter """

        if self.controller_paused_param is not None:
            rospy.set_param(self.controller_paused_param, paused)

    def draw_cross(self, img, x, y, w, h, fraction=0.04, thickness=4, color=(0,0,255)):
        """ Draw a crosshair on an image """

        l1A = (int(x - w*fraction), int(y))
        l1B = (int(x + w*fraction), int(y))
        l2A = (int(x), int(y - w*fraction))
        l2B = (int(x), int(y + w*fraction))
        cv2.line(img, l1A, l1B, color, thickness=thickness)
        cv2.line(img, l2A, l2B, color, thickness=thickness)

    def get_suggested_label(self, predictionA, predictionB):
        """ Provide a suggested label from given predictions
         
        :param predictionA: first prediction
        :param predictionB: second prediction
        :returns: -1, 0, 1 depending on ordinal relation
        """

        diff = predictionB - predictionA
        if diff > self.equality_threshold:
            return 1
        elif diff < -self.equality_threshold:
            return -1
        else:
            return 0
        
    def imshow_scaled(self, window_name, render_img):
        """ Show an image but taking into account the scale factor 
        
        :param window_name: window name
        :param render_img: image to render (unscaled)
        """

        if self.annotation_scale_factor is None:
            cv2.imshow(window_name, render_img)
        else:
            cv2.imshow(
                window_name,
                cv2.resize(render_img, (0,0), fx=self.annotation_scale_factor, fy=self.annotation_scale_factor)
            )

    def label_intra(self, intra_image, intra_inference, window_name):
        """ Provide an intra-image label
        
        :param intra_image: image to label
        :param intra_inference: inference results on image
        :param window_name: name of window
        :returns: dict with embeddings, labeled image, and labels 
        """
        
        W, H = intra_image.shape[:2][::-1]

        # generate locations
        rospy.loginfo("Labeling for intra-image label")
        x1, x2 = random.randrange(0, W), random.randrange(0, W)
        y1, y2 = random.randrange(0, H), random.randrange(0, H)
        while math.sqrt((x1 - x2)**2 + (y1 - y2)**2) < min(0.05*W, 0.05*H): # don't allow locations to be super close to each other
            x2, y2 = random.randrange(0, W), random.randrange(0, H)

        # just to make it a little easier for the annotator, we make x1 always be left-most
        if x2 < x1:
            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)

        if self.equality_threshold is not None: # create initial label
            label = self.get_suggested_label(intra_inference['prediction_raw'][y1,x1], intra_inference['prediction_raw'][y2,x2])
        else:
            label = 0
        
        render_img = None
        while True:
            render_img = cv2.cvtColor(intra_image.copy(), cv2.COLOR_RGB2BGR)
            if label == 0: # 0 => same label
                self.draw_cross(render_img, x1, y1, W, H, color=(255,0,0))
                self.draw_cross(render_img, x2, y2, W, H, color=(255,0,0))
            elif label == 1: # 1 => second is more traversable
                self.draw_cross(render_img, x1, y1, W, H, color=(0,0,255))
                self.draw_cross(render_img, x2, y2, W, H, color=(0,255,0))
            else: # -1 => first is more traversable
                self.draw_cross(render_img, x1, y1, W, H, color=(0,255,0))
                self.draw_cross(render_img, x2, y2, W, H, color=(0,0,255))

            self.imshow_scaled(window_name, render_img)
            key_code = 0
            while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
                key_code = cv2.waitKey(100)
                if key_code in [KEY_CONFIRM, KEY_DOWN, KEY_UP]:
                    break
                
            if key_code == KEY_CONFIRM:
                break
            elif key_code == KEY_DOWN:
                label = max(label - 1, -1)
            elif key_code == KEY_UP:
                label = min(label + 1, 1)
        
        return {
            'embeddings': np.expand_dims(np.stack([intra_inference['features'][:,y1,x1], intra_inference['features'][:,y2,x2]], axis=0), axis=0),
            'rendered_image': render_img,
            'labels': np.array([label])
        }

    def label_cross(self, intra_image, intra_inference, cross_image, cross_inference, window_name):
        """ Provide an cross-image label
        
        :param intra_image: intra image to label
        :param intra_inference: inference results on intra image
        :param cross_image: cross image to label
        :param cross_inference: inference results on cross image
        :param window_name: name of window
        :returns: dict with embeddings, labeled image, and labels 
        """
        
        assert(intra_image.shape[:2][::-1] == cross_image.shape[:2][::-1])
        # We make the assumption of same resolution for both images
        W, H = intra_image.shape[:2][::-1]

        # generate locations
        rospy.loginfo("Labeling for cross-image label")
        # x1,y1 should be the min (most familiar in previous image)
        # x2,y2 should be the max (least familiar in current image)
        # The 'reconstruction' inference results are (H,W)
        y1, x1 = np.unravel_index(cross_inference['reconstruction'].argmin(), cross_inference['reconstruction'].shape)
        y2, x2 = np.unravel_index(intra_inference['reconstruction'].argmax(), intra_inference['reconstruction'].shape)

        if self.equality_threshold is not None: # create initial label
            label = self.get_suggested_label(cross_inference['prediction_raw'][y1,x1], intra_inference['prediction_raw'][y2,x2])
        else:
            label = 0
        
        render_img = None
        while True:
            render_img = cv2.cvtColor(np.concatenate([cross_image.copy(), intra_image.copy()], axis=1), cv2.COLOR_RGB2BGR)
            if label == 0: # 0 => same label
                self.draw_cross(render_img, x1, y1, W, H, color=(255,0,0))
                self.draw_cross(render_img, W+x2, y2, W, H, color=(255,0,0))
            elif label == 1: # 1 => second is more traversable
                self.draw_cross(render_img, x1, y1, W, H, color=(0,0,255))
                self.draw_cross(render_img, W+x2, y2, W, H, color=(0,255,0))
            else: # -1 => first is more traversable
                self.draw_cross(render_img, x1, y1, W, H, color=(0,255,0))
                self.draw_cross(render_img, W+x2, y2, W, H, color=(0,0,255))

            self.imshow_scaled(window_name, render_img)
            key_code = 0
            while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
                key_code = cv2.waitKey(100)
                if key_code in [KEY_CONFIRM, KEY_DOWN, KEY_UP]:
                    break
            
            if key_code == KEY_CONFIRM:
                break
            elif key_code == KEY_DOWN:
                label = max(label - 1, -1)
            elif key_code == KEY_UP:
                label = min(label + 1, 1)
        
        return {
            'embeddings': np.expand_dims(np.stack([cross_inference['features'][:,y1,x1], intra_inference['features'][:,y2,x2]], axis=0), axis=0),
            'rendered_image': render_img,
            'labels': np.array([label])
        }

    def label_images(self,
                     intra_image, intra_inference, intra_labeled_image_file,
                     cross_image, cross_inference, cross_labeled_image_file):
        
        """ Request user to perform labeling of images
        
        :param intra_image: intra image to label
        :param intra_inference: inference results on intra image
        :param intra_labeled_image_file: file to save labeled intra image label to
        :param cross_image: cross image to label
        :param cross_inference: inference results on cross image
        :param cross_labeled_image_file: file to save labeled cross image label to
        """
        
        W, H = intra_image.shape[:2][::-1]

        window_name = "window"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # Intra
        intra_time = time.time()
        annotation_intra = self.label_intra(intra_image, intra_inference, window_name)
        intra_time = time.time() - intra_time

        # Cross
        if cross_image is None:
            cross_time = None
            annotation_cross = None
        else:
            cross_time = time.time()
            annotation_cross = self.label_cross(intra_image, intra_inference, cross_image, cross_inference, window_name)
            cross_time = time.time() - cross_time

        cv2.destroyAllWindows()

        # Add the annotations to the embeddings
        with self.embeddings_lock:
            self.embeddings['timings'].append(intra_time)
            self.embeddings['is_initial_label'].append(False)

            if annotation_cross is not None: # Add cross label if cross image was specified
                self.embeddings['timings'].append(cross_time)
                self.embeddings['is_initial_label'].append(False)
                self.embeddings['embeddings'] = np.concatenate([
                    self.embeddings['embeddings'], annotation_intra['embeddings'], annotation_cross['embeddings']
                ], axis=0)
                self.embeddings['labels'] = np.concatenate([
                    self.embeddings['labels'], annotation_intra['labels'], annotation_cross['labels']
                ], axis=0)
            else: # Cross image not specified
                self.embeddings['embeddings'] = np.concatenate([
                    self.embeddings['embeddings'], annotation_intra['embeddings']
                ], axis=0)
                self.embeddings['labels'] = np.concatenate([
                    self.embeddings['labels'], annotation_intra['labels']
                ], axis=0)
        
        # Now write the labeled images
        cv2.imwrite(str(intra_labeled_image_file), annotation_intra['rendered_image'])
        if annotation_cross is not None:
            cv2.imwrite(str(cross_labeled_image_file), annotation_cross['rendered_image'])
    
    def relabel(self, image, inference_results):
        """ Initiate a relabeling
         
        :param image: image to relabel
        :param inference results: inference results
        """

        # Pause robot during relabeling
        self.pause_controller(paused=True)

        # Compute the novelty for book-keeping reasons
        novelty = self.compute_novelty(inference_results)

        # Get the intra and cross image data
        image_resolution = image.shape[:2][::-1] # (W,H)
        intra_image = image
        with self.embeddings_lock:
            if len(self.embeddings['image_files']) > 0:
                cross_image_index = random.randrange(0, len(self.embeddings['image_files']))
                if self.embeddings['is_initial_image'][cross_image_index] == True:
                    cross_image_file = self.initial_images_folder / Path(self.embeddings['image_files'][cross_image_index])
                else:
                    cross_image_file = self.gen_images_folder / Path(self.embeddings['image_files'][cross_image_index])
            else:
                cross_image_file = None
        
        if cross_image_file is not None:
            rospy.loginfo("Using cross image file {}".format(str(cross_image_file)))
            cross_image = cv2.cvtColor(cv2.resize(cv2.imread(cross_image_file), image_resolution, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            inference_results_cross = self.traversability_prediction_node.perform_inference(cross_image)
        else:
            rospy.loginfo("Not using a cross image label as there are no prior images")
            cross_image = None
            inference_results_cross = None
        
        # Write the new image to label to a file
        label_time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        intra_image_file = self.gen_images_folder / Path("img{}.jpg".format(label_time_str))
        rospy.loginfo("Saving image to file {}".format(str(intra_image_file)))
        cv2.imwrite(str(intra_image_file), cv2.cvtColor(intra_image, cv2.COLOR_RGB2BGR))

        # Now perform labeling
        intra_labeled_image_file = self.label_images_folder / Path("intra_{}.jpg".format(label_time_str))
        cross_labeled_image_file = self.label_images_folder / Path("cross_{}.jpg".format(label_time_str))
        self.label_images(intra_image, inference_results, intra_labeled_image_file,
                          cross_image, inference_results_cross, cross_labeled_image_file)

        # Update the metadata
        with self.embeddings_lock:
            self.embeddings['novelty_scores'].append(novelty['novelty_score'])
            self.embeddings['is_novel'].append(novelty['is_novel'])
            self.embeddings['is_initial_image'].append(False)
            self.embeddings['image_files'].append(intra_image_file)
            # cls_tokens automatically updated by update_index
            # label specific embeddings entries updated in label_images

        # Now add to index
        self.update_index(inference_results)

        # Retrain model
        model_train_time = self.retrain_model()
        self.traversability_prediction_node.model.save_model(self.model_save_file)
        rospy.loginfo("Saving model to file: {}".format(str(self.model_save_file)))
        with self.embeddings_lock:
            self.embeddings['training_times'].append(model_train_time)

        # Save the embeddings to a file
        self.write_embeddings()

        # Resume robot after labeling
        self.pause_controller(paused=False)
