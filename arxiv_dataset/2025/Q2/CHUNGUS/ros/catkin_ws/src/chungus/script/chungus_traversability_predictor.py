#!/root/miniconda3/envs/chungus/bin/python
import cv2
import rospy
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from chungus.networks import create_model
from chungus.chungus_backend import ChungusBackend
from sensor_msgs.msg import CompressedImage, Image


class ChungusTraversabilityPrediction:
    def __init__(self):

        # Model initialization
        self.model_variant = rospy.get_param('~traversability_model_variant', 'dinov2featup_recons')
        self.model_resolution = (rospy.get_param('~traversability_model_resy', 224), rospy.get_param('~traversability_model_resx', 224))
        self.model = create_model(self.model_variant, output_size=self.model_resolution)

        # Setup model
        assert(torch.cuda.is_available())
        self.device = torch.device('cuda')
        self.model.to(self.device)
        self.model.eval()

        # Chungus functionality
        self.run_id = "run_{}".format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f'))
        self.prediction_paused = False
        self.controller_paused_param = rospy.get_param('~controller_paused_param', '/controller_paused')
        self.controller_active_param = rospy.get_param('~controller_active_param', '/controller_active')
        self.retrain_on_start = rospy.get_param('~retrain_on_start', False)
        self.use_novelty_detection = rospy.get_param('~use_novelty_detection', True)
        self.use_novelty_only_on_control_active = rospy.get_param('~use_novelty_only_on_control_active', True) # if true, then novelty detection only occurs when controller is active
        self.novelty_stdevs = rospy.get_param('~novelty_stdevs', 2.0) # stdev above mean distance for novelty
        self.chungus_train_epochs = rospy.get_param('~train_epochs', 50)
        self.chungus_train_decay_gamma = rospy.get_param('~train_decay_gamma', 0.25)
        self.chungus_train_decay_step = rospy.get_param('~train_decay_step', 50)
        self.chungus_train_lr = rospy.get_param('~train_lr', 0.005)
        self.chungus_train_wd = rospy.get_param('~train_wd', 0.0)
        self.chungus_label_eq_threshold = rospy.get_param('~equality_threshold', 0.25) # equality threshold for pseudo labeling that speeds up the annotation process
        self.chungus_lrizz_L = rospy.get_param('~lrizz_L', 0.5)

        self.initial_embeddings_file = Path(rospy.get_param('~init_embeddings_file', ''))
        self.initial_images_folder = Path(rospy.get_param('~init_images_folder', ''))
        self.results_folder = Path(rospy.get_param('~gen_results_folder', '')) / Path(self.run_id)
        self.chungus_backend = ChungusBackend(
            initial_embeddings_file=self.initial_embeddings_file,
            initial_images_folder=self.initial_images_folder,
            results_folder=self.results_folder,
            controller_paused_param=self.controller_paused_param,
            threshold_stdevs=self.novelty_stdevs,
            model_resolution=self.model_resolution,
            device=self.device,
            traversability_prediction_node=self,
            train_epochs=self.chungus_train_epochs,
            train_decay_gamma=self.chungus_train_decay_gamma,
            train_decay_step=self.chungus_train_decay_step,
            train_lr=self.chungus_train_lr,
            train_wd=self.chungus_train_wd,
            equality_threshold=self.chungus_label_eq_threshold,
            lrizz_L=self.chungus_lrizz_L
        )
        self.chungus_backend.pause_controller(False)

        # Subscribed topics
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/color/image')
        self.camera_data_compressed = rospy.get_param('~use_compressed', False)
        self.traversability_topic = rospy.get_param('~traversability_image_topic', '/chungus/traversability/prediction')
        self.traversability_visualize_topic = rospy.get_param('~traversability_image_visualize_topic', '/chungus/traversability/visualization')
        self.traversability_uncertainty_topic = rospy.get_param('~traversability_image_uncertainty_topic', '/chungus/traversability/uncertainty')

        # Setup publishers and subscribers from image
        self.traversability_publisher = rospy.Publisher(self.traversability_topic, Image, queue_size=1)
        self.traversability_visualize_publisher = rospy.Publisher(self.traversability_visualize_topic, Image, queue_size=1)
        self.traversability_uncertainty_publisher = rospy.Publisher(self.traversability_uncertainty_topic, Image, queue_size=1)
        if self.camera_data_compressed:
            self.image_subscriber = rospy.Subscriber(self.camera_topic, CompressedImage, callback=self.image_callback, queue_size=1, buff_size=2**24)
        else:
            self.image_subscriber = rospy.Subscriber(self.camera_topic, Image, callback=self.image_callback, queue_size=1, buff_size=2**24)

        # Expect normalization to be performed by the network itself
        self.transform = transforms.Compose([transforms.ToTensor()])

        if self.retrain_on_start == True:
            rospy.loginfo("Retraining model...")
            self.prediction_paused = True
            self.chungus_backend.retrain_model()
            self.prediction_paused = False

        rospy.loginfo("Saving initial model...")
        self.prediction_paused = True
        self.chungus_backend.save_initial_model()
        self.prediction_paused = False
        
        rospy.spin()
    
    def update_model(self, state_dict):
        """ Update te state dict of the model """
        # Setup model
        self.model.update_model(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def perform_inference(self, image):
        """ Perform inference on an image
        
        Predictions will be having the same shape (in H, W) as the original image, even if inference resolution was actually lower
        (i.e., will adaptively resize to accommodate model resolution and will resize back to accommodate original resolution)

        :param image: image to perform inference on
        :returns: dictionary containing keys 'prediction', 'prediction_raw', 'cls_token', 'reconstruction', and 'features'
        """

        # Perform inference
        original_resolution = image.shape[:2][::-1] # will be (W,H)
        resized_image = cv2.resize(image, self.model_resolution[::-1], interpolation=cv2.INTER_LINEAR)
        with torch.no_grad():
            network_output = self.model(self.transform(resized_image).float().to(self.device))
            # Convert prediction to format expected
            prediction_raw = cv2.resize(network_output['prediction'].cpu().numpy(), original_resolution, interpolation=cv2.INTER_LINEAR) # prediction will be (H,W) - it has only 1 channel
            prediction = np.clip(255 * prediction_raw, 0, 255)
            prediction = prediction.astype('uint8')

            if 'cls_token' in network_output:
                cls_token = network_output['cls_token'].cpu().numpy()
            else:
                cls_token = None

            if 'features' in network_output:
                # Resize features back to the original input image resolution (probably would be best to remove this and move it to only as-needed part if you want to further improve speed)
                features = F.interpolate(network_output['features'].unsqueeze(0), size=original_resolution[::-1], mode='bilinear')[0].cpu().numpy()
            else:
                features = None

            if 'reconstruction' in network_output:
                reconstruction = F.interpolate(network_output['reconstruction'].unsqueeze(0).unsqueeze(0), size=original_resolution[::-1], mode='bilinear')[0][0].cpu().numpy()
            else:
                reconstruction = None
        
        return {
            'prediction': prediction, # mono8 image with predictions (0-255 as uint8)
            'prediction_raw': prediction_raw, # this is the prediction that is still a float and is 0 (lowest trav) to 1 (highest trav)
            'reconstruction': reconstruction,
            'features': features,
            'cls_token': cls_token
        }
    
    def decode_image(self, camera_msg):
        """ Decodes an image from a message """

        if self.camera_data_compressed:
            # Read compressed image
            image = cv2.cvtColor(cv2.imdecode(np.frombuffer(camera_msg.data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            # Read non-compressed image
            if camera_msg.is_bigendian:
                raise ValueError("Big endian image was received -- not implemented yet")
            
            W, H = camera_msg.width, camera_msg.height
            raw_image_data = np.frombuffer(camera_msg.data, np.uint8).reshape((H, W, -1))
            if camera_msg.encoding == 'bgr8':
                image = cv2.cvtColor(raw_image_data, cv2.COLOR_BGR2RGB)
            elif camera_msg.encoding == 'rgb8':
                image = raw_image_data
            elif camera_msg.encoding == 'bgra8':
                image = cv2.cvtColor(raw_image_data, cv2.COLOR_BGRA2RGB)
            elif camera_msg.encoding == 'rgba8':
                image = cv2.cvtColor(raw_image_data, cv2.COLOR_RGBA2RGB)
            else:
                raise ValueError("Traversability image generator does not support encoding '{}'".format(camera_msg.encoding))
        
        return image
    
    def publish_prediction(self, camera_msg, inference_results):
        """ Publishes a prediction using the provided inference results """
        # Publish image
        # Note: prediction is (H,W)
        prediction = inference_results['prediction']
        msg_trav = Image()
        msg_trav.header.stamp = camera_msg.header.stamp
        msg_trav.header.frame_id = camera_msg.header.frame_id
        msg_trav.height = prediction.shape[0]
        msg_trav.width = prediction.shape[1]
        msg_trav.encoding = "mono8"
        msg_trav.is_bigendian = False
        msg_trav.step = prediction.shape[1]
        msg_trav.data = np.array(prediction).tobytes()
        self.traversability_publisher.publish(msg_trav)

        msg_trav_vis = Image()
        msg_trav_vis.header.stamp = camera_msg.header.stamp
        msg_trav_vis.header.frame_id = camera_msg.header.frame_id
        msg_trav_vis.height = prediction.shape[0]
        msg_trav_vis.width = prediction.shape[1]
        msg_trav_vis.encoding = "bgr8"
        msg_trav_vis.is_bigendian = False
        msg_trav_vis.step = prediction.shape[1] * 3
        msg_trav_vis.data = np.array(cv2.applyColorMap(prediction, cv2.COLORMAP_JET)).tobytes()
        self.traversability_visualize_publisher.publish(msg_trav_vis)

        # Publish uncertainty image (if available)
        uncertainty = inference_results['reconstruction']
        if uncertainty is not None:
            msg_trav_uncertainty = Image()
            msg_trav_uncertainty.header.stamp = camera_msg.header.stamp
            msg_trav_uncertainty.header.frame_id = camera_msg.header.frame_id
            msg_trav_uncertainty.height = uncertainty.shape[0]
            msg_trav_uncertainty.width = uncertainty.shape[1]
            msg_trav_uncertainty.encoding = "32FC1"
            msg_trav_uncertainty.is_bigendian = False
            msg_trav_uncertainty.step = uncertainty.shape[1] * 4
            msg_trav_uncertainty.data = np.array(uncertainty).tobytes()
            self.traversability_uncertainty_publisher.publish(msg_trav_uncertainty)
    
    def image_callback(self, camera_msg):
        """ Callback for when images are received """
        if self.prediction_paused:
            # Don't do predictions if paused
            pass
        else:
            rospy.loginfo("Predicting an image")

            self.prediction_paused = True
            controller_active = rospy.get_param(self.controller_active_param, False)

            # Decode and infer
            image = self.decode_image(camera_msg)
            inference_results = self.perform_inference(image)
            self.publish_prediction(camera_msg, inference_results)

            # Perform novelty detection and/or relabeling
            should_provide_label = False
            if self.use_novelty_detection and (self.use_novelty_only_on_control_active == False or controller_active == True): # Check for novelty (if enabled)
                novelty = self.chungus_backend.compute_novelty(inference_results)
                rospy.loginfo("Novelty measured: {:.3f}. Is novel: {} (threshold = {:.3f})".format(novelty['novelty_score'], novelty['is_novel'], self.chungus_backend.novelty_threshold))
                should_provide_label = novelty['is_novel']
            
            if should_provide_label:
                # Request for labeling by chungus backend
                rospy.loginfo("Requesting a label from user")
                self.chungus_backend.relabel(image, inference_results)
            
            self.prediction_paused = False


if __name__ == "__main__":
    rospy.init_node('chungus_predictor_node')
    ChungusTraversabilityPrediction()
