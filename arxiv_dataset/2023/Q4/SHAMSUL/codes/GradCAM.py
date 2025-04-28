# adapted from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/?fbclid=IwAR3mV62vSMoE2LIB77EPHaLc-cyeNnO_cz9xTliYWyMP9jgbFA2fGeVaOJw 
#          and https://colab.research.google.com/drive/1098Hp2icvleIQelkaLmPoIqAKuahs7JH#scrollTo=BUGkcv9c-1m2
import tensorflow as tf
import cv2
import numpy as np

class GradCAM:
	def __init__(self, model, layerName = None):
		self.model = model
		self.layerName = layerName
		if self.layerName == None:
			self.layerName = self.find_target_layer(self.model)

	def find_target_layer(self, model):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

	def get_heatmap(self, classIdx, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		# Create a graph that outputs target convolution and output

		grad_model = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(self.layerName).output, self.model.output])
		# Get the score for target class
		with tf.GradientTape() as tape:
			conv_outputs, predictions = grad_model(image)
			loss = predictions[:, classIdx]

		# Extract filters and gradients
		output = conv_outputs[0]
		grads = tape.gradient(loss, conv_outputs)[0]

		# Apply guided backpropagation
		gate_f = tf.cast(output > 0, 'float32')
		gate_r = tf.cast(grads > 0, 'float32')
		guided_grads = gate_f * gate_r * grads

		# Average gradients spatially
		weights = tf.reduce_mean(guided_grads, axis=(0, 1))
		# Build a ponderated map of filters according to gradients importance
		cam = np.ones(output.shape[0:2], dtype=np.float32)
		for index, w in enumerate(weights):
			cam += w * output[:, :, index]
		# Heatmap visualization
		cam = cv2.resize(cam.numpy(), (224, 224))
		cam = np.maximum(cam, 0)
		heatmap = (cam - cam.min()) / (cam.max() - cam.min() +eps)
		# return the resulting heatmap to the calling function
		return heatmap