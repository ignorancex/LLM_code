import importlib
from keras.layers import Input, TimeDistributed
from keras.layers.core import Dense
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D, Lambda, Flatten
from keras import backend as K
from models.se import squeeze_excite_block
from models.minmax import min_max_pool2d
from models.custom_insert import insert_layer_nonseq
from keras.layers.convolutional import Conv2D



class ModelFactory:
    """
    Model facotry for Keras default models
    """

    def __init__(self):
        self.models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layer="block5_conv3",
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layer="block5_conv4",
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layer="bn",
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layer="activation_49",
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layer="mixed10",
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layer="conv_7b_ac",
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layer="activation_188",
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layer="activation_260",
            ),
        )

    def get_last_conv_layer(self, model_name):
        return self.models_[model_name]["last_conv_layer"]

    def get_input_size(self, model_name):
        return self.models_[model_name]["input_shape"][:2]
    


    def get_model(self, class_names, model_name="DenseNet121", use_base_weights=True,
                  weights_path=None, input_shape=None):

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                f"keras.applications.{self.models_[model_name]['module_name']}"
            ),
            model_name)

            
        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]

        img_input = Input(shape=input_shape)
        if input_shape is None:
            input_shape = self.models_[model_name]["input_shape"]

        img_input = Input(shape=input_shape)

        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        x = base_model.output
        predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)
        weights_path="models/weights.h5"
        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)

        def minmax_layer_factory():
            return Lambda(min_max_pool2d)
        
        def lambda_layer_factory():
            return Lambda(squeeze_excite_block)
        
        model = insert_layer_nonseq(model, '.*pool\d_pool.*', lambda_layer_factory, position='after')
        
        def channel_avg_pool(x):
            return K.mean(x,axis=3,keepdims=True)

        def channel_avg_pool_output_shape(input_shape):
           shape = list(input_shape)
           shape[1] /= 1
           shape[1] = int(shape[1])
           shape[2] /= 1
           shape[2] = int(shape[2])
           shape[3] /= 12
           shape[3] = int(shape[3])
           return tuple(shape)

        
        relu_output = model.get_layer("relu").get_output_at(-1)
        x = Conv2D(filters=168, kernel_size=(1,1), strides=(1, 1))(relu_output)
        #x = Lambda(channel_avg_pool,output_shape=channel_avg_pool_output_shape)(x)
        #x = Lambda(min_max_pool2d)(x)
        #x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)
        
        #dictionary = {v.name: i for i, v in enumerate(model.layers)}
        #print(dictionary)
        return model
