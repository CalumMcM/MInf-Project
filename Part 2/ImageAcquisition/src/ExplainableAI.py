"""
Core Module for Grad CAM Algorithm
"""

import os
import numpy as np
import tensorflow as tf

from tf_explain.utils.display import grid_display, heatmap_display
from tf_explain.utils.saver import save_rgb

from tensorflow.keras import datasets,models,layers

from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, Dropout
from keras.models import Model
from keras.models import Sequential

from ClassifyImages import build_dataset

#################################################
## The following class has been obtained from  ##
##        the GitHub repo for tf-explain       ##
##    https://github.com/sicara/tf-explain     ##
#################################################

class GradCAM:

    """
    Perform Grad CAM algorithm for a given input
    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(
        self,
        validation_data,
        model,
        class_index,
        layer_name=None,
        use_guided_grads=True,
        colormap=cv2.COLORMAP_VIRIDIS,
        image_weight=0.7,
    ):
        """
        Compute GradCAM for a specific class index.
        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            layer_name (str): Targeted layer for GradCAM. If no layer is provided, it is
                automatically infered from the model architecture.
            colormap (int): OpenCV Colormap to use for heatmap visualization
            image_weight (float): An optional `float` value in range [0,1] indicating the weight of
                the input image to be overlaying the calculated attribution maps. Defaults to `0.7`.
            use_guided_grads (boolean): Whether to use guided grads or raw gradients
        Returns:
            numpy.ndarray: Grid of all the GradCAM
        """
        images, _ = validation_data

        if layer_name is None:
            layer_name = self.infer_grad_cam_target_layer(model)

        outputs, grads = GradCAM.get_gradients_and_filters(
            model, images, layer_name, class_index, use_guided_grads
        )

        cams = GradCAM.generate_ponderated_output(outputs, grads)

        heatmaps = np.array(
            [
                # not showing the actual image if image_weight=0
                heatmap_display(cam.numpy(), image, colormap, image_weight)
                for cam, image in zip(cams, images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    @staticmethod
    def infer_grad_cam_target_layer(model):
        """
        Search for the last convolutional layer to perform Grad CAM, as stated
        in the original paper.
        Args:
            model (tf.keras.Model): tf.keras model to inspect
        Returns:
            str: Name of the target layer
        """
        for layer in reversed(model.layers):
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError(
            "Model does not seem to contain 4D layer. Grad CAM cannot be applied."
        )

    @staticmethod
    def get_gradients_and_filters(
        model, images, layer_name, class_index, use_guided_grads
    ):
        """
        Generate guided gradients and convolutional outputs with an inference.
        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Targeted layer for GradCAM
            class_index (int): Index of targeted class
            use_guided_grads (boolean): Whether to use guided grads or raw gradients
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)

        if use_guided_grads:
            grads = (
                tf.cast(conv_outputs > 0, "float32")
                * tf.cast(grads > 0, "float32")
                * grads
            )

        return conv_outputs, grads

    @staticmethod
    def generate_ponderated_output(outputs, grads):
        """
        Apply Grad CAM algorithm scheme.
        Inputs are the convolutional outputs (shape WxHxN) and gradients (shape WxHxN).
        From there:
            - we compute the spatial average of the gradients
            - we build a ponderated sum of the convolutional outputs based on those averaged weights
        Args:
            output (tf.Tensor): Target layer outputs, with shape (batch_size, Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (batch_size, Hl, Wl, Nf)
        Returns:
            List[tf.Tensor]: List of ponderated output of shape (batch_size, Hl, Wl, 1)
        """

        maps = [
            GradCAM.ponderate_output(output, grad)
            for output, grad in zip(outputs, grads)
        ]

        return maps

    @staticmethod
    def ponderate_output(output, grad):
        """
        Perform the ponderation of filters output with respect to average of gradients values.
        Args:
            output (tf.Tensor): Target layer outputs, with shape (Hl, Wl, Nf),
                where Hl and Wl are the target layer output height and width, and Nf the
                number of filters.
            grads (tf.Tensor): Guided gradients with shape (Hl, Wl, Nf)
        Returns:
            tf.Tensor: Ponderated output of shape (Hl, Wl, 1)
        """
        weights = tf.reduce_mean(grad, axis=(0, 1))

        # Perform ponderated sum : w_i * output[:, :, i]
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

        return cam

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.
        Args:
            grid (numpy.ndarray): Grid of all the heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_rgb(grid, output_dir, output_name)




# Following code obtained from Kaggle page (parts have been modified and layers added to fit this project):
# https://www.kaggle.com/songrise/implementing-resnet-18-using-keras/notebook
class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(32, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(32)
        self.res_1_2 = ResnetBlock(32)
        self.dropout1_3 = Dropout(0.1)
        #self.res_2_1 = ResnetBlock(64, down_sample=True)
        #self.res_2_2 = ResnetBlock(64)
        #self.dropout2_3 = Dropout(0.2)
        #self.res_3_1 = ResnetBlock(32, down_sample=True)
        #self.res_3_2 = ResnetBlock(32)
        # self.res_4_1 = ResnetBlock(512, down_sample=True)
        # self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")
        """
        # Using Manual Search Best Hyperparameters:
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(32, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal", input_shape=(51,51,3))
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(32)
        self.res_1_2 = ResnetBlock(32)
        self.dropout1_3 = Dropout(0.2)
        self.res_2_1 = ResnetBlock(64, down_sample=True)
        self.res_2_2 = ResnetBlock(64)
        self.dropout2_3 = Dropout(0.2)
        #self.res_3_1 = ResnetBlock(32, down_sample=True)
        #self.res_3_2 = ResnetBlock(32)
        # self.res_4_1 = ResnetBlock(512, down_sample=True)
        # self.res_4_2 = ResnetBlock(512)
       
        #self.avg_pool = GlobalAveragePooling2D()
        
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")
        
    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        i = 0
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2]:
            out = res_block(out)
            i += 1
            if i ==2:
                out = self.dropout1_3(out)
            elif i == 4:
              out = self.dropout2_3(out)
        #out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out



def get_model():
  model = Sequential()

  # Conv2D layer
  model.add(
      Conv2D(32, 
            (7, 7), 
            strides=2,
            padding="same",
             kernel_initializer="he_normal",
             input_shape=(51,51,3))
  )
  # Batch Norm
  model.add(BatchNormalization())

  # ResNet Layer 1
  model.add(
      ResnetBlock(32)
  )

  # ResNet Layer 1
  model.add(
      ResnetBlock(32)
  )
  # Dropout Layer 1
  model.add(
      layers.Dropout(0.1)
  )

  # ResNet Layer 2
  model.add(
      ResnetBlock(32, down_sample=True)
  )

  # ResNet Layer 2
  model.add(
      ResnetBlock(32)
  )
  # Dropout Layer 2
  model.add(
      layers.Dropout(0.1)
  )
  # Flatten Layer
  model.add(
      layers.Flatten()
  )

  # Dense Layer
  model.add(
      layers.Dense(3, activation = "softmax")
  )
  
  return model  



def main():
    """
    Given a directory (DIR) till load the
    built dataset and predict the biome 
    of each of the images
    """

    # Test quads for each biome
    # Cat quad 3, Cer Quad 4, Ama Quad 2
    
    DIR = r'/Volumes/GoogleDrive-103278653964135897318/My Drive/AmaToInc-2TIF'

    # Convert TIF images to Numpy Arrays
    build_dataset(DIR)

    X_data = np.load(os.path.join(DIR, 'pred_data.npy'))
    images = np.load(os.path.join(DIR, 'cleaned_images.npy'))

    # Build the ResNet model
    model = get_model() 
  
    model.build(input_shape = (None,51,51,3))
    
    model.compile(optimizer = "adam",loss='categorical_crossentropy', metrics=["accuracy"]) 

    print (model.summary())

    # Load in the models weights
    model.load_weights("/content/drive/MyDrive/ResNet18-2019-Training/Model_2022.h5")
    
    # Start explainer
    explainer = GradCAM()
    
    for idx, image in enumerate(X_data):
        print (f"Explaining {images[idx]}...")
        grid = explainer.explain(image, model, class_index=0)  # 2 is the Caatinga class label

        explainer.save(grid, DIR, images[idx])

if __name__ == "__main__":
    main()
