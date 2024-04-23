# ResNets
To demonstrate proficiency with convolutional neural networks and coding

**Skills demonstrated:** multi-classification, tensorflow, ResNet architecture, model evaluation, general coding

Overview: 
> ResNet architecture (He et al, 2015) famously revolutionized deep learning by elegantly addressing the issue of vanishing gradients in very deep networks by adding skip connections. This code builds a deep neural network using ResNet architecture. For the sake of comparison with pre-ResNet architectures, I include helper functions that easily allow the removal of skip connections. The model is trained on images of hands holding up 0-5 fingers and returns an integer corresponding to the gesture in each image.

Models:
> - Images are initially fed into an input layer, then through a ZeroPadding layer, a Conv2D layer, BatchNormalization, ReLU activation, then a MaxPool layer. From there, output is passed through a prescribed number of what I'll call ResNet blocks. Each ResNet block contains a convolutional block and a prescribed number of identity blocks.
>- The convolutional blocks pass the input through three Conv2D layers with BatchNormalization and ReLU activation between, then pass the block input through a Conv2D and BatchNorm and adds it to the output of the third Conv2D layer before passing that sum through a ReLU activation. These blocks can change the dimension of the data.
>- The identity blocks similarly pass the input through three Conv2D layers with BatchNormalization and ReLU activation between, but simply adds the output of the third Conv2D layer to the block input without transforming the input first. These blocks maintain data dimensions.
>- After the last ResNet block, the output is passed through an AveragePooling2D layer, then a Flatten layer, then a Dense layer with softmax activation for prediction.

Purpose: 
> Demonstrate the effectiveness of skip connections in addressing the issue of vanishing gradients in very deep neural networks.

Datasets: 

> SIGNS dataset from DeepLearning.AI's Convolutional Neural Networks course. The dataset consists of 1080 training images and 120 test images of hands gesturing 0 through 5.
    

Target variable:

> The number of digits held up in each image (0-5).
