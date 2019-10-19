## Computer Vision: Facial Key Point Detection #
#### Facial Keypoint Detection Using Haar Cascades on a trained CNN: This project uses computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face

 --------------------------------------------------------------
 
 #### How HAAR Feature Detection works with CNNs ####
 
Summary: HAAR Feature Detection is gradient measuments that look at rectangular regions. HAAR features detect patterns like edges lines and more complex rectangular patterns. The image below is an example of vertical, horizatonal and rectangle "Feature Detectors" or "Line Detectors":

<p style="display:table;">
<img align="left" src="haar-features.png" width="100%" height="auto" />
</p>

So, it looks at an image and applies one of the Haar Feature Detectors, like vertical line detector, and then performs classification of the entire image, if it doesnt get enough of a feature response, it classifies it as "not" face, and discards that info:

<p style="display:table;">
<img align="left" src="not-face.png" width="100%" height="auto" />
</p>


Then it then feeds this reduced image area to the next feature detector and classifies the image again, discarding irrelvant, or in this case, non-face areas at every step. This is called a cascade of classifiers. 


<strong>Production Usage:</strong> HAAR Cascades can also be used to identiy an area of interest for further processing.

 
 --------------------------------------------------------------
 
 #### Contents of Code 
 
- Notebook 1 : Loading and Visualizing the Facial Keypoint Data

- Notebook 2 : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

- Notebook 3 : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

- Notebook 4 : Filters and Keypoint Uses

- models.py: Custom CNN built in PyTorch


### How to launch container

See instructions/readme [here](https://github.com/joehoeller/NVIDIA-GPU-Tensor-Core-Accelerator-PyTorch-OpenCV)
