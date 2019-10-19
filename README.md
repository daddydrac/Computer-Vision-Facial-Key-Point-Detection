## Computer Vision: Facial Key Point Detection ##
Facial Keypoint Detection Using Haar Cascades on a trained CNN: This project uses computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face

 --------------------------------------------------------------
 
 #### How HAAR Feature Detection works with CNNs ####
 
Summary: HAAR Feature Detection is gradient measuments that look at rectangular regions. HAAR features detect patterns like edges lines and more complex rectangular patterns. The image below is an example of vertical, horizatonal and rectangle "HAAR Feature Detectors" or "HAAR Line Detectors":

 ![Image description](https://github.com/joehoeller/Computer-Vision-Facial-Key-Point-Detection/blob/master/app/facial-keypoint-detection/misc/haar-features.png)

So, it looks at an image and applies one of the Haar Feature Detectors, like vertical line detector, and then performs classification of the entire image, if it doesnt get enough of a feature response, it classifies it as "not" face, and discards that info (the region in red is discarded):

 ![HAAR Ferature Detection](https://github.com/joehoeller/Computer-Vision-Facial-Key-Point-Detection/blob/master/app/facial-keypoint-detection/misc/not-face.png)

Then it then feeds this reduced image area to the next feature detector and classifies the image again, discarding irrelvant, or in this case, non-face areas at every step. This is called a cascade of classifiers.


--------------------------------------------------------------

<strong>Production Usage in Drone, IoT and Social Media Apps</strong> 
<p>HAAR Cascades can also be used to identiy an area of interest for further processing.</P>

1. <em>Stanford University Paper:</em> [Cascaded Classification Models: Combining Models for Holistic Scene Understanding](http://ai.stanford.edu/people/koller/Papers/Heitz+al:NIPS08a.pdf). Written by: <em>Geremy Heitz Stephen Gould Department of Electrical Engineering Stanford University, Ashutosh Saxena and Daphne Koller Department of Computer Science Stanford University.</em>

2. <em>Article:</em> <strong>HAAR Wavelets</strong> explained in further detail with working examples (Article with GitHub) as used in Facebook. [Face Detection Using OpenCV With Haar Cascade Classifiers](https://becominghuman.ai/face-detection-using-opencv-with-haar-cascade-classifiers-941dbb25177)

--------------------------------------------------------------
 
 #### Contents of Code 
 
- Notebook 1 : Loading and Visualizing the Facial Keypoint Data

- Notebook 2 : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

- Notebook 3 : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

- Notebook 4 : Filters and Keypoint Uses

- models.py: Custom CNN built in PyTorch

-----------------------------------------------------------------

### How to launch container

### Before you begin (This might be optional) ###

Link to nvidia-docker2 install: [Tutorial](https://medium.com/@sh.tsang/docker-tutorial-5-nvidia-docker-2-0-installation-in-ubuntu-18-04-cb80f17cac65)

You must install nvidia-docker2 and all it's deps first, assuming that is done, run:


 ` sudo apt-get install nvidia-docker2 `
 
 ` sudo pkill -SIGHUP dockerd `
 
 ` sudo systemctl daemon-reload `
 
 ` sudo systemctl restart docker `
 

How to run this container:


### Step 1 ###

` docker build -t <container name> . `  < note the . after <container name>


### Step 2 ###

Run the image, mount the volumes for Jupyter and app folder for your fav IDE, and finally the expose ports `8888` for Jupyter Notebook:


` docker run --rm -it --runtime=nvidia --user $(id -u):$(id -g) --group-add container_user --group-add sudo -v "${PWD}:/app" -p 8888:8888  <container name> `


### Step 3: Check to make sure GPU drivers and CUDA is running ###

- Exec into the container and check if your GPU is registering in the container and CUDA is working:

- Get the container id:

` docker ps `

- Exec into container:

` docker exec -u root -t -i <container id> /bin/bash `

- Check if NVIDIA GPU DRIVERS have container access:

` nvidia-smi `

- Check if CUDA is working:

` nvcc -V `

--------------------------------------------------


### Known conflicts with nvidia-docker and Ubuntu ###

AppArmor on Ubuntu has sec issues, so remove docker from it on your local box, (it does not hurt security on your computer):

` sudo aa-remove-unknown `

--------------------------------------------------
