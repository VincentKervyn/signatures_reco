# Handwritten Signatures detection and recognition project
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
  </a>

  <h3 align="center">Yolov5v</h3>
This project needs some additional work 
  <p align="center">
     YOLO uses convolutional neural networks instead of the region-based methods employed by alogirithms like R-CNN. The convolutional network Only Look Once, ie it requires only one forward pass through the neural network to make predictions. It makes two predictions, the class of object and the position of the objects present in the image.
YOLO divides an image into nxn regions and predicts whether a target class is present in each region or not. It also predicts the bounding box coordinates of the target class. Non-max suppression is used to prevent same object being detected multiple times.
Refer this video for a better, in-depth understanding of YOLO models. The original YOLO paper could be accessed here and YOLOv5 repo could be found here

  ### Training On Custom Signature Dataset
**YOLOv5** is available over [this repo](https://github.com/ultralytics/yolov5).  
The data in CycleGAN format is available [here](tobacco_yolo_format.zip) as a zip file.  
We use it to train the model to work with the Signature detection use case.  
YOLOv5 requires the dataset to be organized in a particular structure.  
`dataset`  
&nbsp;&nbsp;&nbsp;&nbsp; |-> `images`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `train` (.jpg files)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `valid` (.jpg files)  
&nbsp;&nbsp;&nbsp;&nbsp; |-> `labels`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `train` (.txt files)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |-> `valid` (.txt files)  
  
### Training the model
**Use [this notebook](CustomYOLOv5_using_Tobcco800_dataset.ipynb) to train and test the YOLOv5 model.**  
  
Clone the official [YOLOv5 repo] and install the requirements using the `requirements.txt` file.  
We need to create a `tobacco_data.yaml` and add the path of training `train:` and validation `valid:` directories, number of classes `nc:` and class names `['DLLogo', 'DLSignature']` and add this file to the `yolov5` directory we cloned.  

Now, we have to select a pre-trained model from the available pre-trained checkpoints. These `model.yaml` files are present inside `yolov5\models`. I have used `yolov5x` for performing my experiments.  
![Yolo models](../../Images/yolo_model_list.png)  
  
**Training arguments**  
`--img 640` is the width of the images.  
`--batch ` - batch size
`--epochs ` - no of epochs  
`--data` - Dataset.yaml (`tobacco_data.yaml`) path  
`--cfg models/model.yaml` is used to set the model we want to train on. I have used yolov5x.yaml, more information could be found [here.](https://github.com/ultralytics/yolov5#pretrained-checkpoints)  
  
**To Train the model**, run the following line.  
> **!python yolov5/train.py --img 640 --batch 8 --epochs 10 --data tobacco_data.yaml --cfg models/yolov5x.yaml --name Tobacco-run**
  A maximum of 10 epcoh was tested, time lacking

Each line of the txt file contains class x_center y_center width height. Box coordinates are in normalized form. x_center and width are divided by image_width and y_center and height are divided by image_height to get a value between 0 to 1.
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Data">Data</a>
    </li>
   
  </ol>
</details>



<!-- Data -->
## Data
Given for the project:
>Test
>Train
>Train-xml


Produced:
>Scaled: originals images are resizd to reduced training time.<
>tobacco_yolo_format: scaled images are preprocess and placed in correct directories to be used for yolov5 






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
