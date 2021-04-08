# SSD-ObjectDetection and Object Counter

## Project Details

In this project, we will use the SSD MobileNet model for Brick Detection and Counting bricks of Different target classes. For This, I have used TensorFlow 1.15.0 and python 3.7.1

## Model Selection

I have use SSD_Mobilenet_v2 Because This model is lightweight and can be easily used on raspberry pi and other portable devices very easily without converting them into tflite.

### Efficency 

The Model has a speed of 31 microseconds and a Mean Precision Accuracy of 22. 

## Workspace Preparation

Create an Environment with python 3.7.1 
Activate environment and install all these libraries

```bash 
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas OpenCV-python tensorflow==1.15.0
pip install numpy==1.19
conda install -c anaconda protobuf
pip install requests
```
### Download the Github Repository
Create the directory accordingly
'''

          Main Dir
            |
          Models
            |
          Research______Object Detection
            |              |
Main File   |           Utils
            |              |
          Object Detection.py 
'''

In Object Detection.py I have used Post API so that model can publish the count of the bad brick to the server. 
