# Single Shot Object Detection

Single shot object detection using YOLO V3 object detection model. You only look once (YOLO) is a state-of-the-art, real-time object detection system. Here a single neural network is applied to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities. The paper can be founded here:

[YOLOv3: An Incremental Improvement. Joseph Redmon, Ali Farhadi](https://arxiv.org/abs/1804.02767)


### Download Model 

```shell
    cd model && wget https://pjreddie.com/media/files/yolov3.weights
``` 

#### Run App 

Go To 

cd into root directory & cd into app directory.

```shell
    cd .. 
    cd app 
```  
run app 

```shell
    streamlit run ./app.py 
```  
### Run Model

Run depth estimation on a single image

```shell
    object_detection_image.py image_name.jpg 
``` 
