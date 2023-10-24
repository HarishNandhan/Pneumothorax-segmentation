# SIIM-PCR-Pneumothorax-Segmentation
BLOG : https://namrata-thakur893.medium.com/medical-image-classification-and-segmentation-a-case-study-approach-6b5c7a73b9f5

**Demo Deployed App:**

[![Final Product](https://img.youtube.com/vi/t737eVOKeOY/0.jpg)](https://youtu.be/t737eVOKeOY "Click to play")

**Business Problem:**

Pneumothorax is a medical condition which arises when air leaks into the space between the lung and the chest wall. This air pushes on the outside of the lung and makes it collapse. Thus, pneumothorax can be a complete lung collapse or a portion of the lungs may be collapsed. Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or sometimes for no obvious reason at all. It can be a life-threatening event.
Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. The problem that this case study is dealing with predicts whether the condition exists in the chest x-ray image given and if present it segments the portion of the lungs that is affected. An accurate prediction would be useful in a lot of clinical scenarios to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

**Mapping the real world problem as a Deep Learning problem:**

The data is comprised of images in DICOM format containing run-length-encoded (RLE) masks. The instances of pneumothorax are indicated by encoded binary masks in the annotations. Some training images have multiple annotations depicting the multiple locations of the event. Images without pneumothorax have a mask value of -1. The task is to predict the mask of pneumothorax in the given X-ray image. This task can be mapped as a Semantic Image Segmentation problem.

**Data set Analysis:**

Data Source : https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview

•	**Files given:**  train-rle.csv, stage_2_sample_submission.csv (test_data), train_images, test_images.

•	**Total File Size :** 4GB

•	**Total number of records:** 12,954 (train_data), 3204 (test_data)

*The train-rle.csv contains image IDs and their corresponding RLE masks and the test csv file only contains the image IDs.*

**Real World Business Constraints:**

•	Low latency is important.

•	Mis-classification/ mis-segmentation cost is considerably high as we are dealing with medical data and thus it is very sensitive to such errors.

**Performance Metric:**

Metric(s):

*	Dice Coefficient  (IntersectionOverUnion/IOU)

*	Combo Loss – (Binary Cross Entropy + Dice Loss/ F1 loss)

*	Confusion Matrix
* Classification Metrics -- AUC, Precision, Recall

**This repository contains the following solution :**

* A) First we classify the Chest X-Rays as either No Pneumothorax Present (Label:0) or Pneumothorax Present (Label:1). This part is the image classification part where we are applying transfer learning technique using the pre-trained model CheXNet (a 121 layer DenseNet model that is fine-tuned on Chest X-Ray images) to classify the images.

* B) Then, we build the segmentation part where we apply different segmentation models (Unet, Nested Unet and Double Unet with pre-trained VGG19 as backbone encoder) to predict the masks.

* C) Finally, we build the end pipeline where given an image we predict the label (0/1) and if the label is 1 we pass the image through the segmentation model to get the mask.

**Classfication Metrics:**

![Classification Metrics](https://user-images.githubusercontent.com/56768652/112884926-bac42e00-90ed-11eb-84f4-17b9424f8d7c.JPG)

![Classification ROC](https://user-images.githubusercontent.com/56768652/112884975-ca437700-90ed-11eb-86b7-a6535da10f6e.JPG)

![Classification CV CM](https://user-images.githubusercontent.com/56768652/112884991-cfa0c180-90ed-11eb-86ba-1483ff72cacd.JPG)

![Classification Average_Precision_Cureve](https://user-images.githubusercontent.com/56768652/112885011-d62f3900-90ed-11eb-94db-49188c926daf.JPG)

**Segmentation Metrics:**

* Nested Unet :

![Segment_Unet++](https://user-images.githubusercontent.com/56768652/112885100-f232da80-90ed-11eb-8bd9-ee7380733d54.JPG)

* Weighted Nested Unet :

![Segmentation_Metrics](https://user-images.githubusercontent.com/56768652/113031343-3684b000-91ac-11eb-9951-46a4e3fe2503.JPG)

**FINAL PIPELINE OUTPUT IMAGES :**

* Positive Prediction :

![Final_Inference_Positive](https://user-images.githubusercontent.com/56768652/113337367-4f749900-9345-11eb-8b8a-b8088f3630a2.JPG)

![Final_Inference_Positive_2](https://user-images.githubusercontent.com/56768652/113337385-56031080-9345-11eb-84ce-0713d4039dea.JPG)

![Final_Inference_Positive_3](https://user-images.githubusercontent.com/56768652/113337684-c14ce280-9345-11eb-9e7b-7e91a96375e4.JPG)

![Final_Inference_Positive_4](https://user-images.githubusercontent.com/56768652/113337695-c447d300-9345-11eb-8cc5-6bdb104e1cd3.JPG)

* Negative Prediction :

![Final_Inference_Negative](https://user-images.githubusercontent.com/56768652/113337752-d590df80-9345-11eb-9f44-d02f47df30e7.JPG)

* Failure Case :

![Failure Cases](https://user-images.githubusercontent.com/56768652/113337777-dd508400-9345-11eb-893d-7535ef148490.JPG)

THIS REPO IS WORK IN PROGRESS. NEW ADDITION/UPDATION IS DONE EVERYDAY.
