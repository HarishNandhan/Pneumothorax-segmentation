# Pneumothorax Detection and Segmentation 🩺📸

## Business Problem:

Pneumothorax is a critical medical condition where air leaks into the space between the lung and the chest wall, causing the lung to collapse, either partially or completely. This can be life-threatening and is usually diagnosed through chest X-rays. The challenge is to predict the presence of pneumothorax in chest X-ray images and, if present, accurately segment the affected portion of the lungs.

## Real-world Business Constraints:

- Low latency is crucial ⏱️.
- Misclassification or missegmentation can have high costs due to the sensitivity of medical data.

## Mapping as a Deep Learning Problem:

The problem is mapped as a Semantic Image Segmentation task using DICOM format images with run-length-encoded (RLE) masks indicating instances of pneumothorax.

## Data Set Analysis:

- Data Source: [SIIM-ACR Pneumothorax Segmentation Kaggle Competition](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview)
- Files: train-rle.csv, stage_2_sample_submission.csv (test_data), train_images, test_images.
- Total File Size: 4GB
- Total Records: 12,954 (train_data), 3,204 (test_data)

## Performance Metrics:

- Dice Coefficient (Intersection Over Union/IOU) 🎲
- Combo Loss (Binary Cross Entropy + Dice Loss/F1 loss) 📉
- Confusion Matrix 📊
- Classification Metrics: AUC, Precision, Recall 📈

## Solution Overview:

### A) Image Classification using CheXNet:

- Utilizing transfer learning with CheXNet, a pre-trained 121-layer DenseNet model for chest X-rays.
- Classifying images into 'No Pneumothorax Present' (Label: 0) or 'Pneumothorax Present' (Label: 1).

### B) Segmentation Models:

- Applying different segmentation models (Unet, Nested Unet, Double Unet with pre-trained VGG19 as the backbone encoder).
- Predicting masks for the affected areas.

### C) End Pipeline:

- Combining image classification and segmentation to predict the label (0/1) and generate masks.


### Architecture diagram:

![Architecture Diagram](https://github.com/harishnandhan02/Pneumothorax-segmentation/blob/main/Output/Architecture%20Diagram.png)

## Classification Metrics:

![Classification Metrics](https://user-images.githubusercontent.com/56768652/112884926-bac42e00-90ed-11eb-84f4-17b9424f8d7c.JPG)

![Classification ROC](https://user-images.githubusercontent.com/56768652/112884975-ca437700-90ed-11eb-86b7-a6535da10f6e.JPG)

![Classification CV CM](https://user-images.githubusercontent.com/56768652/112884991-cfa0c180-90ed-11eb-86ba-1483ff72cacd.JPG)

![Classification Average Precision Curve](https://user-images.githubusercontent.com/56768652/112885011-d62f3900-90ed-11eb-94db-49188c926daf.JPG)

## Segmentation Metrics:

### Nested Unet:

![Segment_Unet++](https://user-images.githubusercontent.com/56768652/112885100-f232da80-90ed-11eb-8bd9-ee7380733d54.JPG)

### Weighted Nested Unet:

![Segmentation Metrics](https://user-images.githubusercontent.com/56768652/113031343-3684b000-91ac-11eb-9951-46a4e3fe2503.JPG)

## Final Pipeline Output Images:

### Positive Predictions:

![Final_Inference_Positive](https://user-images.githubusercontent.com/56768652/113337367-4f749900-9345-11eb-8b8a-b8088f3630a2.JPG)

![Final_Inference_Positive_2](https://user-images.githubusercontent.com/56768652/113337385-56031080-9345-11eb-84ce-0713d4039dea.JPG)

![Final_Inference_Positive_3](https://user-images.githubusercontent.com/56768652/113337684-c14ce280-9345-11eb-9e7b-7e91a96375e4.JPG)

![Final_Inference_Positive_4](https://user-images.githubusercontent.com/56768652/113337695-c447d300-9345-11eb-8cc5-6bdb104e1cd3.JPG)

### Negative Predictions:

![Final_Inference_Negative](https://user-images.githubusercontent.com/56768652/113337752-d590df80-9345-11eb-9f44-d02f47df30e7.JPG)

### Failure Case:

![Failure Cases](https://user-images.githubusercontent.com/56768652/113337777-dd508400-9345-11eb-893d-7535ef148490.JPG)

## Output Screenshots after Deployment with Streamlit:
![Pneumothorax_Output1](
https://github.com/harishnandhan02/Pneumothorax-segmentation/blob/main/Output/Pneumothorax_Output1.png)
![NoPneumothorax_Output2](https://github.com/harishnandhan02/Pneumothorax-segmentation/blob/main/Output/NoPneumothorax_Output2.png)
