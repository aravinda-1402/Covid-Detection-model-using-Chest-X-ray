# Covid-Detection-model-using-Chest-X-ray
![covid](https://user-images.githubusercontent.com/71303848/131960499-9a9341d1-64c9-4cb6-9b5c-773dd144dedc.jpeg)


<b>Problem Statement:</b> Corona - COVID19 virus affects healthy people's respiratory systems, and chest X-Ray is one of the most significant imaging modalities for detecting the virus.

The objective of this project is to develop a Deep Learning Model to identify the X-Rays of healthy vs. Pneumonia (Corona) afflicted patients using the Chest X-Ray dataset, and use this model to power the AI application to test the Corona Virus in a faster phase.

<b>Dataset Used:</b> The dataset is a collection of Chest X-Ray images of people. It contains images of people who are healthy, those who are tested positive for COVID-19 or other viral and bacterial pneumonias such as SARS (Severe Acute Respiratory Syndrome), Streptococcus and ARDS (Acute Respiratory Distress Syndrome).<br>

<b>Dataset Link:</b>  https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset<br>

There are 2 files present. One of them is the data and the other is the metadata. A sample of the data file is shown in Figure 1 and a sample of the metadata is shown in Figure 2.<br>

<b>Figure 1. Data file sample</b>

|          |     Unnamed:0    |      X_ray_image_name    |      Label    |     Dataset_type    |     Label_2_Virus_category    |     Label_1_Virus_category    |
|:--------:|:----------------:|:------------------------:|:-------------:|:-------------------:|:-----------------------------:|:-----------------------------:|
|     0    |         0        |     IM-0128-0001.jpeg    |     Normal    |         TRAIN       |               NaN             |               NaN             |
|     1    |         1        |     IM-0127-0001.jpeg    |     Normal    |         TRAIN       |               NaN             |               NaN             |
|     2    |         2        |     IM-0125-0001.jpeg    |     Normal    |         TRAIN       |               NaN             |               NaN             |
|     3    |         3        |     IM-0122-0001.jpeg    |     Normal    |         TRAIN       |               NaN             |               NaN             |

<br>

<b>Figure 2. Metadata file sample</b>


|      Unnamed: 0     |             Label     |      Label_1_Virus_category     |      Label_2_Virus_category     |      Image_Count     |
|:-------------------:|:---------------------:|:-------------------------------:|:-------------------------------:|:--------------------:|
|           0         |         Normal        |                NaN              |                NaN              |          1576        |
|           1         |        Pneumonia      |          Stress-Smoking         |               ARDS              |           2          |
|           2         |        Pneumonia      |               Virus             |                NaN              |          1493        |
|           3         |        Pneumonia      |               Virus             |             COVID-19            |           58         |
<br>
<b>Data Cleaning and preparation:</b>

* As we can see from the data file sample, it has 2 categories in the ‘Dataset_type’ column: TRAIN and TEST. So, we separate the train and test images by filtering using the labels.

* Next, we fill all the places with NaN (Not a Number) with ‘NA’ string and we also append ‘Label_2_Virus_category’ column with the ‘Label’ column.

* We then check for all the label types like ‘Normal/NA’, ‘Pneumonia/NA’ and ‘Pneumonia/COVID-19’ in the train and test sets if they are present or not. We notice that ‘Pneumonia/COVID-19’ is not present in the test set. This is going to affect the prediction accuracy of the model. So, we take the last 600 examples of the train set and append it to the test set so that the data distribution with all the various labels is uniform and the overall model’s accuracy is good.

* We then perform image data-augmentation on the train set to produce and add more images into the train set with varied orientations and other properties like zoom and brightness to improve the accuracy of the model.<br>

<br>
<b>Algorithm Used:</b> As we have to classify the data into 3 categories of outputs ‘Normal/NA’, ‘Pneumonia/NA’ and ‘Pneumonia/COVID-19’, I have chosen CNN (Convolutional Neural Network). 

<br>As we can see from Figure 3,  we have the input shape as (256,256,3) and various Convolutional layers with a different number of filters and padding set to ‘same’. With padding set to ‘same’, the image dimension remains the same after every convolutional layer which gives the model more scope to learn features along the edges of the image. <br>

I have used L2 regularization with value 1e-4 which is a technique used for tuning the function by adding an additional penalty term in the error function. The additional term controls the excessively fluctuating function such that the coefficients don't take extreme values. We have the activation function as ReLU and we also have ‘Batch Normalization’ after every convolutional layer so that the model doesn’t overfit and it also reduces the total number of epochs to train the model.<br>   

In the prefinal layer, we flatten the image into a feature vector and feed it to a Dense layer with 3 outputs which correspond to the 3 outputs ‘Normal/NA’, ‘Pneumonia/NA’ and ‘Pneumonia/COVID-19’ and use ‘softmax’ activation function which gives the probability to which the input image may belong among the 3 classes.<br>

For optimizing the model, we use Adam optimizer with 0.0004 learning rate, ‘categorical cross entropy’ as loss function because it is a multi-class classification model.<br>

We then train the model for 40 epochs with 3740/32 steps per epoch. The model yields a training accuracy of 94.76% and validation accuracy of 77.54%. The validation accuracy is a little low and can be improved. So, we perform 2 more epochs after changing the learning rate from 0.0004 to 0.0002 and we end up with a training accuracy of 95.28% and validation accuracy of 89.52%.<br>
Figure 4 and Figure 5 represent ‘Training accuracy and validation accuracy’ and ‘Training loss and validation loss’ respectively.<br>

### Figure 4 

![Training accuracy and validation accuracy](https://user-images.githubusercontent.com/71303848/131959908-2e305082-74fb-4f38-a5c4-106ef991fd99.PNG)
<br>

### Figure 5

![Training loss and validation loss](https://user-images.githubusercontent.com/71303848/131959986-c75a75de-840d-44a8-ab6a-472fd42e1e8d.PNG)

<b>Software Packages Used:</b>
*	Numpy 1.21.1
*	Pandas v1.3.1
*	Matplotlib 3.3.4
*	TensorFlow 2.0
*	Keras 2.3.0

<b>Industrial scope and advantage of this project:</b> Methods for detecting and classifying human illnesses from medical pictures that are automated using novel Machine Learning and Deep Learning Algorithms enable the doctor in driving the consultation in a better way, reducing the time it takes to diagnose the Corona Virus. This would also give physicians an edge and allow them to act with more confidence while they wait for the analysis of a radiologist by having a digital second opinion confirm their assessment of a patient's condition. Also, these tools can provide quantitative scores to consider and use in studies.
