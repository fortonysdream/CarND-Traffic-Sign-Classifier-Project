# *Traffic Sign Recognition*

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

- - - -

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 34799

#### 2. Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

The 3 image files are in the directory

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
I normalized the data because I thought it will help the performance of the network.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		  | 32x32x3 RGB image   				        | 
| Convolution 5x5   | 1x1 stride, same padding, outputs 28x28x6 |
| RELU			  |										 |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6 				 |
| Convolution 5x5	  | 1x1 stride, same padding, outputs 10x10x16|
| RELU                    |                                                                        |
| Max pooling          | 2x2 stride,  outputs 5x5x16                           |
| Flatten                   | 400                                                                 |
| Fully connected    | 200        							          |
| RELU                     |                                                                        |
| Fully connected    | 120                                                                  |
| RELU                    |                                                                         |
| Fully connected    | 84                                                                    |
| RELU                    |                                                                         |
| Fully connected    | 43                                                                    |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
To train the model, I used an AdamOptimizer with learning rate 0.001, batch size of 128, 40 epochs

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
My final model results were:
* training set accuracy of 0.984
* validation set accuracy of 0.942
* test set accuracy of 0.931

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
the original LeNet
* What were some problems with the initial architecture?
underfit
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Used LeNet1, with one more convolution layer
* Which parameters were tuned? How were they adjusted and why?
filter sizes and full connected layer node numbers.
They can help increasing training set performance

### Test a Model on New Images

#### For some reasons, I saved the session successfully after training, but when I tried to restore it, the parameters are present in the form before training. Thus I was not able to apply the correct model on the new images successfully. I was wondering what is the problem with my codes. Thanks!