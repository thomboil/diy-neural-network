# DIY-neural-network
The goal of this project is to build a neural network in order to understand how it works.

The objective of the neural network if to classify handwritten digits so it can tell you what digit it is.

![image](https://user-images.githubusercontent.com/36003383/179868263-b5959e8a-8e71-4840-bbf8-1d2166a3c4bf.png)

Digits data source: https://www.kaggle.com/competitions/digit-recognizer/data
Downloaded in test.csv

## The Math
### Images:
- Image size 28x28 pixels (784 pixels total)
- Each pixel value [0 ... 255] (0: black, 255: white)

We can then put this in an matrix:
- Columns are exemples in other words training samples
- Rows are the pixels (784 long)

Transpose this matrix:

![image](https://user-images.githubusercontent.com/36003383/179868210-33fa52ee-5ea8-4e15-a50d-5d0297e89925.png)

We now have a matrix pixels by training samples

### Neural network:
What will be able to give a good prediction of the digit is our neural network. This simple neural network only has 2 layers.
- 0th layer AKA input layer: 780 nodes (every pixel maps to a node)
- 1st layer AKA hidden layer: 10 units corresponding to each digit (0, 1, 2, ...9)
- 2nd layer AKA output layer:

![image](https://user-images.githubusercontent.com/36003383/179869194-2df385cf-09c6-41e3-abd3-1c1b39d3936c.png)

## Training:

#### Input layer
Input layer  a[0]  will have 784 units corresponding to the 784 pixels in each 28x28 input image

#### Hidden layer
A hidden layer  a[1]  will have 10 units with ReLU activation (in order for it to not just be a fancy linear regression with the input layer, we need to apply an activation function).

Activation function exemples:
- Hyperbolic tangent function (Tanh) and sigmoid functions.
- Rectified linear unit (RelU): RelU(x) is defined as if x>0 then x=x and if x<=0 then x=0

It adds a layer of complexity rather than just a linear model. It makes it more powerfull.

#### Output layer
Finally output layer  a[2]  will have 10 units corresponding to the ten digit classes with softmax activation.

![image](https://user-images.githubusercontent.com/36003383/179871316-63443b81-1dc8-47a1-be41-63262ebd086a.png)

### 1. Forward propogation: 
Take the image and run it through the network and compute what your output is going to be.

![image](https://user-images.githubusercontent.com/36003383/179872264-9cffdc0e-6087-46cc-9f66-a2b0eacfc2c9.png)
![image](https://user-images.githubusercontent.com/36003383/179872432-2854ae50-edb1-4640-a485-c71357da1ffa.png)


### 2. Backward propagation
Evaluates the weights and biasses of the predictions to optimize these weights and biasses. We will mesure how much the prediction deviated for the actual result. Calculate how much these weights and biasses contributed to the the prediction.

![image](https://user-images.githubusercontent.com/36003383/179872360-557bae7a-bff9-4036-8136-c52ff185371e.png)
![image](https://user-images.githubusercontent.com/36003383/179872329-7adfc79b-ab66-4516-a466-3d4a59d6b072.png)

### 3. Update our parameters
![image](https://user-images.githubusercontent.com/36003383/179878624-3f199c84-2926-4421-b3c9-a6cbf1e6449d.png)
![image](https://user-images.githubusercontent.com/36003383/179878667-b02d974c-0772-487d-a310-db6515ca84b5.png)

### Loop again
Once you go through the 3 steps you loop again and again to better your results.

https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang
