# Machine_Learning_Project2

# Neural Network Library & Taxi Trip Duration Prediction

**Project Overview**

This project involved designing and implementing a custom Neural Network Library from scratch using Python and NumPy. The primary goal was to build a modular neural network framework capable of handling forward and backward propagation, custom activations, and loss computations, and then apply it to two problems:

1. The XOR classification problem (a foundational nonlinear problem in ML).
2. The NYC Taxi Trip Duration Prediction task (a regression problem with real-world data).

**Phase 1: Neural Network Library Development**

Objective: Develop a reusable and well-structured neural network library that supports flexible architectures with any number of layers and activation functions.
•	Designed the Base Layer Architecture: Created a parent class `Layer` defining abstract methods `forward()` and `backward()` for polymorphism.
•	Implemented Linear (Fully Connected) Layer with forward (xW^T + b) and backward gradient propagation.
•	Developed Activation Layers: Sigmoid and ReLU with stored activations and gradient calculations.
•	Created Binary Cross-Entropy Loss for classification performance evaluation.
•	Built Sequential Class to manage stacked layers and sequential propagation.
•	Implemented Model Persistence with save/load weight functionality.

**Phase 2: XOR Problem Implementation**

Objective: Train a simple neural network to solve the XOR classification problem using the developed library.
•	Constructed a network with 2 input, 2 hidden, and 1 output node.
•	Trained using Sigmoid and Tanh activations separately.
•	Compared convergence and observed Tanh performed better.
•	Saved trained weights for reproducibility.

**Phase 3: NYC Taxi Trip Duration Prediction**
Objective: Apply the custom neural network library to predict taxi trip durations using the modified NYC Taxi dataset.
•	Loaded and preprocessed data from `nyc_taxi_data.npy` using NumPy.
•	Performed feature cleaning, normalization, and transformation.
•	Engineered features such as datetime breakdown and categorical encoding.
•	Designed multiple architectures: Small (64x32), Medium (128x64x32), Large (256x128x64).
•	Used validation loss for model selection with early stopping.
•	Evaluated using Mean Absolute Error (MAE) and plotted loss curves.

**Results Summary**

Model	Validation Loss	Test MAE (seconds)
Small_64x32	0.2217	361.47
Medium_128x64x32	0.2124 (Best)	351.05
Large_256x128x64	0.2193	355.87

**Phase 4: Visualization & Analysis**
Plotted loss curves for training and validation across different models. Created performance comparison charts and documented the impact of feature scaling and architecture depth on generalization.

**Key Learnings**
•	Building a neural network from scratch deepened understanding of propagation mechanics.
•	Experimentation with activations showed how nonlinearity affects convergence.
•	Model complexity doesn’t always improve results; medium-depth models generalized best.
•	Feature normalization and early stopping were crucial to avoid overfitting.
