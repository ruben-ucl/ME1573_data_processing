# Machine Learning Model Components: Lecture Notes
## Deep Learning for Photodiode Signal Classification

*Course: Advanced Machine Learning*  
*Topic: Convolutional Neural Networks for Time Series Classification*  
*Date: August 2025*

---

## Table of Contents
1. [Introduction](#introduction)
2. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
3. [Dense (Fully Connected) Layers](#dense-fully-connected-layers)
4. [Activation Functions](#activation-functions)
5. [Regularization Techniques](#regularization-techniques)
6. [Optimization](#optimization)
7. [Training Strategies](#training-strategies)
8. [Performance Evaluation](#performance-evaluation)
9. [Model Architecture Summary](#model-architecture-summary)
10. [Further Reading](#further-reading)

---

## Introduction

This document provides a comprehensive overview of the machine learning components used in the photodiode signal classifier implemented in this project. The model is a **Convolutional Neural Network (CNN)** designed for **binary classification** of time-series data from photodiode measurements during laser powder bed fusion (LPBF) manufacturing.

### Problem Context
- **Input**: 1D time series signals from photodiode sensors (100 time steps)
- **Output**: Binary classification (0: stable process, 1: unstable process)
- **Domain**: Additive Manufacturing Quality Control

---

## Convolutional Neural Networks (CNNs)

### Overview
Convolutional Neural Networks are deep learning architectures particularly effective for processing grid-like data such as images or time series. Originally developed for computer vision, CNNs have proven highly effective for 1D signal processing tasks.

![CNN Architecture](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)
*Figure 1: Typical CNN Architecture (Source: Wikipedia)*

### Key Components in Our Model

#### 1. **Convolutional Layers (Conv1D)**
```python
# Example from our model
Conv1D(filters=16, kernel_size=3, activation='relu')
Conv1D(filters=32, kernel_size=3, activation='relu') 
Conv1D(filters=64, kernel_size=3, activation='relu')
```

**Purpose**: Extract local features from the input signal
- **Filters**: Number of feature detectors (16 → 32 → 64 creates hierarchical feature extraction)
- **Kernel Size**: Width of the sliding window (3 time steps in our case)
- **Stride**: Step size (default=1, moves one time step at a time)

**Mathematical Operation**:
```
output[i] = σ(Σ(input[i+j] × kernel[j]) + bias)
```

**Why This Works**:
- Detects local patterns in time series data (spikes, trends, oscillations)
- Translation invariant (same pattern detected regardless of position)
- Parameter sharing reduces overfitting

**Further Reading**: [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/)

#### 2. **Max Pooling Layers**
```python
MaxPooling1D(pool_size=2)
```

![Max Pooling](https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png)
*Figure 2: Max Pooling Operation (Source: Wikipedia)*

**Purpose**: 
- **Downsampling**: Reduces sequence length while preserving important features
- **Translation Invariance**: Small shifts in input don't affect output
- **Computational Efficiency**: Fewer parameters in subsequent layers

**Operation**: Takes maximum value in each pool_size window

#### 3. **Batch Normalization**
```python
BatchNormalization()
```

**Purpose**: Normalizes inputs to each layer during training
- **Internal Covariate Shift**: Addresses changing input distributions
- **Faster Training**: Allows higher learning rates
- **Regularization Effect**: Reduces overfitting

**Mathematical Operation**:
```
BN(x) = γ × (x - μ) / √(σ² + ε) + β
```
Where μ and σ are batch statistics, γ and β are learned parameters.

**Paper**: [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)

---

## Dense (Fully Connected) Layers

### Overview
Dense layers perform traditional neural network operations where every input is connected to every output.

![Dense Layer](https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg)
*Figure 3: Fully Connected Neural Network (Source: Wikipedia)*

### In Our Model
```python
Dense(128, activation='relu')  # First dense layer
Dropout(0.3)                   # Regularization
Dense(64, activation='relu')   # Second dense layer  
Dropout(0.2)                   # More regularization
Dense(1, activation='sigmoid') # Output layer
```

**Purpose**:
- **Feature Combination**: Combines features extracted by convolutional layers
- **Non-linear Mapping**: Creates complex decision boundaries
- **Final Classification**: Maps to output classes

**Mathematical Operation**:
```
output = σ(W × input + b)
```
Where W is the weight matrix, b is bias vector, σ is activation function.

---

## Activation Functions

### ReLU (Rectified Linear Unit)
```python
activation='relu'
```

![ReLU Function](https://upload.wikimedia.org/wikipedia/commons/f/fe/Activation_rectified_linear.svg)
*Figure 4: ReLU Activation Function (Source: Wikipedia)*

**Mathematical Definition**:
```
ReLU(x) = max(0, x)
```

**Advantages**:
- **Computational Efficiency**: Simple max operation
- **Gradient Flow**: No vanishing gradient for positive inputs
- **Sparsity**: Many neurons output exactly zero

**Why We Use It**: Prevents vanishing gradient problem in deep networks

### Sigmoid
```python
activation='sigmoid'  # Output layer only
```

![Sigmoid Function](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)
*Figure 5: Sigmoid Activation Function (Source: Wikipedia)*

**Mathematical Definition**:
```
σ(x) = 1 / (1 + e^(-x))
```

**Use Case**: Binary classification output (gives probability between 0 and 1)

**Further Reading**: [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

---

## Regularization Techniques

### 1. Dropout
```python
Dropout(0.2)  # Randomly sets 20% of inputs to 0 during training
Dropout(0.3)  # Randomly sets 30% of inputs to 0 during training
```

![Dropout](https://jmlr.org/papers/volume15/srivastava14a/dropout.png)
*Figure 6: Dropout During Training (Source: JMLR)*

**Purpose**: Prevents overfitting by randomly "dropping out" neurons during training

**How It Works**:
- During training: Randomly set neurons to zero with probability p
- During inference: Use all neurons but scale outputs by (1-p)

**Benefits**:
- Forces network to not rely on specific neurons
- Creates ensemble effect
- Improves generalization

**Original Paper**: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)

### 2. L2 Regularization (Weight Decay)
```python
kernel_regularizer=l2(0.001)  # L2 penalty coefficient
```

**Purpose**: Prevents overfitting by penalizing large weights

**Mathematical Addition to Loss**:
```
Loss_total = Loss_original + λ × Σ(w²)
```

**Effect**: 
- Keeps weights small
- Creates smoother decision boundaries
- Improves generalization

### 3. Early Stopping
```python
EarlyStopping(patience=10, monitor='val_loss')
```

**Purpose**: Stop training when validation performance stops improving

**How It Works**:
- Monitor validation loss during training
- Stop if no improvement for 'patience' epochs
- Prevents overfitting to training data

---

## Optimization

### Adam Optimizer
```python
optimizer='adam'
```

**Full Name**: Adaptive Moment Estimation

**Key Features**:
- **Adaptive Learning Rates**: Different learning rate for each parameter
- **Momentum**: Uses exponential moving average of gradients
- **Bias Correction**: Corrects bias in moment estimates

**Mathematical Update**:
```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
```

**Why We Use It**: 
- Works well with sparse gradients
- Requires minimal hyperparameter tuning
- Generally robust across different problems

**Paper**: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

### Learning Rate Scheduling
```python
ReduceLROnPlateau(patience=5, factor=0.5)
```

**Purpose**: Reduce learning rate when training plateaus
- **Patience**: Wait 5 epochs without improvement
- **Factor**: Multiply learning rate by 0.5

**Benefits**: 
- Helps fine-tune model in later stages
- Prevents oscillation around minimum
- Improves final convergence

---

## Training Strategies

### 1. K-Fold Cross Validation
```python
k_folds = 5  # Typical value used in our model
```

![K-Fold CV](https://upload.wikimedia.org/wikipedia/commons/1/1c/K-fold_cross_validation_EN.jpg)
*Figure 7: K-Fold Cross Validation (Source: Wikipedia)*

**Purpose**: Robust model evaluation and hyperparameter tuning

**Process**:
1. Split data into K equal folds
2. Train on K-1 folds, validate on 1 fold
3. Repeat K times with different validation fold
4. Average results across all folds

**Benefits**:
- More reliable performance estimates
- Better use of available data
- Reduces variance in performance metrics

### 2. Data Augmentation
```python
# Time series specific augmentation
- Time shifting: shift_range=5
- Stretching: stretch_probability=0.3  
- Noise addition: noise_probability=0.5
- Amplitude scaling: amplitude_scale_probability=0.5
```

**Purpose**: Artificially increase training data variety

**Techniques Used**:
- **Time Shifting**: Shift signal in time domain
- **Amplitude Scaling**: Scale signal magnitude
- **Noise Addition**: Add random noise
- **Time Stretching**: Change signal duration

**Benefits**:
- Improves model robustness
- Reduces overfitting
- Better generalization to new data

---

## Performance Evaluation

### 1. Loss Function: Binary Crossentropy
```python
loss='binary_crossentropy'
```

**Mathematical Definition**:
```
BCE = -1/N × Σ[y × log(ŷ) + (1-y) × log(1-ŷ)]
```

**Why We Use It**: 
- Standard for binary classification
- Probabilistic interpretation
- Smooth gradients for optimization

### 2. Metrics

#### Accuracy
```python
metrics=['accuracy']
```
**Definition**: (TP + TN) / (TP + TN + FP + FN)
**Interpretation**: Percentage of correct predictions

#### Validation Split
```python
validation_split=0.2  # 20% of data for validation
```
**Purpose**: Monitor overfitting during training

### 3. Class Weights
```python
use_class_weights=True  # Handles imbalanced datasets
```

**Purpose**: Address class imbalance in training data
**Method**: Automatically calculate inverse frequency weights

**Benefits**:
- Prevents bias toward majority class
- Improves minority class detection
- Better overall performance on imbalanced data

---

## Model Architecture Summary

### Our Complete Architecture
```
Input Layer (100 time steps)
    ↓
Conv1D(16) → BatchNorm → ReLU → MaxPool(2)
    ↓
Conv1D(32) → BatchNorm → ReLU → MaxPool(2)  
    ↓
Conv1D(64) → BatchNorm → ReLU → MaxPool(2)
    ↓
Flatten
    ↓
Dense(128) → ReLU → Dropout(0.3)
    ↓  
Dense(64) → ReLU → Dropout(0.2)
    ↓
Dense(1) → Sigmoid
    ↓
Output (Binary Classification)
```

### Design Rationale

1. **Progressive Feature Extraction**: 16→32→64 filters capture increasingly complex patterns
2. **Dimensionality Reduction**: MaxPooling reduces computational load
3. **Regularization at Multiple Levels**: BatchNorm, Dropout, L2 prevent overfitting
4. **Hierarchical Learning**: CNN features → Dense combination → Final classification

### Parameter Count Analysis
- **Convolutional Layers**: ~50K parameters (feature extraction)
- **Dense Layers**: ~10K parameters (classification)
- **Total**: ~60K parameters (manageable for our dataset size)

---

## Further Reading

### Foundational Papers
1. [LeCun et al. - Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
2. [Krizhevsky et al. - ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
3. [Goodfellow, Bengio & Courville - Deep Learning Book](https://www.deeplearningbook.org/)

### Time Series with CNNs
1. [Wang et al. - Time Series Classification from Scratch with Deep Neural Networks](https://arxiv.org/abs/1611.06455)
2. [Fawaz et al. - Deep Learning for Time Series Classification: A Review](https://arxiv.org/abs/1809.04356)

### Regularization and Training
1. [Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)
2. [Batch Normalization](https://arxiv.org/abs/1502.03167)
3. [Adam Optimizer](https://arxiv.org/abs/1412.6980)

### Online Resources
1. [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
2. [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
3. [Keras Documentation](https://keras.io/)
4. [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

---

## Glossary

**Activation Function**: Non-linear function applied to neuron outputs  
**Backpropagation**: Algorithm for computing gradients in neural networks  
**Batch Normalization**: Technique to normalize layer inputs during training  
**Convolution**: Mathematical operation for feature extraction  
**Cross-validation**: Method for robust model evaluation  
**Dropout**: Regularization technique that randomly zeros neurons  
**Feature Map**: Output of convolutional layer showing detected features  
**Gradient Descent**: Optimization algorithm for minimizing loss  
**Hyperparameter**: Configuration setting not learned during training  
**Kernel**: Weights of convolutional filter  
**Loss Function**: Measure of prediction error to minimize  
**Overfitting**: When model memorizes training data but fails to generalize  
**Regularization**: Techniques to prevent overfitting  
**Stride**: Step size of convolution operation  
**Vanishing Gradient**: Problem where gradients become too small in deep networks

---

*End of Lecture Notes*  
*For questions or clarifications, refer to the implementation in the accompanying Python files.*