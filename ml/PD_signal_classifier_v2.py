
#############
### Setup ###
#############

import numpy as np
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


###########################
###  Model Architecture ###
###########################

def create_classifier_model(input_shape=(100, 2), num_classes=1):
    """
    Creates a CNN model adapted for small datasets of multi-channel 1D signals.
    
    Args:
        input_shape: Tuple specifying input dimensions (sequence_length, channels)
        num_classes: Number of signal classes to classify
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First convolutional block with strong regularization
    model.add(Conv1D(filters=16, kernel_size=7, 
                     activation='relu', 
                     padding='same',
                     kernel_regularizer=l2(0.001),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    # Second convolutional block
    model.add(Conv1D(filters=32, kernel_size=5, 
                     activation='relu', 
                     padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu', 
                   kernel_regularizer=l2(0.001),
                   activity_regularizer=l1(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(num_classes, activation='sigmoid' if num_classes==1 else 'softmax'))
    
    # Compile model with Adam optimizer
    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if num_classes==1 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Define callbacks for training
def get_training_callbacks():
    """
    Returns a list of callbacks for training the model
    """
    callbacks = [
        # Stop training when validation loss stops improving
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        
        # Save best model during training
        ModelCheckpoint('best_radiation_model.h5', save_best_only=True, monitor='val_loss')
    ]
    return callbacks

# Example training code
def train_model(X_train, y_train, X_val, y_val, batch_size=16, epochs=200):
    """
    Train the model with data augmentation
    """
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]  # Assuming one-hot encoded labels
    
    model = create_classifier_model(input_shape, num_classes)
    callbacks = get_training_callbacks()
    
    # Create data augmentation generator
    train_gen = data_augmentation_generator(X_train, y_train, batch_size)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


#########################
### Data augmentation ###
#########################

def data_augmentation_generator(X, y, batch_size=32):
    """
    Generate batches of augmented data
    
    Args:
        X: Input data with shape (samples, sequence_length, channels)
        y: Target labels (one-hot encoded)
        batch_size: Number of samples per batch
        
    Yields:
        Batches of augmented data
    """
    num_samples = X.shape[0]
    
    while True:
        # Shuffle the data
        indices = np.random.permutation(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X[batch_indices].copy()
            batch_y = y[batch_indices].copy()
            
            # Apply augmentation
            for j in range(len(batch_X)):
                # Apply random augmentation with certain probability
                if np.random.random() < 0.7:
                    # Apply one or more augmentations
                    augmentation_choice = np.random.random()
                    
                    if augmentation_choice < 0.25:
                        # Add calibrated noise
                        noise_level = np.random.uniform(0.01, 0.1)
                        batch_X[j] += np.random.normal(0, noise_level, size=batch_X[j].shape)
                        
                    elif augmentation_choice < 0.5:
                        # Time shift
                        shift = np.random.randint(-10, 10)
                        batch_X[j] = np.roll(batch_X[j], shift, axis=0)
                        
                    elif augmentation_choice < 0.75:
                        # Scale amplitude
                        scale_factor = np.random.uniform(0.8, 1.2)
                        batch_X[j] *= scale_factor
                        
                    else:
                        # Mix channels slightly
                        mix_factor = np.random.uniform(0, 0.1)
                        channel_mix = batch_X[j].copy()
                        # Mix channel 0 into channel 1 and vice versa
                        batch_X[j, :, 0] = (1 - mix_factor) * batch_X[j, :, 0] + mix_factor * batch_X[j, :, 1]
                        batch_X[j, :, 1] = (1 - mix_factor) * batch_X[j, :, 1] + mix_factor * batch_X[j, :, 0]
            
            yield batch_X, batch_y


########################
### Cross-validation ###
########################

def cross_validate_model(X, y, n_splits=5, batch_size=16, epochs=200):
    """
    Perform stratified k-fold cross-validation
    
    Args:
        X: Input data with shape (samples, sequence_length, channels)
        y: Target labels (integers, not one-hot encoded)
        n_splits: Number of folds
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        
    Returns:
        Dictionary with cross-validation results
    """
    # Convert labels to one-hot encoding for each fold to ensure proper stratification
    y_indices = np.argmax(y, axis=1) if len(y.shape) > 1 else y
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results
    cv_scores = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    fold = 1
    # Iterate over folds
    for train_idx, val_idx in skf.split(X, y_indices):
        print(f"Training fold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and train model
        model, history = train_model(X_train, y_train, X_val, y_val, 
                                    batch_size=batch_size, epochs=epochs)
        
        # Get best validation scores
        best_epoch_idx = np.argmin(history.history['val_loss'])
        cv_scores['accuracy'].append(history.history['accuracy'][best_epoch_idx])
        cv_scores['val_accuracy'].append(history.history['val_accuracy'][best_epoch_idx])
        cv_scores['loss'].append(history.history['loss'][best_epoch_idx])
        cv_scores['val_loss'].append(history.history['val_loss'][best_epoch_idx])
        
        fold += 1
    
    # Calculate mean and std of scores
    for metric in cv_scores:
        cv_scores[f'{metric}_mean'] = np.mean(cv_scores[metric])
        cv_scores[f'{metric}_std'] = np.std(cv_scores[metric])
    
    return cv_scores


##########################
### Feature extraction ###
##########################

def extract_features(model, X):
    """
    Extract learned features from the penultimate layer of the model
    
    Args:
        model: Trained Keras model
        X: Input data
        
    Returns:
        Extracted features
    """
    # Create a new model that outputs features from the layer before the final classification layer
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-2].output
    )
    
    # Extract features
    features = feature_extractor.predict(X)
    
    return features

def train_svm_on_features(features_train, y_train, features_val, y_val):
    """
    Train SVM classifier on extracted features
    
    Args:
        features_train: Extracted features from training data
        y_train: Training labels
        features_val: Extracted features from validation data
        y_val: Validation labels
        
    Returns:
        Trained SVM model and accuracy
    """
    
    # Convert one-hot encoded labels to class indices if needed
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)
    if len(y_val.shape) > 1:
        y_val = np.argmax(y_val, axis=1)
    
    # Train SVM
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(features_train, y_train)
    
    # Evaluate
    y_pred = svm.predict(features_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return svm, accuracy