'''
Code by Dr Wei Li and Rubén Lambert-Garcia
20th February 2025
'''


# import libaries as needed

import os, pywt, glob, sys, keras, cv2
from pathlib import Path
import numpy as np
from contextlib import redirect_stdout
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import normalize
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colormaps
from keras.models import load_model

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths

train_new = True # 'train' or 'evaluate'
evaluate = True # Run model evaluation, bool
draw_model_architecture = False # bool
display_cam = True # bool
save_cam = False # bool
show_confusion = True # bool

# Set image path and size

image_directory = Path(r'E:\AlSi10Mg single layer ffc\CWT_ML_training_data\CWT_labelled_cw_powder_in-situ_ss_double_sliced')
model_dir = Path('ml', 'models')
SIZE_X = 50 
SIZE_Y = 256
img_channels = 3
img_size = (SIZE_X, SIZE_Y)


def create_dataset():
    dataset = [] 
    label = []
    # Creat image and label lists
    # Change the image type

    NP_images = sorted(glob.glob(str(Path(image_directory, '0', '*.png'))))
    for i, image_path in enumerate(NP_images):
        image = get_img_array(image_path, img_size)[:, :, :img_channels]
        dataset.append(image)
        label.append(0)

    P_images = sorted(glob.glob(str(Path(image_directory, '1', '*.png'))))
    for i, image_path in enumerate(P_images):
        image = get_img_array(image_path, img_size)[:, :, :img_channels]
        dataset.append(image)
        label.append(1)
    
    # Record number of images and train/test split
    global N0
    N0 = label.count(0)
    global N1
    N1 = label.count(1)
    global TS
    TS = 0.2
    
    # Convert to np array
    dataset = np.array(dataset)
    label = np.array(label)

    # Split the data set as needed
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = TS, random_state = 2)

    #Data normalization (0,1) to help convergence
    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)
    
    return X_train, X_test, y_train, y_test

def visualize_cnn_architecture(model, output_file='ml\model_architecture.png'):
    """
    Visualize the CNN model architecture and save to a file.
    
    Args:
        model: Keras model to visualize
        output_file: Path to save the visualization
    """
    plot_model(
        model, 
        to_file=output_file, 
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=200,
        rankdir='TB'
    )
    
    # Optional: display the architecture
    plt.figure(figsize=(15, 10))
    img = mpimg.imread(output_file)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#Creat CNN stucture for binary classification
    
def train(X_train, X_test, y_train, y_test):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(SIZE_Y, SIZE_X, img_channels)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Call the function after model creation
    if draw_model_architecture == True:
        visualize_cnn_architecture(model)
    
    # Get latest model version in the folder and increment by 1
    v_num = get_latest_version(model_dir)
    v_num = f'{int(v_num)+1:03}'
    
    # Create output folder for supporting file
    if not os.path.exists(Path(model_dir, f'{v_num}_info')):
        os.makedirs(Path(model_dir, f'{v_num}_info'))
    
    #Use CSVlogger to record training process
    csv_logger = CSVLogger(Path(model_dir,
        f'{v_num}_info', 
        f'CWT_image_binary_classification_{v_num}_training_log.csv'),
        append=True)

    # Setting of model training (optimizer, learning rate, etc)

    # Learning rate
    learning_rate = 0.0001

    # Optimizer
    Adam_optimizer = Adam(learning_rate=learning_rate)

    # Model compile
    model.compile(optimizer= Adam_optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
                  
    # Model training
    history = model.fit(X_train, 
        y_train,
        batch_size = 8,
        verbose = 1,
        epochs = 36,
        validation_data=(X_test,y_test),
        shuffle = False,
        callbacks=[csv_logger])
    
    # Save model
    model.save(Path(model_dir, f'CWT_image_binary_classification_{v_num}.h5'))
    # Save model metadata
    with open(Path(model_dir, f'{v_num}_info', f'CWT_image_binary_classification_{v_num}_info.txt'), 'w') as f:
        l1 = f'Version: {v_num}\n'
        l2 = f'Image source: {str(image_directory)}\n'
        l3 = f'Image shape: ({SIZE_X}, {SIZE_Y}, {img_channels})\n'
        l4 = f'Number of images: {N0} \'0\' label, {N1} \'1\' label ({N0+N1} total)\n'
        l5 = f'Train/test split: {int((1-TS)*100)}/{int(TS*100)}\n\n'
        
        f.writelines([l1, l2, l3, l4, l5])
        with redirect_stdout(f):
            model.summary()
        
        f.write('\n\nNotes:')

def model_eval(model, X_test, y_test):
    print('Running model evaluation')
    # Evaluate accuracy
    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy = ", (acc * 100.0), "%")
    
    # Create folder for figures
    v_num = get_latest_version(model_dir)
    print('model version = ', v_num)
    model_version_folder = Path(model_dir, f'{v_num}_info')
    if not os.path.exists(model_version_folder):
        os.makedirs(model_version_folder)
    
    # Predict probabilities and convert to binary predictions    
    y_pred_proba = model.predict(X_test)
    # thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thresholds = [0.7]

    for i, th in enumerate(thresholds, start=1):
        y_pred = (y_pred_proba > th).astype(int).flatten()
        
        # Calculate accuracy metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)*100
        recall = metrics.recall_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        print(f'Threshold: {th}, ',
            f'Accuracy: {round(accuracy, 2)}%, ',
            f'Recall: {round(recall, 2)}, ',
            f'Precision: {round(precision, 2)}')
        
        # Create confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['NP','P'])
        
        # Create plot
        fig, ax = plt.subplots(1, 1)
        cm_display.plot(cmap='Blues', values_format='d', ax=ax)
        fig.suptitle(f'Confusion Matrix for CNN Classification\nThreshold={th}')
        if show_confusion:
            plt.show()
            plt.savefig(Path(model_version_folder, f'confusion_matrix_threshold_{int(th*100)}pc'))
        plt.close()
    
    # create dict to store mean cam heatmaps for each label
    cam_means = {}
    
    # Iterate through images creating and savig grad-CAM activation maps for each
    for label in ('0', '1'):
        print(f'Performing grad-CAM visualisation for \'{label}\' labelled images')
        if save_cam:
            output_folder = Path(model_version_folder, 'grad-CAM', label)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        else:
            output_folder = ''
        images = os.listdir(Path(image_directory, label))
        cams = np.zeros((len(images), img_size[1], img_size[0]))
        for i, image_name in enumerate(images):
            if (image_name[-4:] == '.png'):
                img_path = Path(image_directory, label, image_name)
                image = get_img_array(img_path, img_size)
                img_array = normalize(np.expand_dims(image, axis=0), axis=1)
                heatmap = make_gradcam_heatmap(img_array,
                    model,
                    last_conv_layer_name = "conv2d_5")
                cam = save_and_display_gradcam(img_path,
                    heatmap,
                    display = False,
                    output_path = Path(output_folder, f'{image_name[:-4]}_CAM.png'))
                cams[i] = cam
            print(f'Progress: {i}/{len(images)}', end='\r')
        cam_mean = np.mean(cams, axis=0)
        cam_means[label] = cam_mean
    
    # Calculate frequency values for y axis
    scales = np.logspace(1, 7, num=256, base=2, endpoint=True)
    freqs = []
    for i, s in enumerate(scales):
        # Calculate frequency in kHz based on sampling rate of 100 kHz (0.00001 s period)
        f = round(pywt.scale2frequency("cmor1.5-1.0", s) / 0.01, 5)
        freqs.append(f)
    print(f'{freqs[-1]} Hz - {freqs[0]} Hz')
    
    # Create figure comparing average heatmaps of '0' and '1' datasets
    fig, (ax1, ax2) =  plt.subplots(1, 2,
        sharey=True,
        layout='compressed',
        dpi=300,
        figsize=(3.15, 3.15))
    fig.suptitle('Grad-CAM heatmaps')
    ax1.imshow(cam_means['0'], cmap='jet', vmin=0, vmax=160)
    ytick_indices = [0, 27, 69, 112, 154, 197]
    ax1.set_yticks(ytick_indices)
    ax1.set_yticklabels([round(freqs[i]) for i in ytick_indices])
    ax1.set_ylabel('Frequency [kHz]')
    ax1.set_xticks([])
    pos = ax2.imshow(cam_means['1'], cmap='jet', vmin=0, vmax=160)
    ax2.set_xticks([])
    cbar = fig.colorbar(pos, ax = ax2)
    cbar.set_label('Mean activation')
    ax1.set_title('0')
    ax2.set_title('1')
    plt.show()
    plt.savefig(Path(model_version_folder, 'grad-CAM_means_by_label.png'))
    plt.close()
    
######################
# Grad-CAM functions #
######################

def get_img_array(img_path, size, bw=False):
    # Read RGB image to array
    image = cv2.imread(img_path)
    image = Image.fromarray(image)
    if bw:
        image = image.convert('L')
        image = image.convert('RGB')
    image = image.resize(size)
    image = np.array(image)
    return image
    
def resize_img_array(img, size, bw=False):
    img = keras.utils.array_to_img(img)
    img = img.resize((size[1], size[0]))
    img = keras.utils.img_to_array(img)
    return img
    
def make_gradcam_heatmap(img_array, model, last_conv_layer_name,
    pred_index=None):
    
    # Remove final layer
    model.layers[-1].activation = None
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
    
def save_and_display_gradcam(img_path, heatmap, display=display_cam,
    save=save_cam, output_path=None):
    # Load the original image
    img = get_img_array(img_path, img_size, bw=True)[:, :, 0]

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Resize image with RGB colorized heatmap to match original image size
    heatmap = np.expand_dims(heatmap, axis=2)
    heatmap = resize_img_array(heatmap, img.shape)[:, :, 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2,
        sharey=True,
        layout='compressed',
        dpi=300,
        figsize=(1.05, 2))
    
    ax1.imshow(img, cmap='viridis')
    ax2.imshow(heatmap, cmap='jet')
    for ax in (ax1, ax2):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pass
    fig.patch.set_alpha(0)
    
    if display:
        plt.show()
        
    if save:
        plt.savefig(output_path)
    
    plt.close()
    
    return heatmap

def get_latest_version(model_dir):
    model_list = sorted(glob.glob(str(Path(model_dir, '*.h5'))))
    
    try:
        v_num = model_list[-1][-6:-3]
    except IndexError:
        v_num = '000'
        
    return v_num

def main():
    # Load train and test dataset
    X_train, X_test, y_train, y_test = create_dataset()
    
    # Train new model
    if train_new:
        # Train model on the training dataset and save to file with a new version number
        train(X_train, X_test, y_train, y_test)
        
    # Load the latest model
    v_num = get_latest_version(model_dir)
    model = load_model(Path(model_dir, f'CWT_image_binary_classification_{v_num}.h5'))
    
    # Run model evaluation
    if evaluate:
        model_eval(model, X_test, y_test)
        
if __name__ == '__main__':
    main()