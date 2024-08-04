# Deepfake Detection using Transfer Learning and Multi Modal Fusion
This project presents a novel deepfake detection model that leverages state-of-the-art deep learning architectures and techniques. The model is designed to be robust, scalable, and capable of detecting deepfakes across various modalities, including images and videos.
# Features
-Multimodal Fusion: The model combines grayscale representations with RGB images to enrich the feature space and improve classification performance.

-Convolutional Neural Network (CNN): The model uses a CNN to extract relevant features from preprocessed frames for deepfake detection.

-Generative Adversarial Network (GAN): The model employs a GAN architecture with a generator and discriminator network to generate synthetic images and classify real vs fake.

-Temporal Information: By analyzing frames over time, the model captures temporal inconsistencies that help distinguish real videos from deepfakes.

-Local Noise Residuals: The model analyzes local noise residuals in video frames to identify low-level camera characteristics that serve as discriminative features.

# Implementation Details
1. Setup and Dependencies
- Imports and Installs
At the top, you install several libraries like `moviepy`, `autokeras`, and `keras_nlp`. These
libraries are used for video processing, neural architecture search (NAS), and natural language
processing with Keras, respectively.
- Basic TensorFlow and Other Libraries
The code imports TensorFlow/Keras, various data manipulation libraries (like `numpy`,
`shutil`, `os`), and plotting libraries (`matplotlib.pyplot`). It also includes some specific
modules for audio and video processing, such as `torchaudio` and `VideoFileClip`.
2. Data Preparation
- Directory Creation and File Copying
The code creates directories to organize video data into 'Real' and 'Fake' categories. It then
copies 100 real and 100 fake video files from the source directories to the respective destination
directories.
- Video Frame Extraction
The function `extract_video_frames()` extracts frames from a video. It uses OpenCV (`cv2`) to
read and process the frames.
- Dataset Creation and Splitting
The code collects real and fake video paths, extracts frames from them, and creates a dataset
with images and corresponding labels (0 for real, 1 for fake). It then splits the data into training
and test sets using `train_test_split`.
3. Building and Training the GAN
-Generator

  generator_input = Input(shape=(latent_dim,)): Input layer for the generator, taking a noise
vector of size latent_dim.

x = Dense(64)(generator_input): Dense (fully connected) layer with 64 units.

x = LeakyReLU()(x): LeakyReLU activation function applied to the dense layer.

x = Dense(128)(x): Another dense layer with 128 units.

x = LeakyReLU()(x): LeakyReLU activation function applied again.

num_units = output_shape[0] * output_shape[1] * output_shape[2]: Calculate the total number
of units needed to reshape into the desired output shape.

generated_data = Dense(num_units, activation='tanh')(x): Dense layer with num_units and a
tanh activation function to generate data.

generated_data = Reshape(output_shape)(generated_data): Reshape the generated data to the
desired output shape.

generator = Model(generator_input, generated_data): Create the generator model.

return generator: Return the constructed generator model.

-Discriminator

discriminator_input = Input(shape=input_shape): Input layer for the discriminator, taking
images of shape input_shape.

x = Dense(128)(discriminator_input): Dense layer with 128 units.

x = LeakyReLU()(x): LeakyReLU activation function applied.

x = Dense(64)(x): Another dense layer with 64 units.

x = LeakyReLU()(x): LeakyReLU activation function applied.

validity = Dense(1, activation='sigmoid')(x): Dense layer with 1 unit and a sigmoid activation
function to output the validity of the input (real or fake).

discriminator = Model(discriminator_input, validity): Create the discriminator model.
return discriminator: Return the constructed discriminator model
-GAN

discriminator.trainable = False: Freeze the discriminator's layers so they are not updated during
the generator's training.

gan_input = Input(shape=(latent_dim,)): Input layer for the GAN, taking a noise vector.

generated_data = generator(gan_input): Pass the input through the generator to get generated
data.

gan_output = discriminator(generated_data): Pass the generated data through the discriminator
to get the validity score.

gan = Model(gan_input, gan_output): Create the GAN model that combines the generator and
discriminator.

return gan: Return the constructed GAN model.

4. Multimodal Fusion and HDF5 Data Storage
- Multimodal Fusion

This step creates additional modalities (like grayscale images) and fuses them with the original
data, allowing for richer input to neural networks.

-The block of code creates additional modalities by converting each frame to grayscale:
additional_modalities: An empty list that will store the new modalities.

for i in range(1, num_modalities): Loop to create multiple modalities (starting from 1 since 0 is
the original modality).

Inside the loop, modified_images is an empty list that will store the grayscale frames for each
modality.

for frame in range(num_frames): Loop over each frame.

grayscale_frame = np.mean(images[:, frame], axis=-1, keepdims=True): Convert the frame to

grayscale by taking the mean across the color channels.

modified_images.append(grayscale_frame): Append the grayscale frame to modified_images.

additional_modalities.append(np.array(modified_images)): Convert modified_images to a
NumPy array and append it to additional_modalities.

The next block of code tiles the additional modalities to match the number of frames:

tiled_modalities: An empty list that will store the tiled modalities.

[modalities[:, np.newaxis] for modalities in additional_modalities]: Add a new axis to each
modality array, so it can be tiled correctly.

np.tile(modalities, (1, num_frames, 1, 1, 1)): Tile each modality array to have the same number
of frames as the original images.

fused_data = np.concatenate((images, *tiled_modalities), axis=-1): Concatenate the original
images array and all tiled_modalities arrays along the last axis (the channel axis).

return fused_data: Return the concatenated array, which now includes both the original images
and the additional grayscale modalities.

- HDF5 Storage

The processed data is saved in an HDF5 file, a format suitable for large datasets. It helps with
efficient storage and retrieval during model training.

5. AutoKeras Neural Architecture Search (NAS)

- Using AutoKeras for Image Classification

AutoKeras is used to search for the best neural network architecture for a given dataset. This
code creates an image-based NAS model to find the best architecture for classifying real vs.
fake videos.
- The `ImageClassifier` from AutoKeras is used, and the search is set to perform two trials to
find a suitable model.
- Training and Evaluation
  
The best model from AutoKeras is trained on the training set and evaluated on the validation
set. The code then saves the best model for later use.
-The line initializes an AutoKeras ImageClassifier with a maximum of 2 trials. AutoKeras will
automatically search for the best neural network architecture for image classification.
Next lines train the AutoKeras model using the reshaped training data for 4 epochs and validate
it on the validation set. The training history is stored in history, and the keys of the history (such
as loss and accuracy) are printed.

The exported model architecture to a file named best_model.h5 in the /kaggle/working
directory.Next line creates a visual plot of the best model architecture and saves it as
best_model.png. The show_shapes=True argument ensures that the shapes of the layers are
displayed.

-Autokeras Parameters

The model training process uses the following Autokeras parameters:

1. Batch Size: The batch size is set to 32.
2. Number of Epochs: The number of epochs is set to 10.
3. Learning Rate: The learning rate is set to 0.001.
4. Optimizer: The optimizer used is Adam.

# Result
Experimental results demonstrate the efficacy of the multimodal fusion approach in enhancing video classification accuracy. The model achieves 100% accuracy in detecting deepfakes, making it a valuable tool for identifying and preventing the spread of manipulated media
# Limitations
Data Quality and Diversity: The model's performance relies heavily on the quality and diversity of the training data.

Model Complexity: The integration of multiple modalities and complex architectures like GANs and CNNs increases computational demands.

Adaptability to New Techniques: As deepfake generation techniques evolve, the model may become less effective against new methods.

Real-Time Processing: The computational requirements may hinder real-time processing of videos.

