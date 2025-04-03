# auto-encoder
## Nikhila Jajapuram
## 700759188

## Question 1: Basic Autoencoder
### Objective:
Implement a fully connected autoencoder on the MNIST dataset and evaluate reconstruction quality with varying latent dimensions.
###S teps:
### Define and Preprocess Input Data
- Loaded the MNIST dataset using tensorflow.keras.datasets.
- Normalized pixel values to [0, 1] and reshaped images to (num_samples, 784).
### Define Autoencoder Architecture
- Encoder: Input layer (784) → Dense layer (32 units, ReLU).
- Decoder: Dense layer (784 units, Sigmoid) → Output image.
### Compile and Train the Autoencoder
- Used binary_crossentropy as the loss function and Adam optimizer.
- Trained the model to minimize reconstruction error between input and output.
### Visualize Reconstruction
- Predicted reconstructed images from test data.
- Plotted original vs. reconstructed images for visual comparison.
### Experiment with Latent Dimensions
- Repeated training with latent dimensions 16 and 64.
- Compared reconstruction quality across different compression levels.

## Question 2: Denoising Autoencoder
### Objective:
Train an autoencoder to reconstruct clean images from noisy MNIST inputs.
###Steps:
### Add Noise to Input Images
- Applied Gaussian noise (mean=0, std=0.5) using np.random.normal.
- Clipped values to stay within [0, 1].
###Build Autoencoder
- Reused the same encoder-decoder structure as in Question 1.
### Train the Denoising Model
- Input: Noisy images
- Target: Clean images
- Trained the model to learn denoising behavior.
### Evaluate and Visualize
- Generated and visualized output images from noisy test inputs.
- Plotted side-by-side: noisy vs. denoised outputs.
### Real-World Application
- Useful in medical imaging or security for recovering meaningful data from noisy inputs.

## Question 3: Text Generation with LSTM
### Objective:
Use an LSTM-based RNN to generate text character-by-character based on a given dataset.
### Steps:
### Load and Preprocess Dataset
- Loaded a corpus.
- Created a character-to-index mapping and one-hot encoded input sequences.
### Define LSTM Model
- Built a sequential model: LSTM layer → Dense output with softmax over all characters.
###Train the Model
- Used sequences of fixed length (e.g., 100 characters) to predict the next character.
- Trained using categorical crossentropy.
###Generate New Text with Temperature Scaling
- Used a seed text and generated output one character at a time.
-	Adjusted temperature:
 -	Low (e.g., 0.5): Conservative, repetitive results.
 -	High (e.g., 1.5): Creative, diverse, less predictable results.

## Question 4: Sentiment Classification using LSTM
### Objective:
Classify IMDB movie reviews as positive or negative using an LSTM-based sentiment classifier. 
### Steps:
### Load and Preprocess Data
-	Used tensorflow.keras.datasets.imdb.
-	Limited vocabulary size (e.g., top 10,000 words).
-	Applied pad_sequences to make review lengths uniform.
### Build the Model
-	Embedding layer → LSTM layer → Dense output with sigmoid activation.
### Train the Model
-	Used binary cross-entropy loss for binary classification.
-	Trained and validated on IMDB training/test sets.
### Evaluate Performance
-	Predicted sentiments on test data.
-	Generated confusion matrix and classification report (precision, recall, F1-score, accuracy).





