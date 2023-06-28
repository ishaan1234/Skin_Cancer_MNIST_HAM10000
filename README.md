1. Introduction
In this project, my objective is to detect cancer cells using a dataset obtained from the Kaggle repository. The dataset consists of skin lesion images annotated with various classes of skin cancer. By leveraging the power of Convolutional Neural Networks (CNN), I aim to develop an accurate and reliable model for cancer cell detection.
2. Data Preparation
I begin by loading the dataset's metadata from the provided CSV file using the Pandas library. The metadata contains important information such as image IDs, dx type, age and corresponding labels. Additionally, I define the directories where the image files are stored.
3. Image Path Retrieval
To create the image paths, I extract the image IDs from the metadata and search for the corresponding image files in the provided directories. This is done by comparing the image IDs with the file names in each directory. The resulting image paths are stored in the img_paths list.
4. Image Pre-processing
I preprocess the images before feeding them into the CNN model. Each image is loaded using the PIL library, resized to a consistent size of 128x128 pixels, and converted to a NumPy array. I normalize the pixel values to the range of [0, 1] by dividing them by 255.0. The preprocessed images are stored in the images list, while the corresponding labels are stored in the processed_labels list.
5. Label Encoding
In order to train the CNN model, I perform label encoding on the processed labels using the LabelEncoder from the scikit-learn library. This converts the categorical labels into numeric representations, making them suitable for model training.
6. Dataset Splitting
To evaluate the performance of the CNN model, I split the preprocessed images and their corresponding labels into training and testing sets. The train_test_split function from scikit-learn is used for this purpose, with a test size of 20% and a random state of 42.
7. CNN Model Architecture
I define the architecture of the CNN model using the TensorFlow framework. The model consists of multiple convolutional layers folloId by max pooling layers, which extract important features from the input images. The output of the last max pooling layer is flattened and passed through fully connected layers to perform the final classification. The model is compiled with the Adam optimizer, the sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.
8. Model Training
The CNN model is trained on the training set using the fit method. I set a batch size of 32 and train the model for a maximum of 100 epochs. To prevent overfitting, I incorporate early stopping with a patience of 4 epochs, which stops the training if the validation accuracy does not improve. This helps us find the optimal point of training where the model performs Ill on unseen data.
