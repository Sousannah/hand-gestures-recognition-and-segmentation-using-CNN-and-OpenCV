# hand-gestures-recognition-and-segmentation-using-CNN-and-OpenCV

This GitHub repository contains code for a Convolutional Neural Network (CNN) model to detect five different hand gestures from segmented hand photos. The model is trained on a dataset using Keras with TensorFlow backend and includes data preprocessing, model building, training, and evaluation. Additionally, there is a separate file, "Real_life_test," demonstrating real-time hand gesture recognition using OpenCV.

## Project Structure

The repository is organized as follows:

1. **Data Preparation:**
   - The dataset is stored in the "data" directory, with subdirectories for each class representing different hand gestures. You can download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/sarjit07/hand-gesture-recog-dataset/data) and extract it into the "data" directory.

2. **CNN Model:**
   - The initial model architecture is defined using Keras Sequential API in the main file.
   - The first attempt may show signs of overfitting, so a second model is created with dropout and regularization to address this issue.
   - The third model utilizes a pre-trained ResNet50 model for feature extraction, followed by additional layers for classification.

3. **Training and Evaluation:**
   - The models are trained using the training set and evaluated on the validation and test sets.
   - Training history, loss, and accuracy plots are visualized for each model.
   - Confusion matrices and classification reports provide detailed performance metrics.

4. **Real-time Hand Gesture Recognition:**
   - The "OpenCV_test" file demonstrates how to use OpenCV for real-time hand gesture recognition.
   - It captures video frames from the camera, preprocesses them, and utilizes the trained model to predict gestures.

5. **Dataset Testing:**
   - The "CNN02_Model_Test" file allows testing the trained model on any provided segmented hand photos.
   - Provide the directory path containing the segmented hand photos, and the model will predict the gestures.

## Instructions:

1. **Dataset:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sarjit07/hand-gesture-recog-dataset/data) and extract it into the "data" directory.
   - I used all the training classes except the "blank" one

2. **Training:**
   - Run the "CNN_Train" file to train and evaluate the CNN models.
   - Experiment with model architectures and hyperparameters to achieve optimal performance.

3. **Real-time Testing:**
   - Execute the "OpenCV_test" file to see real-time hand gesture recognition using OpenCV.

4. **Model Testing on Segmented Photos:**
   - Use the "Images_Test" file to test the trained model on segmented hand photos.
   - Provide the directory path containing the photos, and the model will predict the gestures.

5. **Model Saving:**
   - The trained models are saved in the repository for later use.

## Requirements:

- Python 3.x
- Libraries: TensorFlow, Keras, OpenCV, scikit-learn, matplotlib, seaborn

Feel free to customize and extend the code according to your requirements. For any issues or suggestions, please create an issue in the repository.

Happy coding! 🚀
