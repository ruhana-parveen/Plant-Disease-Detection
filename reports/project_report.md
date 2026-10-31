# Plant Disease Detection using CNN

## Abstract
This project demonstrates image classification for plant leaf categories using a Convolutional Neural Network (CNN). The model is trained on the PlantVillage dataset and predicts one of the dataset's 38 categories from a leaf image.

## Problem statement
Manual identification of plant leaf diseases can be difficult for non-experts. This project explores how a CNN can classify leaf images into known dataset categories.

## Objectives
- Load and inspect the PlantVillage dataset.
- Resize and normalize images.
- Build and train a CNN.
- Evaluate training/validation performance.
- Predict the top three categories for an uploaded image.

## Dataset
PlantVillage, accessed through TensorFlow Datasets (`plant_village`). The dataset contains labeled images across 38 categories. Cite the dataset source and its license/terms in the final academic submission.

## Methodology
1. Load labeled images and class names.
2. Split data into training and validation sets.
3. Resize images to 128 × 128 and normalize pixel values to [0, 1].
4. Train a CNN with convolution, max-pooling, dense, dropout, and softmax layers.
5. Save the trained model and exact class-label order.
6. Use Gradio to upload an image and display the top three predictions.

## Results
Insert the actual results from your own Colab run:
- Training accuracy: __________
- Validation accuracy: __________
- Test accuracy (only if a separate test set was evaluated): __________
- Example prediction and confidence: __________

Add your own screenshots of dataset samples, model summary, training output, accuracy/loss plots, confusion matrix (if generated), and Gradio interface.

## Limitations
The model learns from PlantVillage-style images and may perform poorly on real-world photographs with different backgrounds, lighting, camera quality, or plant varieties. Confidence scores are model scores and do not guarantee correctness.

## Conclusion
The project implements a CNN-based plant leaf image classifier and a simple image-upload interface. It provides practical experience with image preprocessing, neural-network training, classification, and model deployment.
