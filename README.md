👤🤖## Face Recognition System
This project implements a facial recognition system using OpenCV's Haar cascades for face detection and LBPH (Local Binary Patterns Histograms) for face recognition. The system is designed to recognize faces in real-time through a webcam. It trains a model using labeled training data and then uses this model to identify faces in video frames.

## Project Overview 🚀

- **Face Detection**: Utilizes Haar cascades to detect faces in images. 🖼️
- **Face Recognition**: Uses the LBPH algorithm to recognize faces based on the trained model. 🔍
- **Real-Time Operation**: Captures video from the webcam and applies face detection and recognition in real-time. 📹

## Dependencies 📦

To run this project, you need to install the following Python packages:

- `opencv-python`: For computer vision tasks including face detection and recognition.
- `numpy`: For numerical operations and array handling.

You can install the required packages using `pip` by running the following command:

`pip install opencv-python numpy`

## 📁 Directory Structure
Prepare your training data in the following structure:

`training_data/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
├── person2/
│   ├── image1.jpg
│   ├── image2.jpg
`

