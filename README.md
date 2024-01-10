
# Automatic Number Plate Recognition (ANPR) using Deep Convolutional Networks

## Overview
This project is an implementation of an Automatic Number Plate Recognition (ANPR) system using Deep Convolutional Networks and OpenCV using C++. It is designed to detect and recognize license plates in various conditions, including different lightings, distances, and angles.

## Key Features
- Utilizes deep learning models for plate detection and recognition.
- Employs OpenCV for image processing and model integration.
- Capable of handling images with ambiguous lighting and minor perspective distortions.

## Core Objectives
- Image Segmentation and Feature Extraction: Introduction to the basics of image processing and pattern recognition.
- Pattern Recognition Algorithms: Detailed exploration and implementation of Support Vector Machine (SVM) and Deep Neural Network (DNN) algorithms, focusing on convolutional networks.
- Plate Detection and Recognition: Precise detection of license plate location within a camera frame and recognition of alphanumeric characters on the plate using OCR (Optical Character Recognition).

## Technical Specifications
- Target Plate Dimensions: Focus on common large license plates (520 mm width x 110 mm height).
- Character Specifications: Two character groups on the plate, separated by specific spaces. The first group has four numeric digits, and the second has three letters (excluding certain vowels and letters).
- Character Segmentation: Involves verifying both characters and blank spaces for accurate recognition.

## ANPR Algorithm Phases
1. Plate Detection

- Segmentation: Applying filters, morphological operations, and contour algorithms to identify possible plate regions.
- Classification: Utilizing an SVM classifier to distinguish between plate and non-plate image patches.
Plate Recognition

2. OCR Segmentation: 
- Processing the detected plate for character segmentation using histogram equalization and threshold filtering.
- Character Recognition: Employing a Deep Learning model to recognize each character.



## Importing and Using the Model in OpenCV C++
Importing a TensorFlow model into OpenCV is straightforward. We use the `readNetFromTensorflow` method for this purpose. The method accepts the model in protobuf format.

```cpp
dnn::Net dnn_net = readNetFromTensorflow("frozen_cut_graph_opt.pb");
```

For classifying each detected segment of our plate:

```cpp
for(auto& segment : segments){
    Mat ch = preprocessChar(segment.img);
    Mat inputBlob;
    blobFromImage(ch, inputBlob, 1.0f, Size(20, 20), Scalar(), true, false);
    dnn_net.setInput(inputBlob);

    Mat outs;
    dnn_net.forward(outs);
    double max;
    Point pos;
    minMaxLoc(outs, NULL, &max, NULL, &pos);
    input->chars.push_back(strCharacters[pos.x]);
    input->charsPos.push_back(segment.pos);
}
```

### Preprocessing Steps
- Each character segment is preprocessed to a uniform size of 20 x 20 pixels.
- The `blobFromImage` function is used to convert images to blob format.

### Blob Conversion
The `blobFromImage` function takes several parameters including image, scalefactor, size, mean, swapRB, crop, and ddepth.

### Model Prediction
- The `dnn_net.forward(outs)` function passes the input forward through the network to obtain results.
- The `minMaxLoc` function retrieves the label value with the highest probability.

## Building the Project using CMake

### Linux
```bash
export OpenCV_DIR="~/OpenCV/build"
mkdir build
cd build
cmake -D OpenCV_DIR=$OpenCV_DIR ..
make 
```

### MacOSX (Xcode)
```bash
export OpenCV_DIR="~/OpenCV/build"
mkdir build
cd build
cmake -G Xcode -D OpenCV_DIR=$OpenCV_DIR ..
open ANPR.xcodeproj
```

### Windows (MS Visual Studio)
```bash
set OpenCV_DIR="C:\OpenCV\build"
mkdir build
cd build
cmake -G "Visual Studio 9 2008" -D OpenCV_DIR=%OpenCV_DIR% ..
start ANPR.sln 
```

## Conclusion
This ANPR project demonstrates the integration of classical image processing techniques with advanced machine learning algorithms to create a robust system capable of performing automatic number plate recognition under various conditions. The project not only showcases the technical implementation but also provides insights into the challenges and solutions in the domain of computer vision and pattern recognition.