// Header guards to prevent multiple inclusions of this header file
#ifndef DetectRegions_h
#define DetectRegions_h

// Include necessary header files
#include <string.h>
#include <vector>

#include "Plate.h" // Includes the definition of the 'Plate' class

// OpenCV headers for core functionalities, image processing, and GUI handling
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Standard namespaces to avoid prefixing
using namespace std;
using namespace cv;

// Definition of the DetectRegions class
class DetectRegions {
    public:
        // Constructor
        DetectRegions();

        // Member variable to store the filename
        string filename;

        // Method to set the filename
        void setFilename(string f);

        // Boolean flags to control the saving of regions and displaying processing steps
        bool saveRegions;
        bool showSteps;

        // Method to process the input image and return detected plates
        vector<Plate> run(Mat input);

    private:
        // Method for segmenting the input image to find potential plates
        vector<Plate> segment(Mat input);

        // Method to verify if the sizes of detected regions are within acceptable limits
        bool verifySizes(RotatedRect mr);

        // Method for histogram equalization to improve image contrast
        Mat histeq(Mat in);
};

// End of header guard
#endif
