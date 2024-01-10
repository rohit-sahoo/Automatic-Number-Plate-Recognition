#ifndef OCR_h
#define OCR_h

#include <string.h>
#include <vector>

#include "Plate.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;


#define HORIZONTAL    1
#define VERTICAL    0

class CharSegment{
public:
    CharSegment();
    CharSegment(Mat i, Rect p);
    Mat img;
    Rect pos;
};

class OCR{
    public:
        bool DEBUG;
        bool saveSegments;
        string filename;
        static const int numCharacters;
        static const char strCharacters[];
        OCR(string trainFile);
        OCR();
        string run(Plate *input);
        int charSize;
        Mat preprocessChar(Mat in);
        int classify(Mat f);
        void train(Mat trainData, Mat trainClasses, int nlayers);
        int classifyKnn(Mat f);
        void trainKnn(Mat trainSamples, Mat trainClasses, int k);
        Mat features(Mat input, int size);

    private:
        bool trained;
        vector<CharSegment> segment(Plate input);
        Mat Preprocess(Mat in, int newSize);        
        Mat getVisualHistogram(Mat *hist, int type);
        void drawVisualFeatures(Mat character, Mat hhist, Mat vhist, Mat lowData);
        Mat ProjectedHistogram(Mat img, int t);
        bool verifySizes(Mat r);
        Ptr<ml::ANN_MLP>  ann;
        Ptr<ml::KNearest> knnClassifier;
        dnn::Net dnn_net;
        int K;
};

#endif
