#ifndef SVMDETECTOR_H
#define SVMDETECTOR_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include <time.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../SVM_Commen/svmDefines.h"

int main(int argc, const char** argv);

int imageDetection(cv::Mat* inputImage, cv::Ptr<cv::ml::SVM> svm);
int webcamDetection(cv::VideoCapture* capture, cv::Ptr<cv::ml::SVM> svm);
cv::Mat faceDetection(cv::Mat* inputImage, cv::Ptr<cv::ml::SVM> svm);
// Sort from the lowest to the highest
bool sortPreditcionVector(std::pair<cv::Point, cv::Vec2f> left, std::pair<cv::Point, cv::Vec2f> right);

#endif // SVMDETECTOR_H