#ifndef SVMFACEDETECTION_H
#define SVMFACEDETECTION_H

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

cv::Mat faceDetection(cv::String* imagePath, cv::String* svmPath);

#endif // SVMFACEDETECTION_H