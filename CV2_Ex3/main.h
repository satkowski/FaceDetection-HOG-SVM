#ifndef MAIN_H
#define MAIN_H

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

#define RANDOM_PATCH_COUNT 100
#define SVM_ITERATIONS 100000
#define SVM_OUTPUT_NAME "SVM_MARC.yaml"

int main(int argc, const char** argv);

cv::String trainSVM(cv::String* positiveTrainPath, cv::String* negativeTrainPath, int windowSize);
void testSVM(cv::String* positiveTestPath, cv::String* negativTestPath, cv::String* svmPath);

#endif // MAIN_H