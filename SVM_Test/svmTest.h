#ifndef SVMTEST_H
#define SVMTEST_H

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

bool testSVM(cv::String* positiveTestPath, cv::String* negativTestPath, cv::String* svmPath);

#endif // SVMTEST_H