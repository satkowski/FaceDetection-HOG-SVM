// -svm=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\SVM_Train\SVM_MARC.yaml -img=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\testImages\detection\t1.jpg
#include "svmDetector.h"

using namespace cv;

int main(int argc, const char** argv)
{
#pragma region Argument parsing

	// Creating a keymap for all the arguments that can passed to that programm
	const String keyMap = "{help h usage ?  |   | show this message}"
						  "{svm |   | path to the trained svm}"
						  "{img image  |   | path for the image in wich it find the faces}";

	// Reading the calling arguments
	CommandLineParser parser(argc, argv, keyMap);
	parser.about("FaceDetection-HOG-SVM");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String svmPath = parser.get<String>("svm");
	String imagePath = parser.get<String>("img");

	if (imagePath == "")
	{
		printf("There is no positivePath\n");
		return -1;
	}

#pragma endregion

#pragma region Show Image

	Mat outputImage = faceDetection(&imagePath, &svmPath);
	if (outputImage.empty())
		return -1;

	namedWindow("Face Detection");
	imshow("Face Detection", outputImage);

#pragma endregion

	waitKey();
	return 0;
}

Mat faceDetection(String* imagePath, String* svmPath)
{
#pragma region Initialization

	Mat inputImage = imread(*imagePath);
	if (inputImage.empty())
	{
		printf("This is no image: %s\n", *imagePath);
		return Mat();
	}
	Mat greyImage;
	cvtColor(inputImage, greyImage, CV_BGR2GRAY);

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm = svm->load<ml::SVM>(*svmPath);
	if (!svm->isTrained())
	{
		printf("The SVM isn't trained through this path: %s\n", *svmPath);
		return Mat();
	}
	// Vector that saves the Point in whicht the match was and the preditcion value and the scale factor
	std::vector<std::pair<Point, Vec2f> > positivePatches;

	HOGDescriptor hogD;
	hogD.winSize = Size(WINDOW_SIZE, WINDOW_SIZE);

	clock_t beginTime = clock();

#pragma endregion 

#pragma region Face Detection

	Mat scaledImage = greyImage;
	float scaleFactor = 1;
	std::cout << "Begin the face detection (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";

	while (scaledImage.rows >= WINDOW_SIZE && scaledImage.cols >= WINDOW_SIZE)
	{
#pragma omp parallel for
		for (int cY = 0; cY < (scaledImage.rows - WINDOW_SIZE); cY += 5)
		{
			std::vector<float> descriptorsValues;
			std::vector<Point> locations;
			for (int cX = 0; cX < (scaledImage.cols - WINDOW_SIZE); cX += 5)
			{
				// Take the patch from the image
				Mat imagePatch = scaledImage(Range(cY, cY + WINDOW_SIZE), Range(cX, cX + WINDOW_SIZE));
				// Calculating the HOG
				hogD.compute(imagePatch, descriptorsValues, Size(0, 0), Size(0, 0), locations);
				// Predict with the SVM
				float rawPrediction = svm->predict(descriptorsValues, noArray(), ml::StatModel::RAW_OUTPUT);
#pragma omp critical
				{
					if (rawPrediction < 0)
						positivePatches.push_back(std::pair<Point, Vec2f>(Point(cX, cY), Vec2f(rawPrediction, scaleFactor)));
				}
			}
		}
		// Donwscale the image
		resize(scaledImage, scaledImage, Size(scaledImage.cols * DOWNSCALE_FACTOR, scaledImage.rows * DOWNSCALE_FACTOR));
		// Save the new scalfactor (zoomfactor)
		scaleFactor /= DOWNSCALE_FACTOR;
	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

#pragma endregion

#pragma region Draw Boxes in the image

	// Sort the vector
	std::sort(positivePatches.begin(), positivePatches.end(), sortPreditcionVector);

	Mat outputImage = inputImage;
	std::cout << "Begin the drawing (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
	for (std::vector<std::pair<Point, Vec2f> >::iterator patches = positivePatches.begin(); patches != positivePatches.end(); ++patches)
	{
		// Get the upper-left und lower-right point for the rect 
		Point rectPointUL = patches->first * patches->second[1];
		Point rectPointLR = rectPointUL + (Point(64, 64) * patches->second[1]);
		// Draw the rectangle in the image
		if (patches == positivePatches.begin())
			rectangle(outputImage, rectPointUL, rectPointLR, Scalar(150, 100, 200), 3);
		else
			rectangle(outputImage, rectPointUL, rectPointLR, Scalar(0, 0, 255));
	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

#pragma endregion

	return outputImage;

}

bool sortPreditcionVector(std::pair<Point, Vec2f> left, std::pair<Point, Vec2f> right)
{
	return left.second[0] < right.second[0];
}