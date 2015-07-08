// -pos=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\testImages\positive -neg=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\testImages\negative

#include "main.h"

using namespace cv;

int main(int argc, const char** argv)
{
	// Creating a keymap for all the arguments that can passed to that programm
	const String keyMap = "{help h usage ?  |   | show this message}"
						  "{pos p positive  |   | path for the positiv images}"
						  "{neg n negative  |   | path for the negativ images}";

	// Reading the calling arguments
	CommandLineParser parser(argc, argv, keyMap);
	parser.about("FaceDetection-HOG-SVM");

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	String positivePath = parser.get<String>("pos");
	String negativePath = parser.get<String>("neg");

	// Finding all images in both pathes
	std::vector<String> positiveFileNames, negativeFileNames;
	glob(positivePath, positiveFileNames);
	glob(negativePath, negativeFileNames);

	// Testing if there are images in the pathes
	if (positiveFileNames.size() <= 0)
	{
		printf("There are no images in %s\n", positivePath);
		return -1;
	}
	if (negativeFileNames.size() <= 0)
	{
		printf("There are no images in %s\n", negativePath);
		return -1;
	}
	srand(static_cast<unsigned>(time(0)));

	std::vector<Mat> positiveImages, negativeImages;
	std::vector<HOGDescriptor> positiveHOGs, negativeHOGs;

	// Converting the positve images and calculating the HOG
	for (std::vector<String>::iterator fileName = positiveFileNames.begin(); fileName != positiveFileNames.end(); ++fileName)
	{
		Mat actualImage = imread(*fileName);

		// Testing if the file is an image
		if (actualImage.empty())
		{
			printf("Couldn't read the image %s\n", *fileName);
			return -1;
		}
		positiveImages.push_back(actualImage);
		cvtColor(actualImage, actualImage, CV_BGR2GRAY);


		// Calculating the HOG
		HOGDescriptor actualHogD;
		actualHogD.winSize = Size(64, 64);
		std::vector<float> descriptorsValues;
		std::vector<Point> locations;
		actualHogD.compute(actualImage, descriptorsValues, Size(0, 0), Size(0, 0), locations);

		positiveHOGs.push_back(actualHogD);
	}

	// Calculating the HOG of the negativ images
	for (std::vector<String>::iterator fileName = positiveFileNames.begin(); fileName != positiveFileNames.end(); ++fileName)
	{
	}
	
}