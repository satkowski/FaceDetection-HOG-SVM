// -pos=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\trainingImages\positive -neg=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\trainingImages\negative
#include "svmTraining.h"

using namespace cv;

int main(int argc, const char** argv)
{
#pragma region Argument parsing

	// Creating a keymap for all the arguments that can passed to that programm
	const String keyMap = "{help h usage ?  |   | show this message}"
						  "{pos p positive  |   | path for the positiv trainimages}"
						  "{neg n negative  |   | path for the negativ trainimages}";

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

	if (positivePath == "")
	{
		printf("There is no positivePath\n");
		return -1;
	}
	if (negativePath == "")
	{
		printf("There is no negativePath\n");
		return -1;
	}

#pragma endregion

	bool train = trainSVM(&positivePath, &negativePath);
	if (!train)
		return -1;
	return 0;
}


bool trainSVM(String* positiveTrainPath, String* negativeTrainPath)
{
#pragma region Initialization

	printf("Initialize\n");
	// Finding all images in both pathes
	std::vector<String> positiveFileNames, negativeFileNames;
	glob(*positiveTrainPath, positiveFileNames);
	glob(*negativeTrainPath, negativeFileNames);

	// Testing if there are images in the pathes
	if (positiveFileNames.size() <= 0)
	{
		printf("There are no images in %s\n", *positiveTrainPath);
		return false;
	}
	if (negativeFileNames.size() <= 0)
	{
		printf("There are no images in %s\n", *negativeTrainPath);
		return false;
	}

	Mat trainingLabel = Mat_<int>(1, positiveFileNames.size() + negativeFileNames.size() * RANDOM_PATCH_COUNT);
	Mat trainingData = Mat_<float>(1764, positiveFileNames.size() + negativeFileNames.size() * RANDOM_PATCH_COUNT);
	int trainingCount = 0;

	clock_t beginTime = clock();

#pragma endregion

#pragma region Positive HOG Descriptors

	// Converting the positve images and calculating the HOG
	std::cout << "Calculate positive HOG Descriptors (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
	for (std::vector<String>::iterator fileName = positiveFileNames.begin(); fileName != positiveFileNames.end(); ++fileName)
	{
		Mat actualImage = imread(*fileName);

		// Testing if the file is an image
		if (actualImage.empty())
		{
			printf("Couldn't read the image %s\n", *fileName);
			return false;
		}
		cvtColor(actualImage, actualImage, CV_BGR2GRAY);
		resize(actualImage, actualImage, Size(WINDOW_SIZE, WINDOW_SIZE));

		// Calculating the HOG
		HOGDescriptor actualHogD;
		actualHogD.winSize = Size(WINDOW_SIZE, WINDOW_SIZE);
		std::vector<float> descriptorsValues;
		std::vector<Point> locations;
		actualHogD.compute(actualImage, descriptorsValues, Size(0, 0), Size(0, 0), locations);

		Mat descriptorsVector = Mat_<float>(descriptorsValues, true);
		descriptorsVector.col(0).copyTo(trainingData.col(trainingCount));
		trainingLabel.at<float>(0, trainingCount) = 1.0;
		trainingCount++;
	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

#pragma endregion

#pragma region Negative HOG Descriptors

	// Calculating the HOG of the negativ images
	std::cout << "Calculate negative HOG Descriptors (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
	for (std::vector<String>::iterator fileName = negativeFileNames.begin(); fileName != negativeFileNames.end(); ++fileName)
	{
		Mat actualImage = imread(*fileName);

		// Testing if the file is an image
		if (actualImage.empty())
		{
			printf("Couldn't read the image %s\n", *fileName);
			return false;
		}
		cvtColor(actualImage, actualImage, CV_BGR2GRAY);

		// Choose the random windows and theire size
		for (int c = 0; c < RANDOM_PATCH_COUNT; c++)
		{
			int rWidth = (rand() % 191) + 10;
			Point rPoint = Point(rand() % (actualImage.cols - rWidth),
								 rand() % (actualImage.rows - rWidth));
			// Pick the window out of the image
			Mat actualWindow;

			resize(actualImage(Range(rPoint.y, rPoint.y + rWidth), Range(rPoint.x, rPoint.x + rWidth)), actualWindow, Size(WINDOW_SIZE, WINDOW_SIZE));

			// Calculating the HOG
			HOGDescriptor actualHogD;
			actualHogD.winSize = Size(WINDOW_SIZE, WINDOW_SIZE);
			std::vector<float> descriptorsValues;
			std::vector<Point> locations;
			actualHogD.compute(actualWindow, descriptorsValues, Size(0, 0), Size(0, 0), locations);

			Mat descriptorsVector = Mat_<float>(descriptorsValues, true);
			descriptorsVector.col(0).copyTo(trainingData.col(trainingCount));
			trainingLabel.at<float>(0, trainingCount) = -1.0;
			trainingCount++;
		}
	}
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

#pragma endregion

#pragma region SVM Training

	// Set up SVM's parameters
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, SVM_ITERATIONS, 1e-6));
	// Create the Trainingdata
	Ptr<ml::TrainData> tData = ml::TrainData::create(trainingData, ml::SampleTypes::COL_SAMPLE, trainingLabel);

	std::cout << "Start SVM training (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
	svm->train(tData);
	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;

	svm->save(SVM_OUTPUT_NAME);

#pragma endregion

	return true;
}
