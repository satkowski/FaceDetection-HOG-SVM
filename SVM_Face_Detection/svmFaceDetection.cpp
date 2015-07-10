// -svm=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\SVM_Train\SVM_MARC.yaml -img=D:\Dokumente\Workspaces\C++_VS\CV2_Ex3\data\testImages\detection\t1.jpg
#include "svmFaceDetection.h"

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
}

Mat faceDetection(String* imagePath, String* svmPath)
{

}