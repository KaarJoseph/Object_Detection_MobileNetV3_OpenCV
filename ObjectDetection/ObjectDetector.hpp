#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class ObjectDetector
{
public:
	ObjectDetector(cv::dnn::Net neuralNet, std::vector<std::string> classes);
	~ObjectDetector();

	enum SourceFileType {
		Image = 0,
		Video = 1
	};

	void setIODirectory(std::string inputDirectory, std::string outputDirectory);
	int detectObjects(std::string fileName, SourceFileType fileType);

private:
	int analyzeImage(std::string inputFilePath, std::string outputFilePath);
	int analyzeVideo(std::string inputFilePath, std::string outputFilePath);
	std::string filePath(std::string fileDirectory, std::string fileName);
	void analyzeFrame(cv::Mat& currentFrame);
	void drawBoxes(cv::Mat& currentFrame, std::vector<cv::Rect>& boxes, std::vector<int>& accesptedBoxesIndeces, std::vector<std::string>& labels);

	cv::dnn::Net m_neuralNet;
	std::string m_inputDirectory;
	std::string m_outputDirectory;
	std::vector<std::string> m_classes;
	const float m_confidenceTreshold = 0.5;
	const float m_nmsThreshold = 0.35;
};

