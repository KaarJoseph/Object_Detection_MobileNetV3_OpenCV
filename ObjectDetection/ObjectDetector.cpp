#include "ObjectDetector.hpp"

ObjectDetector::ObjectDetector(cv::dnn::Net neuralNet, std::vector<std::string> classes)
{
	m_neuralNet = neuralNet;
	m_classes = classes;

	m_inputDirectory = std::string();
	m_outputDirectory = std::string();
}

ObjectDetector::~ObjectDetector()
{

}

void ObjectDetector::setIODirectory(std::string inputDirectory, std::string outputDirectory)
{
	m_inputDirectory = inputDirectory;
	m_outputDirectory = outputDirectory;
}

std::string ObjectDetector::filePath(std::string fileDirectory, std::string fileName)
{
	std::string filePath = fileDirectory + "/" + fileName;
	return filePath;
}

int ObjectDetector::detectObjects(std::string fileName, SourceFileType fileType)
{
	int processResult = 0;
	switch (fileType)
	{
		case Image:
		{
			processResult = analyzeImage(filePath(m_inputDirectory, fileName), filePath(m_outputDirectory, fileName));
			break;
		}
		case Video:
		{
			processResult = analyzeVideo(filePath(m_inputDirectory, fileName), filePath(m_outputDirectory, fileName));
			break;
		}
		default:
			break;
	}
	return processResult;
}

int ObjectDetector::analyzeImage(std::string inputFilePath, std::string outputFilePath)
{
	cv::Mat currentFrame;
	currentFrame = cv::imread(inputFilePath);

	if (currentFrame.empty())
		return 1;

	analyzeFrame(currentFrame);

	cv::imwrite(outputFilePath, currentFrame);
	cv::imshow("Result Window", currentFrame);
	cv::waitKey(0);

	cv::destroyAllWindows();
	return 0;
}

int ObjectDetector::analyzeVideo(std::string inputFilePath, std::string outputFilePath)
{
	cv::Mat currentFrame;
	cv::VideoCapture videoCapture;
	cv::VideoWriter videoWriter;
	int videoCodec = videoWriter.fourcc('M', 'P', '4', 'V');

	videoCapture.open(inputFilePath);
	if (!videoCapture.isOpened())
		return 1;

	while (videoCapture.isOpened())
	{
		videoCapture.read(currentFrame);
		if (currentFrame.empty())
			break;

		if (!videoWriter.isOpened())
			videoWriter.open(outputFilePath, -1, 25, cv::Size(currentFrame.cols, currentFrame.rows), true);

		cv::imshow("Original", currentFrame);
		analyzeFrame(currentFrame);
		videoWriter.write(currentFrame);
		cv::imshow("Result Window", currentFrame);
		cv::waitKey(1);
	}

	videoCapture.release();
	videoWriter.release();
	return 0;
}

void ObjectDetector::analyzeFrame(cv::Mat& currentFrame)
{
	int currentFrameRows;
	int currentFrameCols;
	cv::Mat unalteredFrame;
	cv::Mat outputLayer;
	cv::Mat blob;
	std::vector<int> acceptedResults;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<std::string> labels;

	if (((currentFrame.cols > 600) && (currentFrame.cols > currentFrame.rows)) ||
		((currentFrame.rows > 600) && (currentFrame.rows > currentFrame.cols)))
	{
		unalteredFrame = currentFrame.clone();
		int width = (600. / currentFrame.rows) * currentFrame.cols;
		int height = 600;
		cv::Size newDimension = cv::Size(width, height);
		cv::resize(currentFrame, currentFrame, newDimension, cv::INTER_LINEAR);

		currentFrameRows = unalteredFrame.rows;
		currentFrameCols = unalteredFrame.cols;
	}
	else if ((currentFrame.rows > 600) && (currentFrame.rows == currentFrame.cols))
	{
		unalteredFrame = currentFrame.clone();
		int width = 600;
		int height = 600;

		cv::Size newDimension = cv::Size(width, height);
		cv::resize(currentFrame, currentFrame, newDimension, cv::INTER_LINEAR);

		currentFrameRows = unalteredFrame.rows;
		currentFrameCols = unalteredFrame.cols;
	}
	else
	{
		unalteredFrame = currentFrame;
		currentFrameRows = unalteredFrame.rows;
		currentFrameCols = unalteredFrame.cols;
	}

	blob = (cv::dnn::blobFromImage(currentFrame, (1. / 255), cv::Size(currentFrame.cols, currentFrame.rows), cv::Scalar(), true));
	m_neuralNet.setInput(blob);
	outputLayer = m_neuralNet.forward();
	cv::Mat detectionResults(outputLayer.size[2], outputLayer.size[3], CV_32F, outputLayer.ptr<float>());

	cv::Rect rectangle;
	std::string label;
	for (int i = 0; i < detectionResults.rows; i++) {
		float confidence = detectionResults.at<float>(i, 2);

		if (confidence < m_confidenceTreshold)
			continue;

		int left = static_cast<int>(detectionResults.at<float>(i, 3) * currentFrameCols);
		int top = static_cast<int>(detectionResults.at<float>(i, 4) * currentFrameRows);
		int right = static_cast<int>(detectionResults.at<float>(i, 5) * currentFrameCols);
		int bottom = static_cast<int>(detectionResults.at<float>(i, 6) * currentFrameRows);

		int width = right - left;
		int height = bottom - top;

		rectangle = cv::Rect(left, top, width, height);
		boxes.push_back(rectangle);
		confidences.push_back(confidence);

		label = std::string(m_classes[detectionResults.at<float>(i, 1)]);
		label[0] = toupper(label[0]);
		labels.push_back(std::string(label + ":" + std::to_string(confidence).substr(0, 4)));
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, m_confidenceTreshold, m_nmsThreshold, indices);
	drawBoxes(unalteredFrame, boxes, indices, labels);

	currentFrame = unalteredFrame;
}

void ObjectDetector::drawBoxes(cv::Mat& currentFrame, std::vector<cv::Rect>& boxes, std::vector<int>& accesptedBoxesIndeces, std::vector<std::string>& labels)
{
	cv::Rect labelRectangle;
	cv::Size labelSize;
	double scaleFactor;
	int baseline;

	for (auto index : accesptedBoxesIndeces)
	{
		cv::rectangle(currentFrame, boxes[index], cv::Scalar(0, 255, 0), 2);

		labelSize = cv::getTextSize(labels[index], cv::QT_STYLE_NORMAL, 1, 2, &baseline);
		scaleFactor = (boxes[index].height * 0.1) / labelSize.height;

		if (scaleFactor < 1)
			scaleFactor = 1;
		else if (scaleFactor > 3)
			scaleFactor = 3;

		labelRectangle = cv::Rect(cv::Point(boxes[index].x, boxes[index].y - ((labelSize.height * scaleFactor) + baseline)), cv::Size(labelSize.width * scaleFactor, (labelSize.height * scaleFactor) + baseline));
		cv::rectangle(currentFrame, labelRectangle, cv::Scalar(0, 255, 0), cv::FILLED);
		cv::putText(currentFrame, labels[index], cv::Point(boxes[index].x, (boxes[index].y - baseline)), cv::QT_STYLE_NORMAL, scaleFactor, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
	}
}

