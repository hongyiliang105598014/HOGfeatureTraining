#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <string>
#include <fstream>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

class FeatureExtractor
{
private:
	Size WINDOW_SIZE;
	Size CELL_SIZE;
	string dir;
public:
	FeatureExtractor();
	virtual ~FeatureExtractor();
	Mat ExtractorPositiveSample();
	Mat ExtractorNegativeSample();
};

#endif