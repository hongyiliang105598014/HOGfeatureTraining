#include "FeatureExtractor.h"


FeatureExtractor::FeatureExtractor()
{		
	//dir = "騎士人頭/";
	//WINDOW_SIZE = Size(32,32);

	//dir = "cate23/";
	//WINDOW_SIZE = Size(48, 104);

	//dir = "機車正面全身/";
	//WINDOW_SIZE = Size(48, 104);

	//dir = "機車背面全身/";
	//WINDOW_SIZE = Size(48, 104);

	dir = "cate23/";
	WINDOW_SIZE = Size(72, 88);

	//string dir = "機車側面半身/";
	//WINDOW_SIZE = Size(48, 72);//側面半身

	CELL_SIZE = Size(8, 8);
}

FeatureExtractor::~FeatureExtractor()
{
}

Mat FeatureExtractor::ExtractorPositiveSample()
{	                
	string path = "pos/" + dir;
	ifstream ifs(path + "positive.txt", ios::in);
	//vector<vector<Point>> locationList;
	vector<vector<float>> descriptorValueList;

	if (!ifs.is_open())
	{
		cout << "fail" << endl;
		system("PAUSE");
		return Mat();
	}
	while (!ifs.eof())
	{
		string fileName;
		getline(ifs, fileName);
		if (fileName != "" && fileName.substr(fileName.size() - 3, 3) != "txt")
		{
			Mat img = imread(path + fileName, 0);			
			if (img.empty())
			{
				cout << fileName << "error" << endl;
				system("PAUSE");
				continue;
			}
			else {
				//cout << fileName << endl;
			}
			//resize(img, img, WINDOW_SIZE);
			//imshow("img", img);
			//waitKey(0);			
			vector<Point> location;
			vector<float> descriptorValue;
			HOGDescriptor descriptor(WINDOW_SIZE, Size(CELL_SIZE.width * 2, CELL_SIZE.height * 2), CELL_SIZE, CELL_SIZE, 9);
			//HOGDescriptor descriptor(WINDOW_SIZE, Size(12,12), Size(2,2), Size(4, 4), 9);
			descriptor.compute(img, descriptorValue, Size(0, 0), Size(0, 0), location);
			//locationList.push_back(location);
			descriptorValueList.push_back(descriptorValue);
		}
	}

	int row = descriptorValueList.size();
	int col = descriptorValueList[0].size();
	Mat sample(row, col, CV_32F);
	for (int i = 0; i < row; i++)
	{
		memcpy(&(sample.data[col*i * sizeof(float)]), descriptorValueList[i].data(), col * sizeof(float));
	}
	return sample;
}

Mat FeatureExtractor::ExtractorNegativeSample()
{
	string path = "neg/機車側面全身/";
	
	ifstream ifs(path + "negative.txt", ios::in);
	//vector<vector<Point>> locationList;
	vector<vector<float>> descriptorValueList;

	if (!ifs.is_open())
	{
		cout << "fail" << endl;
		return Mat();
	}
	
	while (!ifs.eof())
	{
		string fileName;
		getline(ifs, fileName);
		if (fileName != "" && fileName.substr(fileName.size() - 3, 3) != "txt")
		{
			
			
			Mat img = imread(path + fileName, 0);
			cout << path + fileName << endl;
			if (img.empty())
			{
				cout << fileName << " error" << endl;
				system("PAUSE");
				continue;
			}
			else {
				//cout << fileName << endl;
			}
			//resize(img, img, WINDOW_SIZE);
			//imshow("img", img);
			//waitKey(0);			
			vector<Point> location;
			vector<float> descriptorValue;
			HOGDescriptor descriptor(WINDOW_SIZE, Size(CELL_SIZE.width * 2, CELL_SIZE.height * 2), CELL_SIZE, CELL_SIZE, 9);
			//HOGDescriptor descriptor(WINDOW_SIZE, Size(12, 12), Size(2, 2), Size(4, 4), 9);
			descriptor.compute(img, descriptorValue, Size(0, 0), Size(0, 0), location);
			//locationList.push_back(location);
			descriptorValueList.push_back(descriptorValue);
			//Mat img = imread(dir + fileName, 0);
			//
			//if (img.rows < WINDOW_SIZE.height || img.cols < WINDOW_SIZE.width)
			//{
			//	resize(img, img, WINDOW_SIZE);
			//}
			//for (int i = 0; (i + WINDOW_SIZE.height) <= img.rows; i += CELL_SIZE.height)
			//{
			//	for (int j = 0; (j + WINDOW_SIZE.width) <= img.cols; j += CELL_SIZE.width)
			//	{
			//		Mat window(img, Rect(j, i, WINDOW_SIZE.width, WINDOW_SIZE.height));
			//		vector<Point> location;
			//		vector<float> descriptorValue;
			//		HOGDescriptor descriptor(WINDOW_SIZE, Size(CELL_SIZE.width * 2, CELL_SIZE.height * 2), CELL_SIZE, CELL_SIZE, 9);
			//		descriptor.compute(window, descriptorValue, Size(0, 0), Size(0, 0), location);
			//		descriptorValueList.push_back(descriptorValue);
			//	}
			//}
		}
	}

	int row = descriptorValueList.size();
	int col = descriptorValueList[0].size();
	Mat sample(row, col, CV_32F);
	for (int i = 0; i < row; i++)
	{
		memcpy(&(sample.data[col*i * sizeof(float)]), descriptorValueList[i].data(), col * sizeof(float));
	}
	return sample;
}