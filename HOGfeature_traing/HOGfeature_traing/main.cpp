#include <iostream>

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "FeatureExtractor.h"

Size WINDOW_SIZE;
Size  CELL_SIZE;
void extractorFeature() 
{
	FeatureExtractor* extractor = new FeatureExtractor();
	Mat positive = extractor->ExtractorPositiveSample();
	std::cout << "positive extract finish!" << std::endl;
	Mat negative = extractor->ExtractorNegativeSample();
	//Mat negative = extractor->ExtractorNegativeSample();
	std::cout << "negative extract finish!" << std::endl;


	Mat trainingDataMat(positive.rows + negative.rows, positive.cols, CV_32FC1);
	//將正樣本資料放入mixData
	memcpy(trainingDataMat.data, positive.data, sizeof(float)*positive.rows*positive.cols);
	//將負樣本資料放入mixData
	memcpy(&trainingDataMat.data[sizeof(float)*positive.rows*positive.cols], negative.data, sizeof(float)*negative.rows*negative.cols);
	Mat dataProperty(positive.rows + negative.rows, 1, CV_32SC1, Scalar(-1.0));
	dataProperty.rowRange(0, positive.rows) = Scalar(1.0);

	std::cout << "memcpy finish!" << std::endl;

	positive.release();
	negative.release();

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(cv::ml::SVM::Types::C_SVC);
	svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
	//svm->setKernel(cv::ml::SVM::KernelTypes::RBF);
	svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, 100000, 1e-6));
	svm->train(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, dataProperty);
	svm->save("Cate23_C_SVC_LINEAR.xml");
	std::cout << "svm training is done!" << std::endl;		
}


Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = CELL_SIZE.height;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(0, 0, 255), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(255, 0, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu
void seeHogFeature() 
{	
	vector<float> features;
	cv::Mat visualization;	
	string name = "pos11.jpg";
	Mat pic = imread(name, 0);
	WINDOW_SIZE = Size(pic.cols, pic.rows);
	CELL_SIZE = Size(4, 4);
	HOGDescriptor descriptor(WINDOW_SIZE, Size(12, 12), Size(2, 2), CELL_SIZE, 9);
	/*WINDOW_SIZE= Size(pic.cols,pic.rows);
	CELL_SIZE=Size(8,8);
	HOGDescriptor descriptor(WINDOW_SIZE, Size(CELL_SIZE.width * 2, CELL_SIZE.height * 2), CELL_SIZE, CELL_SIZE, 9);*/
	descriptor.compute(pic, features, Size(0, 0), Size(0, 0), vector<Point>());
	visualization = get_hogdescriptor_visu(imread(name, 1), features, WINDOW_SIZE);
	cv::imshow("Visualization", visualization);
	cv::waitKey();	
}
int main()
{
	int mode= 0;
	switch (mode)
	{
		case 0:
			extractorFeature();
		break;			
		case 1:
			seeHogFeature();
		break;
	}
	system("pause");
}
