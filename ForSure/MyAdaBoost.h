#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#include <windows.h>
#include <conio.h>
#include "ConfusionMatrix.h"

using namespace cv;
using namespace ml;

const int MAX_PATID = 10;
const int WIDTH = 144;
const int HEIGHT = 192;
const int SLICES_PER_ROW = 16;
const int SLICES_PER_COL = 7;
const int MAX_SLICES = SLICES_PER_ROW * SLICES_PER_COL;
const int VOLSIZE = WIDTH * HEIGHT * MAX_SLICES;
const int NR_FEATURES = 21;
const int NR_COORDS = 4;
const int BATCH = 10000;
const int MAX_PIXELS = 8000000;

const int GREY_CSF = 50;
const int GREY_GM = 150;
const int GREY_WM = 250;

const int STAGE_ONE = 1;
const int STAGE_TWO = 2;

struct PatientBasicData
{
	int firstIndex;
	int lastIndex;
	int pixelCount;
};

class MyAdaBoost : public ConfusionMatrix
{
public:
	MyAdaBoost(char* dataSource);
	~MyAdaBoost();

	void run(int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int forceNewTraining = 0);
private:
	void startTime();
	int endTime();

	void setGray(IplImage* im, int x, int y, uchar v);
	void setColor(IplImage* im, int x, int y, uchar r, uchar g, uchar b);
	void getColor(IplImage* im, int x, int y, int& r, int& g, int& b);
	void paintPixel(int x, int y, int z, uchar v);
	uchar getGray(IplImage* im, int x, int y);
	void plane2space(int X, int Y, int& x, int& y, int& z);
	void space2plane(int x, int y, int z, int& X, int& Y);
	int getNeighborCount(int X, int Y, int th);

	void CreateBuffers();
	void ReleaseBuffers();

	void LoadFeatures(char* fname);
	void AdaBoost(int stage, uchar* parti);
	void PostProcessing();
	void neighborCheck(bool whichCase, int color);
private:
	LARGE_INTEGER procFreq, timeStart, timeEnd;
	int patID; 
	int trainSize;
	int depthLimit;
	int nextTestID;
	bool trainOnlyIfNecessary;
	IplImage* imRe;

	uchar** FeatBuffer;
	uchar** PosBuffer;
	uchar* GTBuffer;
	PatientBasicData patLimits[MAX_PATID];
	char* evalFileName;
	FILE* evalFile;
};
