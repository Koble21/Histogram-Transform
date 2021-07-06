#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"

#include <windows.h>
#include <conio.h>


#pragma once


const int MAX_PATID = 10;
const int MAX_SLICES = 256;
const int WIDTH = 144;
const int HEIGHT = 192;
const int VOLSIZE = WIDTH * HEIGHT * MAX_SLICES;
const int MAX_PIXELS = 8000000;
const int MILLIO = 1000000;
const int OBSERVED_CHANNELS = 2;
const int MAX_MILESTONES = 11;
const int MAX_INTENSITY = 1000;
const int NR_FEATURES = 21;
const int NR_COORDS = 4;
const int FIRSTSLICES[MAX_PATID] = { 85,91,91,97,92,89,89,94,98,98 };


struct PatientBasicData
{
	int firstIndex;
	int lastIndex;
	int pixelCount;
};


class HistNormLinear
{
public:
	HistNormLinear(int _qBitDepth, float _l25);
	HistNormLinear();
	~HistNormLinear() {}

	void run();

protected:
	void GenerateFeatures();
	void CreateBuffers();
	void ReleaseBuffers();

protected:
	unsigned short** BigFootBuffer;
	uchar** FeatBuffer;
	uchar** PosBuffer;
	uchar* GTBuffer;
	PatientBasicData patLimits[MAX_PATID];

	float lambda25, lambda75, p_lo, p_hi;
	int qBitDepth, nrMilestones;
	int ALG;
};

class HistNormNyul : public HistNormLinear
{
public:
	HistNormNyul();
	HistNormNyul(int _qBitDepth, float _pLO, int _milestones);
	~HistNormNyul() {};

	virtual void run();
};

class HistNormRoka : public HistNormLinear
{
public:
	HistNormRoka();
	HistNormRoka(int _qBitDepth, float _pLO, int _milestones);
	~HistNormRoka() {};

	virtual void run();

private:
	void HistFuzzyCMeans(int* hist, int lo, int hi);
	
	int fuzzyClusterPrototypes[MAX_MILESTONES];
};
