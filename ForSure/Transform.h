#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"

#include <windows.h>
#include <conio.h>
#include <time.h>

#pragma once

using namespace cv;
using namespace ml;


const int MAX_MILESTONES = 11;
const int NR_COORDS = 4;

const int INFANT_DATA = 1;
const int TUMOR_DATA = 2;

const int ALG_KNN = 2;
//const int ALG_DTREE = 2;
const int ALG_RF = 1;
//const int ALG_ANN = 4;
//const int ALG_ADA = 5;
//const int ALG_LOGREG = 6;
const int ALG_SVM = 3;


struct PatientBasicData
{
	int firstIndex;
	int lastIndex;
	int pixelCount;
};


class ConfusionMatrix
{
public:
	ConfusionMatrix(int _size)
	{
		size = _size;
		CM = (int**)malloc(size * sizeof(int*));
		for (int i = 0; i < size; ++i)
			CM[i] = (int*)calloc(size, sizeof(int));
		row = (int*)calloc(size, sizeof(int));
		col = (int*)calloc(size, sizeof(int));
		tpr = (float*)malloc(size * sizeof(float));
		ppv = (float*)malloc(size * sizeof(float));
		tnr = (float*)malloc(size * sizeof(float));
		dsc = (float*)malloc(size * sizeof(float));
	};
	~ConfusionMatrix()
	{
		for (int i = 0; i < size; ++i)
			free(CM[i]);
		free(CM);
		free(row);
		free(col);
		free(tpr);
		free(ppv);
		free(tnr);
		free(dsc);
	};
	void reset()
	{
		for (int i = 0; i < size; ++i) for (int j = 0; j < size; ++j) CM[i][j] = 0;
	}
	void addItem(int gt, int deci)
	{
		if (gt >= 0 && gt < size && deci >= 0 && deci < size)
			CM[gt][deci]++;
	}
	char* compute()
	{
		char buffer[1000];
		buffer[0] = 0;
		sum = 0;
		diag = 0;
		for (int i = 0; i < size; ++i)
		{
			row[i] = 0;
			col[i] = 0;
		}

		for (int gt = 0; gt < size; ++gt)
			for (int deci = 0; deci < size; ++deci)
			{
				row[gt] += CM[gt][deci];
				col[deci] += CM[gt][deci];
				sum += CM[gt][deci];
				if (gt == deci)
					diag += CM[gt][deci];
			}
		acc = (float)diag / (float)sum;
		for (int gt = 0; gt < size; ++gt)
		{
			tpr[gt] = (float)CM[gt][gt] / (float)row[gt];
			tnr[gt] = (float)(sum - row[gt] - col[gt] + CM[gt][gt]) / (float)(sum - row[gt]);
			dsc[gt] = (float)(2 * CM[gt][gt]) / (float)(row[gt] + col[gt]);
		}
		for (int deci = 0; deci < size; ++deci)
			ppv[deci] = (float)CM[deci][deci] / (float)(col[deci] > 0 ? col[deci] : 1);

//		FILE* G = fopen("csongor-images/eval.csv", "at");
		FILE* F = fopen("kungfu.txt", "at");
		fprintf(F, "\nConfusion matrix [CSF][GM][WM]\n");
		printf("\nConfusion matrix [CSF][GM][WM]\n");
		for (int gt = 0; gt < size; ++gt)
		{
			for (int deci = 0; deci < size; ++deci)
			{
				sprintf(buffer, "%s%d,", buffer, CM[gt][deci]);
//				fprintf(G, "%d,", CM[gt][deci]);
				fprintf(F, "%10d", CM[gt][deci]);
				printf("%10d", CM[gt][deci]);
			}
			fprintf(F, "\n");
			printf("\n");
		}
		for (int gt = 0; gt < size; ++gt)
		{
			sprintf(buffer, "%s%.5f,%.5f,%.5f,%.5f,", buffer, dsc[gt], tpr[gt], tnr[gt], ppv[gt]);
//			fprintf(G, "%.5f,%.5f,%.5f,%.5f,", dsc[gt], tpr[gt], tnr[gt], ppv[gt]);
			fprintf(F, "Class: %d, DSC=%.5f, TPR=%.5f, TNR=%.5f, PPV=%.5f\n", gt, dsc[gt], tpr[gt], tnr[gt], ppv[gt]);
			printf("Class: %d, DSC=%.5f, TPR=%.5f, TNR=%.5f, PPV=%.5f\n", gt, dsc[gt], tpr[gt], tnr[gt], ppv[gt]);
		}
		sprintf(buffer, "%s%.5f", buffer, acc);
//		fprintf(G, "%.5f\n", acc);
		fprintf(F, "Global accuracy: ACC=%.5f\n", acc);
		printf("Global accuracy: ACC=%.5f\n", acc);
		fclose(F);
//		fclose(G);
		return strdup(buffer);
	}
private:
	int** CM;
	int size;
	int* row;
	int* col;
	int sum;
	int diag;
	float* tpr;
	float* ppv;
	float* tnr;
	float* dsc;
	float acc;
};


class Tester
{
public:
//	Tester(char* fname, int _testID, int _whichData);
	Tester(char* fname, int _testID, int _whichData, int _atlas);
	~Tester();

	void TrainAndTest(int _classifier, int _testItem, int _nrTrainPixPerPatient, int _param, int _param2);
	void MultiSVM(int _testItem, int _nrTrainPixPerPatient, int _param, int RES_IMAGE);
//	void MultiKNN(int _testItem, int _nrTrainPixPerPatient, int _param, int RES_IMAGE);
//	void OcvBoost(int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES);

protected:
	void GenerateFeatures();
	void CreateBuffers();
	void ReleaseBuffers();


private:
	void startTime()
	{
		QueryPerformanceFrequency(&Frekk);
		QueryPerformanceCounter(&Start);
	}

	int endTime()
	{
		QueryPerformanceCounter(&End);
		return (int)(1000.0 * (double(End.QuadPart - Start.QuadPart) / double(Frekk.QuadPart)));
	}

	char* timestamp(); 

	void setColor(IplImage* im, int X, int Y, int r, int g, int b)
	{
		int addr = Y * im->widthStep + 3 * X;
		im->imageData[addr++] = b;
		im->imageData[addr++] = g;
		im->imageData[addr++] = r;
	}

	void setGray(IplImage* im, int X, int Y, int v)
	{
		im->imageData[Y * im->widthStep + X] = v;
	}


private:
	unsigned short** BigFootBuffer;
	uchar** FeatBuffer;
	uchar** PosBuffer;
	uchar* GTBuffer;
	PatientBasicData* patLimits;
//	ConfusionMatrix kungfu, overall;

	int ALG;
	int VOLTYPE;

	int MAX_PATID;
	int MAX_SLICES;
	int WIDTH;
	int HEIGHT;
	int VOLSIZE;
	int MAX_PIXELS;
	int OBSERVED_CHANNELS;
	int MAX_INTENSITY;
	int NR_FEATURES;
	int NR_TISSUES;
	int BATCH;
	int BITS;
	int testID;

	int SLICES_PER_ROW;
	int SLICES_PER_COL;


	LARGE_INTEGER Frekk, Start, End;
};