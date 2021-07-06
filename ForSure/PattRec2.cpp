#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#include <windows.h>
#include <time.h>
#include <conio.h>

using namespace cv;
using namespace ml;

const int ALG_KNN = 1;
const int ALG_DTREE = 2;
const int ALG_RF = 3;
const int ALG_ANN = 4;
const int ALG_ADA = 5;
const int ALG_LOGREG = 6;
const int ALG_SVM = 7;

const int MAX_PATID = 10;
const int MAX_SLICES = 256;
const int WIDTH = 144;
const int HEIGHT = 192;
const int SLICES_PER_ROW = 16;
const int SLICES_PER_COL = 7;
const int VOLSIZE = WIDTH*HEIGHT*MAX_SLICES;
const int FIRSTSLICES[10] = {85,91,91,97,92,89,89,94,98,98};
const int MAX_PIXELS = 8000000;
const int NR_FEATURES = 21;
const int NR_COORDS = 4;
const int NR_TISSUES = 3;
const int VERY_BIG_VALUE = MAX_PIXELS;

uchar** FeatBuffer;
uchar** PosBuffer;
uchar* GTBuffer;
int Limits[3][MAX_PATID];

class ConfusionMatrix
{
public:
	ConfusionMatrix(int _size)
	{
		size = _size;
		CM = (int**)malloc(size * sizeof(int*));
		for (int i=0; i<size; ++i)
			CM[i] = (int*)calloc(size,sizeof(int));
		row = (int*)calloc(size, sizeof(int));
		col = (int*)calloc(size, sizeof(int));
		tpr = (float*)malloc(size * sizeof(float));
		ppv = (float*)malloc(size * sizeof(float));
		tnr = (float*)malloc(size * sizeof(float));
		dsc = (float*)malloc(size * sizeof(float));
	};
	~ConfusionMatrix()
	{
		for (int i=0; i<size; ++i)
			free(CM[i]);
		free(CM);
		free(row);
		free(col);
		free(tpr);
		free(ppv);
		free(tnr);
		free(dsc);
	};
	void addItem(int gt, int deci)
	{
		if (gt>=0 && gt<size && deci>=0 && deci<size)
			CM[gt][deci]++;
	}
	void compute()
	{
		sum = 0;
		diag = 0;
		for (int gt = 0; gt<size; ++gt)
		for (int deci = 0; deci<size; ++deci)
		{
			row[gt] += CM[gt][deci];
			col[deci] += CM[gt][deci];
			sum += CM[gt][deci];
			if (gt == deci)
				diag += CM[gt][deci];
		}
		acc = (float)diag / (float)sum;
		for (int gt = 0; gt<size; ++gt)
		{
			tpr[gt] = (float)CM[gt][gt] / (float)row[gt];
			tnr[gt] = (float)(sum - row[gt] - col[gt] + CM[gt][gt]) / (float)(sum - row[gt]);
			dsc[gt] = (float)(2 * CM[gt][gt]) / (float)(row[gt] + col[gt]);
		}
		for (int deci = 0; deci<size; ++deci)
			ppv[deci] = (float)CM[deci][deci] / (float)(col[deci]>0 ? col[deci] : 1);

		FILE* G = fopen("kingkong.csv", "at");
		FILE* F = fopen("kungfu.txt","at");
		fprintf(F, "\nConfusion matrix [CSF][GM][WM]\n");
		printf("\nConfusion matrix [CSF][GM][WM]\n");
		for (int gt=0; gt<size; ++gt)
		{
			for (int deci=0; deci<size; ++deci)
			{	
				if (size>2)
					fprintf(G, "%d,", CM[gt][deci]);
				fprintf(F,"%10d",CM[gt][deci]);
				printf("%10d",CM[gt][deci]);
			}
			fprintf(F,"\n");
			printf("\n");
		}
		for (int gt=0; gt<size; ++gt)
		{
			if (size > 2)
				fprintf(G, "%.5f,%.5f,%.5f,%.5f,", dsc[gt], tpr[gt], tnr[gt], ppv[gt]);
			fprintf(F, "Class: %d, DSC=%.5f, TPR=%.5f, TNR=%.5f, PPV=%.5f\n",gt,dsc[gt],tpr[gt],tnr[gt],ppv[gt]);
			printf("Class: %d, DSC=%.5f, TPR=%.5f, TNR=%.5f, PPV=%.5f\n",gt,dsc[gt],tpr[gt],tnr[gt],ppv[gt]);
		}
		if (size > 2)
			fprintf(G, "%.5f\n", acc);
		fprintf(F, "Global accuracy: ACC=%.5f\n",acc);
		printf("Global accuracy: ACC=%.5f\n",acc);
	    fclose(F);
		fclose(G);
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

LARGE_INTEGER Frekk, Start, End;

void startTime()
{	
	QueryPerformanceFrequency(&Frekk);
	QueryPerformanceCounter(&Start);
}

int endTime()
{
	QueryPerformanceCounter(&End);
	return (int)(1000.0*(double(End.QuadPart-Start.QuadPart)/double(Frekk.QuadPart)));
}


void setGray(IplImage* im, int x, int y, uchar v)
{
	im->imageData[im->widthStep * y + x] = v;
}

uchar getGray(IplImage* im, int x, int y)
{
	return (uchar)(im->imageData[im->widthStep * y + x]);
}

int compare(const void* p1, const void* p2)
{
	if (*((uint*)p1) > *((uint*)p2)) return 1;
	if (*((uint*)p1) < *((uint*)p2)) return -1;
	return 0;
}

void setColor(IplImage* im, int x, int y, uchar blue, uchar green, uchar red)
{
	im->imageData[y*im->widthStep + 3 * x + 2] = red;
	im->imageData[y*im->widthStep + 3 * x + 1] = green;
	im->imageData[y*im->widthStep + 3 * x] = blue;
}

void getColor(IplImage* im, int x, int y, uchar& blue, uchar& green, uchar& red)
{
	red = (uchar)(im->imageData[y*im->widthStep + 3*x + 2]);
	green = (uchar)(im->imageData[y*im->widthStep + 3*x + 1]);
	blue = (uchar)(im->imageData[y*im->widthStep + 3*x]);
}


void plane2space(int X, int Y, int& x, int& y, int& z)
{
	x = X % WIDTH;
	y = Y % HEIGHT;
	z = (X / WIDTH) + (Y / HEIGHT) * SLICES_PER_ROW;
}

void space2plane(int x, int y, int z, int& X, int& Y)
{
	X = x + (z % SLICES_PER_ROW) * WIDTH;
	Y = y + (z / SLICES_PER_ROW) * HEIGHT;
}


void CreateBuffers()
{
	FeatBuffer = (uchar**)malloc(NR_FEATURES * sizeof(uchar*));
	for (int f=0; f<NR_FEATURES; ++f)
		FeatBuffer[f] = (uchar*)malloc(MAX_PIXELS);

	PosBuffer = (uchar**)malloc(NR_COORDS * sizeof(uchar*));
	for (int c=0; c<NR_COORDS; ++c)
		PosBuffer[c] = (uchar*)malloc(MAX_PIXELS);

	GTBuffer = (uchar*)malloc(MAX_PIXELS);
}

void ReleaseBuffers()
{
	for (int f=0; f<NR_FEATURES; ++f)
		free(FeatBuffer[f]);
	free(FeatBuffer);

	for (int c=0; c<NR_COORDS; ++c)
		free(PosBuffer[c]);
	free(PosBuffer);

	free(GTBuffer);
}


void LoadDataFromImages()
{
	CreateBuffers();

	int x,y,z;
	char fname[100];
	int index = 0;
	for (int p=0; p<MAX_PATID; ++p)
	{
		Limits[0][p] = index;
		sprintf(fname,"features/baba-%d-GT.png",p);
		IplImage* imGT = cvLoadImage(fname,0);
		for (int Y=0; Y<imGT->height; ++Y) for (int X=0; X<imGT->width; ++X) 
			if (getGray(imGT,X,Y)>0)
			{
				GTBuffer[index] = getGray(imGT,X,Y) / 100;
				plane2space(X,Y,x,y,z);
				PosBuffer[0][index] = p;
				PosBuffer[1][index] = z;
				PosBuffer[2][index] = y;
				PosBuffer[3][index] = x;
				++index;
			}
		for (int f=0; f<NR_FEATURES; ++f)
		{
			sprintf(fname,"features/baba-%d-ch%02d.png",p,f);
			IplImage* im = cvLoadImage(fname,0);
			int count = Limits[0][p];
			for (int Y=0; Y<imGT->height; ++Y) for (int X=0; X<imGT->width; ++X)
				if (getGray(imGT,X,Y)>0)
				{
					FeatBuffer[f][count] = getGray(im,X,Y);
					++count;
				}
			if (count!=index) printf("[%d %d %d %d]",p,f,index,count);
			cvReleaseImage(&im);
		}
		printf("<%d %d>\n",p,index);
		cvReleaseImage(&imGT);
		Limits[1][p] = index-1;
		Limits[2][p] = Limits[1][p] - Limits[0][p] + 1;
	}
	printf("Saving...");
	FILE* F = fopen("bigbaba.dat","wb");
	fwrite(&index,sizeof(int),1,F);
	fwrite(&NR_FEATURES,sizeof(int),1,F);
	fwrite(Limits[0],sizeof(int),MAX_PATID,F);
	fwrite(Limits[1],sizeof(int),MAX_PATID,F);
	fwrite(Limits[2],sizeof(int),MAX_PATID,F);
	fwrite(GTBuffer,sizeof(uchar),index,F);
	for (int c=0; c<NR_COORDS; ++c)
		fwrite(PosBuffer[c],sizeof(uchar),index,F);
	for (int f=0; f<NR_FEATURES; ++f)
		fwrite(FeatBuffer[f],sizeof(uchar),index,F);
	fclose(F);
	printf("Done.\n");
}

void LoadFeatures()
{
	CreateBuffers();
	int head[2];
	FILE* F = fopen("bigbaba.dat","rb");
	fread(head,sizeof(int),2,F);
	fread(Limits[0],sizeof(int),MAX_PATID,F);
	fread(Limits[1],sizeof(int),MAX_PATID,F);
	fread(Limits[2],sizeof(int),MAX_PATID,F);
	fread(GTBuffer,sizeof(uchar),head[0],F);
	for (int c=0; c<NR_COORDS; ++c)
		fread(PosBuffer[c],sizeof(uchar),head[0],F);
	for (int f=0; f<NR_FEATURES; ++f)
		fread(FeatBuffer[f],sizeof(uchar),head[0],F);
	fclose(F);
	printf("Successfully loaded %d pixel data.\n",head[0]);
}

void DrawSliceFeatures(int patID, int slice)
{
	char fname[100];
	if (patID>=0 && patID<=9 && slice>=0 && slice<=112)
	{
		IplImage* imRe = cvCreateImage(cvSize(5*WIDTH,5*HEIGHT),IPL_DEPTH_8U,1);
		cvSet(imRe,cvScalar(0));
		for (int i=Limits[0][patID]; i<=Limits[1][patID]; ++i)
			if (PosBuffer[0][i] == patID && PosBuffer[1][i] == slice)
		{
			for (int f=0; f<NR_FEATURES; ++f)
			{
				setGray(imRe,(f%5)*WIDTH+PosBuffer[3][i],(f/5)*HEIGHT+PosBuffer[2][i],FeatBuffer[f][i]);
			}
			setGray(imRe,4*WIDTH+PosBuffer[3][i],4*HEIGHT+PosBuffer[2][i],50+100*GTBuffer[i]);
		}
		sprintf(fname,"Patient: %d, Slice: %d",patID,slice);
		cvShowImage(fname,imRe);
		cvWaitKey();
		cvReleaseImage(&imRe);
	}
}

void OwnKNN(int testBaby, int nrTrainPixPerTrainBaby, int kKNN, int USED_FEATURES = NR_FEATURES)
{
	const int PERMUTE[NR_FEATURES] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int kknn = kKNN;
	if (kknn < 1) kknn = 1;
	if (kknn > 25) kknn = 25;

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;
	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

// confusion matrix declaration

	int CM[25][NR_TISSUES][NR_TISSUES] = {0};

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	int* dist = (int*)malloc(sizeof(int)*nrTrainData);
	for (int pix=Limits[0][testBaby]; pix<=Limits[1][testBaby]; ++pix)
	{
		for (int tr = 0; tr<nrTrainData; ++tr)
		{
			dist[tr]=0;
			for (int f=0; f<nrFeat; ++f)
			{
				int testValue = FeatBuffer[PERMUTE[f]][pix];
				int trainValue = FeatBuffer[PERMUTE[f]][selForTrain[tr]];
				dist[tr] += ((testValue-trainValue)*(testValue-trainValue));
			}
		}

		int votes[NR_TISSUES] = {0};
		for (int k=0; k<kknn; ++k)
		{
			int minAt = 0;
			for (int tr = 1; tr<nrTrainData; ++tr)
				if (dist[minAt] > dist[tr]) minAt = tr;
			
			dist[minAt] = VERY_BIG_VALUE;
			votes[GTBuffer[selForTrain[minAt]]]++;

			int deci = 1;
			if (votes[2]>votes[1] && votes[2]>votes[0]) deci=2;
			if (votes[0]>votes[1] && votes[0]>votes[2]) deci=0;
		
			CM[k][GTBuffer[pix]][deci]++;
		}

		if (pix%10000==5678) printf(".");
	}


	FILE* F = fopen("report.txt","at");
	for (int k=0; k<kknn; ++k)
	{
		fprintf(F,"\nConfusion matrix [CSF][GM][WM] for k=%d:\n",k+1);
		printf("\nConfusion matrix [CSF][GM][WM] for k=%d:\n",k+1);
		for (int i=0; i<NR_TISSUES; ++i)
		{
			for (int j=0; j<NR_TISSUES; ++j)
			{	
				fprintf(F,"%8d",CM[k][i][j]);
				printf("%8d",CM[k][i][j]);
			}
			fprintf(F,"\n");
			printf("\n");
		}

		fprintf(F, "Correct decisions %d out of %d, that is %.3f%%\n",CM[k][0][0]+CM[k][1][1]+CM[k][2][2],Limits[2][testBaby],100.0f * ((float)(CM[k][0][0]+CM[k][1][1]+CM[k][2][2]))/((float)(Limits[2][testBaby])));
		printf("Correct decisions %d out of %d, that is %.3f%%\n",CM[k][0][0]+CM[k][1][1]+CM[k][2][2],Limits[2][testBaby],100.0f * ((float)(CM[k][0][0]+CM[k][1][1]+CM[k][2][2]))/((float)(Limits[2][testBaby])));
	}
	fclose(F);
	free(dist);
}

// mindegyik tudjon időt mérni, a tanulás és a tesztelés idejét
// mindegyik tudja kimenteni a megtanult osztályozót
// mindegyik tudja kimenteni az osztályozás eredményét
// egyesített függvény +1 paraméter: algoritmus - switch
// végezni teszteteket: találni olyan beállítást ami kihozza a 81%-ot...osztályozót lementeni, eredményt lementeni, mennyi ideig futott?

void OcvKNN(int testBaby, int nrTrainPixPerTrainBaby, int kKNN, int USED_FEATURES = NR_FEATURES)
{
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = {0,1,12,13,18,19,20,2,3,14,15,16,17,8,9,10,11,4,5,6,7};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int kknn = kKNN;
	if (kknn < 1) kknn = 1;
	if (kknn > 25) kknn = 25;

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;

	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<int> trainLabels(nrTrainData, 1);

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		trainLabels(o,0) = (int)(GTBuffer[selForTrain[o]]);
	}

	Ptr<KNearest> knn(KNearest::create());
// training KNN
	knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);

// confusion matrix declaration

//	int CM[NR_TISSUES][NR_TISSUES] = {0};
	ConfusionMatrix kungfu(NR_TISSUES);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	int pixelsToTest = Limits[2][testBaby];
	int nextTestPixel = Limits[0][testBaby];

	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		knn->findNearest(testFeatures, kknn, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci = response.at<float>(pix-nextTestPixel, 0);
			int gt = GTBuffer[pix];
//			CM[gt][deci]++;
			kungfu.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}

	kungfu.compute();
}

void OcvDecTree(int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES, int RES_IMAGE = 0)
{
	IplImage* imRe;
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int depilimit = maxDepth;
	if (depilimit < 5) depilimit = 5;
	if (depilimit > 50) depilimit = 50;

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;

	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<int> trainLabels(nrTrainData, 1);

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		trainLabels(o,0) = (int)(GTBuffer[selForTrain[o]]);
	}

	Ptr<ml::DTrees> dec_trees = ml::DTrees::create();
	//dec_trees->load("innentoltsdbe.txt");
	//params
	dec_trees->setMaxDepth(depilimit);
	dec_trees->setMinSampleCount(0);
	dec_trees->setRegressionAccuracy(0.0f);
	dec_trees->setUseSurrogates(false);
	dec_trees->setMaxCategories(3);
	dec_trees->setCVFolds(1);
	dec_trees->setUse1SERule(false);
	dec_trees->setTruncatePrunedTree(false);
	dec_trees->setPriors(Mat());

	startTime();
	dec_trees->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
	int trainLapse = endTime();
	printf("Training duration: %.3f seconds.\n",(float)trainLapse*0.001);
	dec_trees->save("dectree.txt");
// confusion matrix declaration

	ConfusionMatrix kungfu(NR_TISSUES);
	if (RES_IMAGE)
	{
		imRe = cvCreateImage(cvSize(WIDTH*SLICES_PER_ROW,HEIGHT*SLICES_PER_COL),IPL_DEPTH_8U,1);
		cvSet(imRe,cvScalar(0));
	}
// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	startTime();
	int pixelsToTest = Limits[2][testBaby];
	int nextTestPixel = Limits[0][testBaby];

	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		dec_trees->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci = response.at<float>(pix-nextTestPixel, 0);
			int gt = GTBuffer[pix];
			if (RES_IMAGE)
			{
				int z = PosBuffer[1][pix];
				int y = PosBuffer[2][pix];
				int x = PosBuffer[3][pix];
				int X, Y;
				space2plane(x,y,z,X,Y);
				setGray(imRe,X,Y,100*deci+50);
			}
			kungfu.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	int testLapse = endTime();
	printf("\nTesting duration: %.3f seconds.\n",(float)testLapse*0.001);

	kungfu.compute();
	if (RES_IMAGE)
	{
		cvSaveImage("dec_tree_result.png",imRe);
		cvReleaseImage(&imRe);
	}
}

void OcvRF(int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES)
{
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int depilimit = maxDepth;
	if (depilimit < 5) depilimit = 5;
	if (depilimit > 50) depilimit = 50;

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;

	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<int> trainLabels(nrTrainData, 1);

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		trainLabels(o,0) = (int)(GTBuffer[selForTrain[o]]);
	}

	Ptr<ml::RTrees> rand_trees = ml::RTrees::create();
	//params
	rand_trees->setMaxDepth(depilimit);
	rand_trees->setMinSampleCount(0);
	rand_trees->setRegressionAccuracy(0.0f);
	rand_trees->setUseSurrogates(false);
	rand_trees->setMaxCategories(3);
	rand_trees->setCVFolds(1);
	rand_trees->setUse1SERule(false);
	rand_trees->setTruncatePrunedTree(false);
//	rand_trees->setPriors(Mat());
	rand_trees->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 45, 0.01f));

	rand_trees->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);

// confusion matrix declaration

	ConfusionMatrix kungfu(NR_TISSUES);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	int pixelsToTest = Limits[2][testBaby];
	int nextTestPixel = Limits[0][testBaby];

	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		rand_trees->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci = response.at<float>(pix-nextTestPixel, 0);
			int gt = GTBuffer[pix];
			kungfu.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}

	kungfu.compute();
}

void OcvANN(int testBaby, int nrTrainPixPerTrainBaby, int kKNN, int USED_FEATURES = NR_FEATURES)
{
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int kknn = kKNN;
	if (kknn < 1) kknn = 1;
	if (kknn > 25) kknn = 25;
	// kknn is ignored in this function

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;

	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<float> trainLabels(nrTrainData, NR_TISSUES);

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		for (int i=0; i<NR_TISSUES; ++i) trainLabels(o,i) = 0.0f;
		trainLabels(o,GTBuffer[selForTrain[o]]) = 1.0f;
	}

	Ptr<ml::ANN_MLP>  nn = ml::ANN_MLP::create();

	//setting the NN layer size
	Mat layers = cv::Mat(4, 1, CV_32SC1);
	layers.row(0) = cv::Scalar(nrFeat);
	layers.row(1) = cv::Scalar(25);
	layers.row(2) = cv::Scalar(15);
	layers.row(3) = cv::Scalar(NR_TISSUES);
	nn->setLayerSizes(layers);

	nn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	nn->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
	nn->setBackpropMomentumScale(0.1);
	nn->setBackpropWeightScale(0.1);
	nn->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 200, 1e-6));
	
	int res = nn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);

// confusion matrix declaration

	ConfusionMatrix kungfu(NR_TISSUES);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	int pixelsToTest = Limits[2][testBaby];
	int nextTestPixel = Limits[0][testBaby];

	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		nn->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			float fres[3];
			for (int o = 0; o < NR_TISSUES; ++o)
				fres[o] = response.at<float>(pix-nextTestPixel, o);
			int deci = 0;
			for (int o = 1; o < NR_TISSUES; ++o)
				if (fres[o]>fres[deci]) deci = o;
			int gt = GTBuffer[pix];
			kungfu.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}

	kungfu.compute();
}

void OcvALL(int alg, int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES, int RES_IMAGE = 0)
{
	IplImage* imRe;
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 };
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int depilimit = maxDepth;
	if (depilimit < 5) depilimit = 5;
	if (depilimit > 50) depilimit = 50;
	int kknn = maxDepth;
	if (kknn < 1) kknn = 1;
	if (kknn > 25) kknn = 25;

	printf("\nTraining begins...\n");
	// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID - 1) * nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int) * nrTrainData);
	int trainCount = 0;

	for (int patID = 0; patID < MAX_PATID; ++patID)
		if (testBaby != patID)
		{
			int count = 0;
			int index = 0;
			while (count < nrTrainPixPerTrainBaby)
			{
				index = (index + rand() * rand() + rand()) % Limits[2][patID];
				selForTrain[trainCount] = Limits[0][patID] + index;
				++count;
				++trainCount;
			}
		}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<int> trainLabels(nrTrainData, 1);
	Mat_<float> ANNtrainLabels(nrTrainData, NR_TISSUES);

	for (int o = 0; o < nrTrainData; ++o)
	{
		for (int f = 0; f < nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o, f) = value;
		}
		trainLabels(o, 0) = (int)(GTBuffer[selForTrain[o]]);
		for (int i = 0; i < NR_TISSUES; ++i) ANNtrainLabels(o, i) = 0.0f;
		ANNtrainLabels(o, GTBuffer[selForTrain[o]]) = 1.0f;
	}

	int trainLapse, testLapse;
	Ptr<ml::DTrees> dec_trees = ml::DTrees::create();
	Ptr<ml::RTrees> rand_trees = ml::RTrees::create();
	Ptr<KNearest> knn(KNearest::create());

	Ptr<ml::ANN_MLP>  nn = ml::ANN_MLP::create();
	Mat layers = cv::Mat(4, 1, CV_32SC1);

	switch (alg)
	{
	case ALG_KNN:
		startTime();
		knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		trainLapse = endTime();
		break;
	case ALG_ANN:
		//setting the NN layer size
		layers.row(0) = cv::Scalar(nrFeat);
		layers.row(1) = cv::Scalar(32);
		layers.row(2) = cv::Scalar(16);
		layers.row(3) = cv::Scalar(NR_TISSUES);
		nn->setLayerSizes(layers);
		nn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
		nn->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
		nn->setBackpropMomentumScale(0.1);
		nn->setBackpropWeightScale(0.1);
		nn->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 500, 1e-6));
		startTime();
		nn->train(trainFeatures, ml::ROW_SAMPLE, ANNtrainLabels);
		trainLapse = endTime();
		nn->save("anna.txt");
		break;
	case ALG_DTREE:
		//dec_trees->load("innentoltsdbe.txt");
		//params
		dec_trees->setMaxDepth(depilimit);
		dec_trees->setMinSampleCount(0);
		dec_trees->setRegressionAccuracy(0.0f);
		dec_trees->setUseSurrogates(false);
		dec_trees->setMaxCategories(3);
		dec_trees->setCVFolds(1);
		dec_trees->setUse1SERule(false);
		dec_trees->setTruncatePrunedTree(false);
		dec_trees->setPriors(Mat());
		startTime();
		dec_trees->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		trainLapse = endTime();
		dec_trees->save("dectree.txt");
		break;
	case ALG_RF:
		//params
		rand_trees->setMaxDepth(depilimit);
		rand_trees->setMinSampleCount(0);
		rand_trees->setRegressionAccuracy(0.0f);
		rand_trees->setUseSurrogates(false);
		rand_trees->setMaxCategories(3);
		rand_trees->setCVFolds(1);
		rand_trees->setUse1SERule(false);
		rand_trees->setTruncatePrunedTree(false);
		//	rand_trees->setPriors(Mat());
		rand_trees->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 45, 0.01f));
		startTime();
		rand_trees->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		trainLapse = endTime();
		rand_trees->save("rf.txt");
		break;
	}
	printf("Training duration: %.3f seconds.\n", (float)trainLapse * 0.001);
	// confusion matrix declaration

	ConfusionMatrix kungfu(NR_TISSUES);
	if (RES_IMAGE)
	{
		imRe = cvCreateImage(cvSize(WIDTH * SLICES_PER_ROW, HEIGHT * SLICES_PER_COL), IPL_DEPTH_8U, 1);
		cvSet(imRe, cvScalar(0));
	}
	// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	startTime();
	int pixelsToTest = Limits[2][testBaby];
	int nextTestPixel = Limits[0][testBaby];

	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix = nextTestPixel; pix < nextTestPixel + thisBatch; ++pix)
		{
			for (int f = 0; f < nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix - nextTestPixel, f) = value;
			}
		}
		Mat response;

		if (alg==ALG_DTREE)
			dec_trees->predict(testFeatures, response);
		else if (alg==ALG_RF)
			rand_trees->predict(testFeatures, response);
		else if (alg == ALG_KNN)
			knn->findNearest(testFeatures, kknn, response);
		else if (alg == ALG_ANN)
			nn->predict(testFeatures, response);

		for (int pix = nextTestPixel; pix < nextTestPixel + thisBatch; ++pix)
		{
			int deci;
			if (alg == ALG_ANN)
			{
				float fres[3];
				for (int o = 0; o < NR_TISSUES; ++o)
					fres[o] = response.at<float>(pix - nextTestPixel, o);
				deci = 0;
				for (int o = 1; o < NR_TISSUES; ++o)
					if (fres[o] > fres[deci]) deci = o;
			}
			else
				deci = response.at<float>(pix - nextTestPixel, 0);
			int gt = GTBuffer[pix];
			if (RES_IMAGE)
			{
				int z = PosBuffer[1][pix];
				int y = PosBuffer[2][pix];
				int x = PosBuffer[3][pix];
				int X, Y;
				space2plane(x, y, z, X, Y);
				setGray(imRe, X, Y, 100 * deci + 50);
			}
			kungfu.addItem(gt, deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	testLapse = endTime();
	printf("\nTesting duration: %.3f seconds.\n", (float)testLapse * 0.001);

	FILE* G = fopen("kingkong.csv", "at");
	fprintf(G, "%d,%d,%d,%d,%d,%d,%d,", alg, testBaby, nrTrainPixPerTrainBaby, maxDepth, USED_FEATURES, trainLapse, testLapse);
	fclose(G);
	kungfu.compute();
	if (RES_IMAGE)
	{
		cvSaveImage("prediction_result.png", imRe);
		cvReleaseImage(&imRe);
	}
}


void OcvBoost(int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES)
{
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int depilimit = maxDepth;
	if (depilimit < 5) depilimit = 5;
	if (depilimit > 50) depilimit = 50;

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;

	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<int> trainLabels(nrTrainData, 1);

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
//		trainLabels(o,0) = (int)(GTBuffer[selForTrain[o]]);
		trainLabels(o,0) = (GTBuffer[selForTrain[o]] > 0 ? 1 : 0);
	}

	Ptr<ml::Boost> ada = ml::Boost::create();
	ada->setMaxDepth(depilimit);
	startTime();
	ada->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
	int lapseTrain = endTime();
	printf("Training duration so far: %.3f seconds.\n", (float)lapseTrain * 0.001);
	ConfusionMatrix kungfu(2);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	int pixelsToTest = Limits[2][testBaby];
	int firstTestPixel = Limits[0][testBaby];
	int nextTestPixel = Limits[0][testBaby];
	// partition
	uchar* parti = (uchar*)calloc(pixelsToTest,1);

	startTime();
	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		ada->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci = response.at<float>(pix-nextTestPixel, 0);
			if (deci>0) deci=1;
			parti[pix-firstTestPixel]=deci;
			int gt = GTBuffer[pix];
			if (gt>0) gt=1;
			kungfu.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	int lapseTest = endTime();
	printf("\nTesting duration so far: %.3f seconds.\n", (float)lapseTest * 0.001);
	kungfu.compute();

	// 2nd stage
	trainCount = 0;
	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			if (GTBuffer[Limits[0][patID]+index] > 0)
			{
				selForTrain[trainCount] = Limits[0][patID] + index;
				++count;
				++trainCount;
			}
		}
	}

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		trainLabels(o,0) = (GTBuffer[selForTrain[o]] > 1 ? 1 : 0);
	}

	Ptr<ml::Boost> ada2 = ml::Boost::create();
	ada2->setMaxDepth(depilimit);
	startTime();
	ada2->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
	lapseTrain += endTime();
	printf("Training duration total: %.3f seconds.\n", (float)lapseTrain * 0.001);

	ConfusionMatrix kungfu2(NR_TISSUES);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	pixelsToTest = Limits[2][testBaby];
	firstTestPixel = Limits[0][testBaby];
	nextTestPixel = Limits[0][testBaby];

	startTime();
	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		ada2->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci = 0;
			if (parti[pix-firstTestPixel]>0)
			{
				deci = response.at<float>(pix-nextTestPixel, 0);
				if (deci>0) deci=2; else deci=1;
				parti[pix-firstTestPixel]=deci;
			}
			int gt = GTBuffer[pix];
			kungfu2.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	lapseTest += endTime();
	printf("\nTesting duration total: %.3f seconds.\n", (float)lapseTest * 0.001);
	kungfu2.compute();
}

void OcvLogReg(int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES)
{
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int depilimit = maxDepth;
	if (depilimit < 5) depilimit = 5;
	if (depilimit > 50) depilimit = 50;

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;

	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<float> trainLabels(nrTrainData, 1);

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		trainLabels(o,0) = (GTBuffer[selForTrain[o]] > 0 ? 1.0f : 0.0f);
	}

	Ptr<ml::LogisticRegression> logreg = ml::LogisticRegression::create();
	logreg->setLearningRate(0.1);
    logreg->setIterations(1000);
    logreg->setRegularization(LogisticRegression::REG_L2);
    logreg->setTrainMethod(LogisticRegression::BATCH);
    logreg->setMiniBatchSize(10);
	logreg->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
	ConfusionMatrix kungfu(2);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	int pixelsToTest = Limits[2][testBaby];
	int firstTestPixel = Limits[0][testBaby];
	int nextTestPixel = Limits[0][testBaby];
	// partition
	uchar* parti = (uchar*)calloc(pixelsToTest,1);

	int xsum = 0;
	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		logreg->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci = response.at<int>(pix-nextTestPixel, 0);
			xsum += deci;
			//if (deci>0) deci=1;
			parti[pix-firstTestPixel]=deci;
			int gt = GTBuffer[pix];
			if (gt>0) gt=1;
			kungfu.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	kungfu.compute();

	// 2nd stage
	trainCount = 0;
	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			if (GTBuffer[Limits[0][patID]+index] > 0)
			{
				selForTrain[trainCount] = Limits[0][patID] + index;
				++count;
				++trainCount;
			}
		}
	}

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		trainLabels(o,0) = (GTBuffer[selForTrain[o]] > 1 ? 1 : 0);
	}

	Ptr<ml::LogisticRegression> logreg2 = ml::LogisticRegression::create();
	logreg2->setLearningRate(0.1);
    logreg2->setIterations(1000);
    logreg2->setRegularization(LogisticRegression::REG_L2);
    logreg2->setTrainMethod(LogisticRegression::BATCH);
    logreg2->setMiniBatchSize(10);
	logreg2->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);

	ConfusionMatrix kungfu2(NR_TISSUES);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	pixelsToTest = Limits[2][testBaby];
	firstTestPixel = Limits[0][testBaby];
	nextTestPixel = Limits[0][testBaby];

	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		logreg2->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci = 0;
			if (parti[pix-firstTestPixel]>0)
			{
				deci = response.at<int>(pix-nextTestPixel, 0);
				if (deci>0) deci=2; else deci=1;
				parti[pix-firstTestPixel]=deci;
			}
			int gt = GTBuffer[pix];
			kungfu2.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	kungfu2.compute();
}


void OcvTwoStage(int alg, int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES)
{
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int depilimit = maxDepth;
	if (depilimit < 5) depilimit = 5;
	if (depilimit > 50) depilimit = 50;

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;

	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<int> trainLabels(nrTrainData, 1);
	Mat_<float> fTrainLabels(nrTrainData, 1);

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
//		trainLabels(o,0) = (int)(GTBuffer[selForTrain[o]]);
		trainLabels(o,0) = (GTBuffer[selForTrain[o]] > 0 ? 1 : 0);
		fTrainLabels(o,0) = (GTBuffer[selForTrain[o]] > 0 ? 1 : 0);
	}

	Ptr<ml::Boost> ada = ml::Boost::create();
	Ptr<ml::LogisticRegression> logreg = ml::LogisticRegression::create();

	switch (alg)
	{
	case ALG_ADA:
			ada->setMaxDepth(depilimit);
			ada->setBoostType(ml::Boost::DISCRETE);
//			ada->setBoostType(ml::Boost::REAL);
//			ada->setBoostType(ml::Boost::LOGIT);
//			ada->setBoostType(ml::Boost::GENTLE);
			startTime();
			ada->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		break;
	case ALG_LOGREG:
			logreg->setLearningRate(1);
			logreg->setIterations(1000);
			logreg->setRegularization(LogisticRegression::REG_L2);
			logreg->setTrainMethod(LogisticRegression::BATCH);
			logreg->setMiniBatchSize(10);
			startTime();
			logreg->train(trainFeatures, ml::ROW_SAMPLE, fTrainLabels);
		break;
	}
	int lapseTrain = endTime();
	printf("Training duration so far: %.3f seconds.\n", (float)lapseTrain * 0.001);
	ConfusionMatrix kungfu(2);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	int pixelsToTest = Limits[2][testBaby];
	int firstTestPixel = Limits[0][testBaby];
	int nextTestPixel = Limits[0][testBaby];
	// partition
	uchar* parti = (uchar*)calloc(pixelsToTest,1);

	startTime();
	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		if (alg==ALG_ADA)
			ada->predict(testFeatures, response);
		else if (alg==ALG_LOGREG)
			logreg->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci;
			if (alg==ALG_ADA)
				deci = response.at<float>(pix-nextTestPixel, 0);
			else if (alg==ALG_LOGREG)
				deci = response.at<int>(pix-nextTestPixel, 0);
			if (deci>0) deci=1;
			parti[pix-firstTestPixel]=deci;
			int gt = GTBuffer[pix];
			if (gt>0) gt=1;
			kungfu.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	int lapseTest = endTime();
	printf("\nTesting duration so far: %.3f seconds.\n", (float)lapseTest * 0.001);
	kungfu.compute();

	// 2nd stage
	trainCount = 0;
	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			if (GTBuffer[Limits[0][patID]+index] > 0)
			{
				selForTrain[trainCount] = Limits[0][patID] + index;
				++count;
				++trainCount;
			}
		}
	}

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		trainLabels(o,0) = (GTBuffer[selForTrain[o]] > 1 ? 1 : 0);
		fTrainLabels(o,0) = (GTBuffer[selForTrain[o]] > 1 ? 1 : 0);
	}

	Ptr<ml::Boost> ada2 = ml::Boost::create();
	Ptr<ml::LogisticRegression> logreg2 = ml::LogisticRegression::create();

	switch (alg)
	{
	case ALG_ADA:
			ada2->setMaxDepth(depilimit);
			ada2->setBoostType(ml::Boost::DISCRETE);
//			ada2->setBoostType(ml::Boost::REAL);
//			ada2->setBoostType(ml::Boost::LOGIT);
//			ada2->setBoostType(ml::Boost::GENTLE);
			startTime();
			ada2->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		break;
	case ALG_LOGREG:
			logreg2->setLearningRate(1);
			logreg2->setIterations(1000);
			logreg2->setRegularization(LogisticRegression::REG_L2);
			logreg2->setTrainMethod(LogisticRegression::BATCH);
			logreg2->setMiniBatchSize(10);
			startTime();
			logreg2->train(trainFeatures, ml::ROW_SAMPLE, fTrainLabels);
		break;
	}
	lapseTrain += endTime();
	printf("Training duration total: %.3f seconds.\n", (float)lapseTrain * 0.001);

	ConfusionMatrix kungfu2(NR_TISSUES);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	pixelsToTest = Limits[2][testBaby];
	firstTestPixel = Limits[0][testBaby];
	nextTestPixel = Limits[0][testBaby];

	startTime();
	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		if (alg==ALG_ADA)
			ada2->predict(testFeatures, response);
		else if (alg==ALG_LOGREG)
			logreg2->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci=0;
			if (parti[pix-firstTestPixel]>0)
			{
				if (alg==ALG_ADA)
					deci = response.at<float>(pix-nextTestPixel, 0);
				else if (alg==ALG_LOGREG)
					deci = response.at<int>(pix-nextTestPixel, 0);
				if (deci>0) deci=2; else deci=1;
				parti[pix-firstTestPixel]=deci;
			}
			int gt = GTBuffer[pix];
			kungfu2.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	lapseTest += endTime();
	printf("\nTesting duration total: %.3f seconds.\n", (float)lapseTest * 0.001);
	FILE* G = fopen("kingkong.csv", "at");
	fprintf(G, "%d,%d,%d,%d,%d,%d,%d,", alg, testBaby, nrTrainPixPerTrainBaby, maxDepth, USED_FEATURES, lapseTrain, lapseTest);
	fclose(G);
	kungfu2.compute();
}


void main()
{
//	Szabolcs1();
//	getch();
//	Watershed();

	srand(time(0));
//	LoadDataFromImages();
	LoadFeatures();
//	DrawSliceFeatures(1,46);
//	OwnKNN(7,100,9,4);
//	OcvKNN(7,100,21,4);
//	OcvBoost(7,5000,12,21);
//	OcvTwoStage(ALG_ADA,7,5000,12,21);
//	for (int patID = 0; patID < MAX_PATID; ++patID)
		OcvALL(ALG_RF, 7, 100000, 26, 21,1);
		//OcvTwoStage(ALG_LOGREG, patID, 5000, 0, 21);
	//	OcvLogReg(7,5000,0,21);
//	for (int patID=0; patID<MAX_PATID; ++patID)
//		OcvDecTree(patID,1000,12,18);
//		OcvDecTree(7,30000,9,21,1);
/*	for (int patID=0; patID<MAX_PATID; ++patID)
		OcvRF(patID,1000,21,18);*/
//		OcvRF(7,100000,8,18);
/*	for (int patID=0; patID<MAX_PATID; ++patID)
		OcvANN(patID,1000,21,18);*/
//	OcvANN(7,500,21,18);
//	ReleaseBuffers();
	getch();
}

/***********************************/
void x01()
{
	IplImage* im;
	char fname[100];
	unsigned char* bufferT1 = (unsigned char*)malloc(WIDTH*HEIGHT*MAX_SLICES*2);
	unsigned char* bufferT2 = (unsigned char*)malloc(WIDTH*HEIGHT*MAX_SLICES*2);
	unsigned char* bufferGT = (unsigned char*)malloc(WIDTH*HEIGHT*MAX_SLICES);
	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		int indexT1 = 0;
		int indexT2 = 0;
		int indexGT = 0;
		int fcount = 0;
		for (int i=0; i<MAX_SLICES; ++i)
		{
			sprintf(fname,"babainput/subject-%d/t1/subject-%d-T1%04d.tif",patID+1,patID+1,i);
			FILE* F1 = fopen(fname,"rb");
			if (F1) 
			{
				++fcount;
				fseek(F1,295,SEEK_SET);
				int res = fread(bufferT1+indexT1,1,WIDTH*HEIGHT*2,F1);
				indexT1 += res;
				fclose(F1);
			}
			sprintf(fname,"babainput/subject-%d/t2/subject-%d-T2%04d.tif",patID+1,patID+1,i);
			FILE* F2 = fopen(fname,"rb");
			if (F2) 
			{
				++fcount;
				fseek(F2,295,SEEK_SET);
				int res = fread(bufferT2+indexT2,1,WIDTH*HEIGHT*2,F2);
				indexT2 += res;
				fclose(F2);
			}
			sprintf(fname,"babainput/subject-%d/label/subject-%d-label%04d.tif",patID+1,patID+1,i);
			FILE* F3 = fopen(fname,"rb");
			if (F3) 
			{
				++fcount;
				fseek(F3,230,SEEK_SET);
				int res = fread(bufferGT+indexGT,1,WIDTH*HEIGHT,F3);
				indexGT += res;
				fclose(F3);
			}
		}
		printf("Baby no. %d, %d files (%d %d %d).\n",patID,fcount,indexT1,indexT2,indexGT);
		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* F = fopen(fname,"wb");
		fwrite(bufferT1,1,indexT1,F);
		fwrite(bufferT2,1,indexT2,F);
		fwrite(bufferGT,1,indexGT,F);
		fclose(F);
	}
	printf("Done.\n");
	getch();
	free(bufferT1);
	free(bufferT2);
	free(bufferGT);
}

void x02()
{
	int hist[20000]={0};
	char fname[100];
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);
	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* F = fopen(fname,"rb");
		fread(bufferT1,2,VOLSIZE,F);
		fread(bufferT2,2,VOLSIZE,F);
		fread(bufferGT,1,VOLSIZE,F);
		fclose(F);

		int count = 0;
		int count1 = 0;
		int count2 = 0;
		int maxT1 = 0;
		int maxT2 = 0;
		int minT1 = 9999;
		int minT2 = 9999;
		int sumT1 = 0;
		int sumT2 = 0;
/*		for (int i=0; i<VOLSIZE; ++i)
		{
			bufferT1[i] = ((bufferT1[i]%256)*256 + (bufferT1[i]/256))%32768;
			bufferT2[i] = ((bufferT2[i]%256)*256 + (bufferT2[i]/256))%32768;
		}*/
		int first = 0;
		int last = 0;
		for (int i=0; i<VOLSIZE; ++i)
		{
			if (bufferGT[i]>0)
			{
				if (!first) first = i;
				last = i;
				++count;
				if (bufferT1[i]>0) 
				{
					++count1;
					if (maxT1 < bufferT1[i]) maxT1 = bufferT1[i];
					if (minT1 > bufferT1[i]) minT1 = bufferT1[i];
					sumT1 += bufferT1[i];
					hist[patID*2000+bufferT1[i]-1]++;
				}
				if (bufferT2[i]>0) 
				{
					++count2;
					if (maxT2 < bufferT2[i]) maxT2 = bufferT2[i];
					if (minT2 > bufferT2[i]) minT2 = bufferT2[i];
					sumT2 += bufferT2[i];
					hist[patID*2000+1000+bufferT2[i]-1]++;
				}
			}
		}
		printf("Baba %d, [%d %d %d] [%d %d %d] [%d %d %d] <%d %d>\n",patID,count,count1,count2,minT1,sumT1/count1,maxT1,minT2,sumT2/count2,maxT2,first/(WIDTH*HEIGHT),last/(WIDTH*HEIGHT));
		
		F = fopen(fname,"wb");
		fwrite(bufferT1,1,VOLSIZE*2,F);
		fwrite(bufferT2,1,VOLSIZE*2,F);
		fwrite(bufferGT,1,VOLSIZE,F);
		fclose(F);
	}

	FILE* F = fopen("babainput/orighist.hst","wb");
	fwrite(hist,4,20000,F);
	fclose(F);

	getch();
	free(bufferT1);
	free(bufferT2);
	free(bufferGT);

}

/*
void main()
{
	x02();
}
*/

void x03()
{
	int hist[20000]={0};
	int coltrans[20000]={0};
	unsigned short* flat = (unsigned short*)malloc(2000000);

	FILE* F = fopen("babainput/orighist.hst","rb");
	fread(hist,4,20000,F);
	fclose(F);

	FILE* G = fopen("histstat.csv","wt");

	for (int offset = 0; offset < 1111; offset+=1000)
	{
		for (int patID=0; patID<MAX_PATID; ++patID)
		{
			int index = 0;
			for (int h=0; h<1000; ++h)
			{
				int o = hist[offset+2000*patID + h];
				for (int i=0; i<o; ++i)
					flat[index++] = h+1;
			}
			float p25 = flat[(index-1)/4];
			float p75 = flat[(3*(index-1))/4];
			float a = 0.2f / (p75-p25);
			float b = 0.4f - a*p25;

			for (int h=0; h<1000; ++h)
			{
				float v = a*((float)h+1.0f) + b;
				if (v<0.0f) v=0.0f;
				if (v>1.0f) v=1.0f;

				coltrans[offset+2000*patID + h] = cvRound(14.0f * v) + 1;
			}

			printf("%d (x -> %.6fx + %.6f)\n",index,a,b);

			for (int j=0; j<=100; ++j)
				fprintf(G,"%d,",flat[(index-1)*j/100]);
			fprintf(G,"\n");
		}
	}
	fclose(G);
	getch();
	free(flat);
	F = fopen("babainput/coltrans4.hst","wb");
	fwrite(coltrans,4,20000,F);
	fclose(F);
}

void x04()
{
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	char fname[100];
	sprintf(fname,"babainput/baba-%d.bab",0);
	FILE* G = fopen(fname,"rb");
	fread(bufferT1,2,VOLSIZE,G);
	fread(bufferT2,2,VOLSIZE,G);
	fread(bufferGT,1,VOLSIZE,G);
	fclose(G);

	int coltrans[20000]={0};
	IplImage* im7 = cvCreateImage(cvSize(7*WIDTH,HEIGHT),IPL_DEPTH_8U,1);
	cvSet(im7,cvScalar(0));
	for (int r=0; r<7; ++r)
	{	
		sprintf(fname,"babainput/coltrans%d.hst",r+4);
		FILE* F = fopen(fname,"rb");
		fread(coltrans,4,20000,F);
		fclose(F);
	
		int s = 120;
		for (int x=0; x<WIDTH; ++x) for (int y=0; y<HEIGHT; ++y)
		{
			int index = s*WIDTH*HEIGHT + y*WIDTH + x;
			if (bufferGT[index] > 0)
			{
				int o = bufferT1[index];
				o = coltrans[o];
				int u = cvRound(1.0 + 254.0f * (((float)o-1.0f)/((float)(coltrans[999]-coltrans[0]))));
				setGray(im7,r*WIDTH+x,y,u);		
			}
		}
	}
	cvShowImage("seven",im7);
	cvWaitKey();
	free(bufferT1);
	free(bufferT2);
	free(bufferGT);
}


void x05()
{
	int hist[20000]={0};
	float coltrans[20000]={0};
	float nyuszitrans[20000]={0};
	unsigned short* flat = (unsigned short*)malloc(2000000);
	float* fflat = (float*)malloc(4000000);
	float miles[7] = {0};

	FILE* F = fopen("babainput/orighist.hst","rb");
	fread(hist,4,20000,F);
	fclose(F);


	for (int offset = 0; offset < 1111; offset+=1000)
	{
		for (int patID=0; patID<MAX_PATID; ++patID)
		{
			int index = 0;
			for (int h=0; h<1000; ++h)
			{
				int o = hist[offset+2000*patID + h];
				for (int i=0; i<o; ++i)
					flat[index++] = h+1;
			}
			int p01 = flat[(index-1)/100];
			int p99 = flat[(99*(index-1))/100];
			printf("<%d,%d>",p01,p99);
			for (int h=0; h<1000; ++h)
			{
				float v;
				if (h<=p01) v=0.0f;
				else if (h>=p99) v=1.0f;
				else v=((float)(h-p01))/((float)(p99-p01));

				coltrans[offset+2000*patID + h] = v;
			}

		}

		printf("\n\n");

		for (int o=0; o<7; ++o) miles[o] = 0.0f;

		for (int patID=0; patID<MAX_PATID; ++patID)
		{
			int index = 0;
			for (int h=0; h<1000; ++h)
			{
				int o = hist[offset+2000*patID + h];
				for (int i=0; i<o; ++i)
					fflat[index++] = coltrans[offset+2000*patID + h];
			}
			float p01 = 0.0f;//fflat[(index-1)/100];
			float p10 = fflat[(10*(index-1))/100];
			float p25 = fflat[(25*(index-1))/100];
			float p50 = fflat[(50*(index-1))/100];
			float p75 = fflat[(75*(index-1))/100];
			float p90 = fflat[(90*(index-1))/100];
			float p99 = 1.0f;//fflat[(99*(index-1))/100];
			printf("<%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f>\n",p01,p10,p25,p50,p75,p90,p99);
			miles[0] += p01;
			miles[1] += p10;
			miles[2] += p25;
			miles[3] += p50;
			miles[4] += p75;
			miles[5] += p90;
			miles[6] += p99;
		}
		for (int o=0; o<7; ++o) miles[o] *= 0.1f;
		printf("[[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]]\n",miles[0],miles[1],miles[2],miles[3],miles[4],miles[5],miles[6]);

		for (int patID=0; patID<MAX_PATID; ++patID)
		{
			int index = 0;
			for (int h=0; h<1000; ++h)
			{
				int o = hist[offset+2000*patID + h];
				for (int i=0; i<o; ++i)
					flat[index++] = h+1;
			}
			int p01 = flat[(index-1)/100];
			int p10 = flat[(10*(index-1))/100];
			int p25 = flat[(25*(index-1))/100];
			int p50 = flat[(50*(index-1))/100];
			int p75 = flat[(75*(index-1))/100];
			int p90 = flat[(90*(index-1))/100];
			int p99 = flat[(99*(index-1))/100];

			for (int o=1; o<=p01; ++o)   nyuszitrans[offset+2000*patID + (o-1)] = miles[0];
			for (int o=p01; o<=p10; ++o) nyuszitrans[offset+2000*patID + (o-1)] = miles[0] + (miles[1]-miles[0])*(((float)(o-p01))/((float)(p10-p01)));
			for (int o=p10; o<=p25; ++o) nyuszitrans[offset+2000*patID + (o-1)] = miles[1] + (miles[2]-miles[1])*(((float)(o-p10))/((float)(p25-p10)));
			for (int o=p25; o<=p50; ++o) nyuszitrans[offset+2000*patID + (o-1)] = miles[2] + (miles[3]-miles[2])*(((float)(o-p25))/((float)(p50-p25)));
			for (int o=p50; o<=p75; ++o) nyuszitrans[offset+2000*patID + (o-1)] = miles[3] + (miles[4]-miles[3])*(((float)(o-p50))/((float)(p75-p50)));
			for (int o=p75; o<=p90; ++o) nyuszitrans[offset+2000*patID + (o-1)] = miles[4] + (miles[5]-miles[4])*(((float)(o-p75))/((float)(p90-p75)));
			for (int o=p90; o<=p99; ++o) nyuszitrans[offset+2000*patID + (o-1)] = miles[5] + (miles[6]-miles[5])*(((float)(o-p90))/((float)(p99-p90)));
			for (int o=p99; o<=1000; ++o)nyuszitrans[offset+2000*patID + (o-1)] = miles[6];
		}
	}

	F = fopen("babainput/roka.hst","wb");
	fwrite(coltrans,4,20000,F);
	fclose(F);
	F = fopen("babainput/nyuszi.hst","wb");
	fwrite(nyuszitrans,4,20000,F);
	fclose(F);

	getch();
	free(flat);
}

void x06()
{
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	char fname[100];
	sprintf(fname,"babainput/baba-%d.bab",0);
	FILE* G = fopen(fname,"rb");
	fread(bufferT1,2,VOLSIZE,G);
	fread(bufferT2,2,VOLSIZE,G);
	fread(bufferGT,1,VOLSIZE,G);
	fclose(G);

	float nyuszitrans[20000]={0};
	IplImage* im7 = cvCreateImage(cvSize(7*WIDTH,HEIGHT),IPL_DEPTH_8U,1);
	cvSet(im7,cvScalar(0));

	sprintf(fname,"babainput/nyuszi.hst");
	FILE* F = fopen(fname,"rb");
	fread(nyuszitrans,4,20000,F);
	fclose(F);
	
	int rr = 14;
	for (int r=0; r<7; ++r)
	{	
		int s = 120;
		for (int x=0; x<WIDTH; ++x) for (int y=0; y<HEIGHT; ++y)
		{
			int index = s*WIDTH*HEIGHT + y*WIDTH + x;
			if (bufferGT[index] > 0)
			{
				int o = bufferT1[index];
				float of = nyuszitrans[o];
				int z = 1 + cvRound((float)rr * of); 
				int u = cvRound(1.0 + 254.0f * (((float)z-1.0f)/((float)rr)));
				setGray(im7,r*WIDTH+x,y,u);		
			}
		}
		rr = 2*rr + 2;
	}
	cvShowImage("seven_nyuszi",im7);
	cvWaitKey();
	free(bufferT1);
	free(bufferT2);
	free(bufferGT);
}

void x07()  // nyuszi
{
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	char fname[100];

	float nyuszitrans[20000]={0};
	sprintf(fname,"babainput/nyuszi.hst");
	FILE* F = fopen(fname,"rb");
	fread(nyuszitrans,4,20000,F);
	fclose(F);

	for (int patID=0; patID < MAX_PATID; ++patID)
	{
		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* G = fopen(fname,"rb");
		fread(bufferT1,2,VOLSIZE,G);
		fread(bufferT2,2,VOLSIZE,G);
		fread(bufferGT,1,VOLSIZE,G);
		fclose(G);

		IplImage* imT1 = cvCreateImage(cvSize(16*WIDTH,7*HEIGHT),IPL_DEPTH_8U,1);
		cvSet(imT1,cvScalar(0));
		IplImage* imT2 = cvCloneImage(imT1);
		IplImage* imGT = cvCloneImage(imT1);
		IplImage* imM = cvCloneImage(imT1);

	
		int rr = 254;
		for (int s=FIRSTSLICES[patID]; s<FIRSTSLICES[patID]+112; ++s)
		{	
			for (int x=0; x<WIDTH; ++x) for (int y=0; y<HEIGHT; ++y)
			{
				int index = s*WIDTH*HEIGHT + y*WIDTH + x;
				if (bufferGT[index] > 0)
				{
					{
						int o = bufferT1[index];
						if (o==0) o=333;
						float of = nyuszitrans[2000*patID + (o-1)];
						int z = 1 + cvRound((float)rr * of); 
						int u = cvRound(1.0 + 254.0f * (((float)z-1.0f)/((float)rr)));
						setGray(imT1,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,u);	
					}
					{
						int o = bufferT2[index];
						if (o==0) o=333;
						float of = nyuszitrans[2000*patID + 1000 + (o-1)];
						int z = 1 + cvRound((float)rr * of); 
						int u = cvRound(1.0 + 254.0f * (((float)z-1.0f)/((float)rr)));
						setGray(imT2,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,u);	
					}
					if (bufferGT[index] > 200)
						setGray(imGT,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,250);	
					else if (bufferGT[index] > 100)
						setGray(imGT,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,150);	
					else 
						setGray(imGT,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,50);	
				}
			}
		}
		
		sprintf(fname,"nyuszi/baba-%d-T1.png",patID);
		cvSaveImage(fname,imT1);
		sprintf(fname,"nyuszi/baba-%d-T2.png",patID);
		cvSaveImage(fname,imT2);
		sprintf(fname,"nyuszi/baba-%d-GT.png",patID);
		cvSaveImage(fname,imGT);

		float agi[6];
		CvScalar avg, sdv;
		cvCmpS(imGT,50,imM,CV_CMP_EQ);
		cvAvgSdv(imT1,&avg,&sdv,imM);
		agi[0] = avg.val[0];
		cvAvgSdv(imT2,&avg,&sdv,imM);
		agi[3] = avg.val[0];
		cvCmpS(imGT,150,imM,CV_CMP_EQ);
		cvAvgSdv(imT1,&avg,&sdv,imM);
		agi[1] = avg.val[0];
		cvAvgSdv(imT2,&avg,&sdv,imM);
		agi[4] = avg.val[0];
		cvCmpS(imGT,250,imM,CV_CMP_EQ);
		cvAvgSdv(imT1,&avg,&sdv,imM);
		agi[2] = avg.val[0];
		cvAvgSdv(imT2,&avg,&sdv,imM);
		agi[5] = avg.val[0];

		printf("<%.3f %.3f %.3f> <%.3f %.3f %.3f>\n",agi[0],agi[1],agi[2],agi[3],agi[4],agi[5]);
		cvReleaseImage(&imT1);
		cvReleaseImage(&imT2);
		cvReleaseImage(&imGT);
	}
	cvWaitKey();
	free(bufferT1);
	free(bufferT2);
	free(bufferGT);
	getch();
}


void x08()  // CDLT
{
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	char fname[100];

	int coltrans[20000]={0};
	sprintf(fname,"babainput/coltrans8.hst");
	FILE* F = fopen(fname,"rb");
	fread(coltrans,4,20000,F);
	fclose(F);

	for (int patID=0; patID < MAX_PATID; ++patID)
	{
		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* G = fopen(fname,"rb");
		fread(bufferT1,2,VOLSIZE,G);
		fread(bufferT2,2,VOLSIZE,G);
		fread(bufferGT,1,VOLSIZE,G);
		fclose(G);

		IplImage* imT1 = cvCreateImage(cvSize(16*WIDTH,7*HEIGHT),IPL_DEPTH_8U,1);
		cvSet(imT1,cvScalar(0));
		IplImage* imT2 = cvCloneImage(imT1);
		IplImage* imGT = cvCloneImage(imT1);
		IplImage* imM = cvCloneImage(imT1);

	
		int rr = 254;
		for (int s=FIRSTSLICES[patID]; s<FIRSTSLICES[patID]+112; ++s)
		{	
			for (int x=0; x<WIDTH; ++x) for (int y=0; y<HEIGHT; ++y)
			{
				int index = s*WIDTH*HEIGHT + y*WIDTH + x;
				if (bufferGT[index] > 0)
				{
					{
						int o = bufferT1[index];
						if (o==0) o=333;
						int z = coltrans[2000*patID + (o-1)];
						setGray(imT1,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,z);	
					}
					{
						int o = bufferT2[index];
						if (o==0) o=333;
						int z = coltrans[2000*patID + 1000 + (o-1)];
						setGray(imT2,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,z);	
					}
					if (bufferGT[index] > 200)
						setGray(imGT,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,250);	
					else if (bufferGT[index] > 100)
						setGray(imGT,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,150);	
					else 
						setGray(imGT,((s-FIRSTSLICES[patID])%16)*WIDTH+x,((s-FIRSTSLICES[patID])/16)*HEIGHT+y,50);	
				}
			}
		}
		
		sprintf(fname,"cdlt/baba-%d-T1.png",patID);
		cvSaveImage(fname,imT1);
		sprintf(fname,"cdlt/baba-%d-T2.png",patID);
		cvSaveImage(fname,imT2);
		sprintf(fname,"cdlt/baba-%d-GT.png",patID);
		cvSaveImage(fname,imGT);

		float agi[6];
		CvScalar avg, sdv;
		cvCmpS(imGT,50,imM,CV_CMP_EQ);
		cvAvgSdv(imT1,&avg,&sdv,imM);
		agi[0] = avg.val[0];
		cvAvgSdv(imT2,&avg,&sdv,imM);
		agi[3] = avg.val[0];
		cvCmpS(imGT,150,imM,CV_CMP_EQ);
		cvAvgSdv(imT1,&avg,&sdv,imM);
		agi[1] = avg.val[0];
		cvAvgSdv(imT2,&avg,&sdv,imM);
		agi[4] = avg.val[0];
		cvCmpS(imGT,250,imM,CV_CMP_EQ);
		cvAvgSdv(imT1,&avg,&sdv,imM);
		agi[2] = avg.val[0];
		cvAvgSdv(imT2,&avg,&sdv,imM);
		agi[5] = avg.val[0];

		printf("<%.3f %.3f %.3f> <%.3f %.3f %.3f>\n",agi[0],agi[1],agi[2],agi[3],agi[4],agi[5]);
		cvReleaseImage(&imT1);
		cvReleaseImage(&imT2);
		cvReleaseImage(&imGT);
	}
	cvWaitKey();
	free(bufferT1);
	free(bufferT2);
	free(bufferGT);
	getch();
}

void x09()  // CDLT
{
	char fname[100];

	for (int patID=0; patID < MAX_PATID; ++patID)
	{
		sprintf(fname,"cdlt/baba-%d-T1.png",patID);
		IplImage* imT1 = cvLoadImage(fname,0);
		sprintf(fname,"cdlt/baba-%d-T2.png",patID);
		IplImage* imT2 = cvLoadImage(fname,0);
		sprintf(fname,"cdlt/baba-%d-GT.png",patID);
		IplImage* imGT = cvLoadImage(fname,0);
		IplImage* imRe = cvCloneImage(imGT);
		IplImage* imMin = cvCloneImage(imGT);
		IplImage* imMax = cvCloneImage(imGT);
	
		for (int neigh=1; neigh<=5; ++neigh)
		{
			cvSet(imRe,cvScalar(0));
			for (int x=0; x<imGT->width; ++x) for (int y=0; y<imGT->height; ++y)
			{
				if (getGray(imGT,x,y) > 0)
				{
					int count = 0;
					int sum = 0;
					for (int ix=x-neigh; ix<=x+neigh; ++ix) for (int iy=y-neigh; iy<=y+neigh; ++iy)
					{
						if (getGray(imGT,ix,iy) > 0)
						{
							++count;
							sum+=getGray(imT1,ix,iy);
						}
					}
					sum = (sum + count/2) / count;
					setGray(imRe,x,y,sum);
				}
			}
			sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,2*neigh);
			cvSaveImage(fname,imRe);
		}
		printf(".");
		for (int neigh=1; neigh<=5; ++neigh)
		{
			cvSet(imRe,cvScalar(0));
			for (int x=0; x<imGT->width; ++x) for (int y=0; y<imGT->height; ++y)
			{
				if (getGray(imGT,x,y) > 0)
				{
					int count = 0;
					int sum = 0;
					for (int ix=x-neigh; ix<=x+neigh; ++ix) for (int iy=y-neigh; iy<=y+neigh; ++iy)
					{
						if (getGray(imGT,ix,iy) > 0)
						{
							++count;
							sum+=getGray(imT2,ix,iy);
						}
					}
					sum = (sum + count/2) / count;
					setGray(imRe,x,y,sum);
				}
			}
			sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,2*neigh+1);
			cvSaveImage(fname,imRe);
		}


		int ooo=0;
		cvSet(imRe,cvScalar(0));
		cvSet(imMin,cvScalar(0));
		cvSet(imMax,cvScalar(0));
		for (int x=0; x<imGT->width; ++x) for (int y=0; y<imGT->height; ++y)
		{
			if (getGray(imGT,x,y) > 0)
			{
				++ooo;
				int mini = 255;
				int maxi = 0;
				int count = 0;
				int sum = 0;

				int X = x % WIDTH;
				int Y = y % HEIGHT;
				int Z = (y / HEIGHT)*16 + (x / WIDTH);

				for (int ix=X-1; ix<=X+1; ++ix) for (int iy=Y-1; iy<=Y+1; ++iy) for (int iz=Z-1; iz<=Z+1; ++iz)
				{
					if (iz>=0)
					{
						int xx = (iz % 16) * WIDTH + ix;
						int yy = (iz / 16) * HEIGHT + iy;
						if (getGray(imGT,xx,yy) > 0)
						{
							int q = getGray(imT1,xx,yy);
							++count;
							sum+=q;
							if (mini > q) mini=q;
							if (maxi < q) maxi=q;
						}
					}
				}
				sum = (sum + count/2) / count;
				setGray(imRe,x,y,sum);
				setGray(imMin,x,y,mini);
				setGray(imMax,x,y,maxi);
			}
		}
		printf("<%d>",ooo);
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,12);
		cvSaveImage(fname,imRe);
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,14);
		cvSaveImage(fname,imMin);
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,16);
		cvSaveImage(fname,imMax);

		cvSet(imRe,cvScalar(0));
		cvSet(imMin,cvScalar(0));
		cvSet(imMax,cvScalar(0));
		for (int x=0; x<imGT->width; ++x) for (int y=0; y<imGT->height; ++y)
		{
			if (getGray(imGT,x,y) > 0)
			{
				int mini = 255;
				int maxi = 0;
				int count = 0;
				int sum = 0;

				int X = x % WIDTH;
				int Y = y % HEIGHT;
				int Z = (y / HEIGHT)*16 + (x / WIDTH);

				for (int ix=X-1; ix<=X+1; ++ix) for (int iy=Y-1; iy<=Y+1; ++iy) for (int iz=Z-1; iz<=Z+1; ++iz)
				{
					if (iz>=0)
					{
						int xx = (iz % 16) * WIDTH + ix;
						int yy = (iz / 16) * HEIGHT + iy;
						if (getGray(imGT,xx,yy) > 0)
						{
							int q = getGray(imT2,xx,yy);
							++count;
							sum+=q;
							if (mini > q) mini=q;
							if (maxi < q) maxi=q;
						}
					}
				}
				sum = (sum + count/2) / count;
				setGray(imRe,x,y,sum);
				setGray(imMin,x,y,mini);
				setGray(imMax,x,y,maxi);
			}
		}
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,13);
		cvSaveImage(fname,imRe);
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,15);
		cvSaveImage(fname,imMin);
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,17);
		cvSaveImage(fname,imMax);

		printf("@");

		cvSet(imRe,cvScalar(0));
		cvSet(imMin,cvScalar(0));
		cvSet(imMax,cvScalar(0));
		int xMin = WIDTH;
		int xMax = 0;
		int yMin = HEIGHT;
		int yMax = 0;
		int zMin = MAX_SLICES;
		int zMax = 0;
		int xSum = 0;
		int ySum = 0;
		int zSum = 0;
		int count = 0;
		for (int x=0; x<imGT->width; ++x) for (int y=0; y<imGT->height; ++y)
		{
			if (getGray(imGT,x,y) > 0)
			{
				++count;
				int X = x % WIDTH;
				int Y = y % HEIGHT;
				int Z = (y / HEIGHT)*16 + (x / WIDTH);
				if (X > xMax) xMax = X;
				if (Y > yMax) yMax = Y;
				if (Z > zMax) zMax = Z;
				if (X < xMin) xMin = X;
				if (Y < yMin) yMin = Y;
				if (Z < zMin) zMin = Z;
				xSum += X;
				ySum += Y;
				zSum += Z;
			}
		}
		int xAvg = (xSum + count/2) / count;
		int yAvg = (ySum + count/2) / count;
		int zAvg = (zSum + count/2) / count;

		for (int x=0; x<imGT->width; ++x) for (int y=0; y<imGT->height; ++y)
		{
			if (getGray(imGT,x,y) > 0)
			{
				int X = x % WIDTH;
				int Y = y % HEIGHT;
				int Z = (y / HEIGHT)*16 + (x / WIDTH);
				if (Z < zAvg)
					setGray(imRe,x,y,1.0f + 127.0f * ((float)(Z-zMin))/((float)(zAvg-zMin)));
				else
					setGray(imRe,x,y,128.0f + 127.0f * ((float)(Z-zAvg))/((float)(zMax-zAvg)));
				if (Y < yAvg)
					setGray(imMax,x,y,1.0f + 127.0f * ((float)(Y-yMin))/((float)(yAvg-yMin)));
				else
					setGray(imMax,x,y,128.0f + 127.0f * ((float)(Y-yAvg))/((float)(yMax-yAvg)));
				if (X < xAvg)
					setGray(imMin,x,y,1.0f + 127.0f * ((float)(X-xMin))/((float)(xAvg-xMin)));
				else
					setGray(imMin,x,y,128.0f + 127.0f * ((float)(X-xAvg))/((float)(xMax-xAvg)));
			}
		}
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,18);
		cvSaveImage(fname,imRe);
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,19);
		cvSaveImage(fname,imMin);
		sprintf(fname,"cdlt/baba-%d-ch%02d.png",patID,20);
		cvSaveImage(fname,imMax);

		sprintf(fname,"cdlt/baba-%d-ch00.png",patID);
		cvSaveImage(fname,imT1);
		sprintf(fname,"cdlt/baba-%d-ch01.png",patID);
		cvSaveImage(fname,imT2);


		printf("/");
		cvReleaseImage(&imT1);
		cvReleaseImage(&imT2);
		cvReleaseImage(&imGT);
		cvReleaseImage(&imRe);
	}
	getch();
}


void Watershed()
{
	const uchar bits[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
	const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	cvNamedWindow("Ikkuna", 1);
	cvNamedWindow("Ablak", 1);

	IplImage* imColor = cvLoadImage("macau.jpg", 1);
	IplImage* imO = cvCreateImage(cvGetSize(imColor), imColor->depth, 1);
	cvConvertImage(imColor, imO, CV_BGR2GRAY);
	
	//IplImage* imMask = cvLoadImage("hand2.png",0);
	
	cvShowImage("Ikkuna", imColor);
	
	IplImage* imG = cvCloneImage(imO);
	IplImage* imE = cvCloneImage(imO);
	IplImage* imKi = cvCloneImage(imO);
	IplImage* imBe = cvCloneImage(imO);
	IplImage* imLabel = cvCloneImage(imO);
	IplImage* imSegm = cvCloneImage(imColor);
	IplImage* imSegmMed = cvCloneImage(imColor);
	IplImage* imMap = cvCloneImage(imO);

	IplImage* imL = cvCreateImage(cvGetSize(imBe), IPL_DEPTH_16S, 1);

	IplImage* imRed = cvCloneImage(imO);
	IplImage* imGreen = cvCloneImage(imO);
	IplImage* imBlue = cvCloneImage(imO);
	IplImage* imSum = cvCloneImage(imO);
	cvSet(imSum, cvScalar(0));

	cvSplit(imColor, imBlue, imGreen, imRed, NULL);
	cvSobel(imBlue, imL, 1, 0);
	cvConvertScaleAbs(imL, imE);
	cvSobel(imBlue, imL, 0, 1);
	cvConvertScaleAbs(imL, imG);
	cvAdd(imE, imG, imG);
	cvAddWeighted(imSum, 1, imG, 0.33333, 0, imSum);
	cvSobel(imGreen, imL, 1, 0);
	cvConvertScaleAbs(imL, imE);
	cvSobel(imGreen, imL, 0, 1);
	cvConvertScaleAbs(imL, imG);
	cvAdd(imE, imG, imG);
	cvAddWeighted(imSum, 1, imG, 0.33333, 0, imSum);
	cvSobel(imRed, imL, 1, 0);
	cvConvertScaleAbs(imL, imE);
	cvSobel(imRed, imL, 0, 1);
	cvConvertScaleAbs(imL, imG);
	cvAdd(imE, imG, imG);
	cvAddWeighted(imSum, 1, imG, 0.33333, 0, imG);

	/*
	cvSobel(imO, imL, 1, 0);
	cvConvertScaleAbs(imL, imE);
	cvSobel(imO, imL, 0, 1);
	cvConvertScaleAbs(imL, imG);
	cvAdd(imE, imG, imG);
	*/
	//cvCmpS(imG, 9, imE, CV_CMP_LT);
	//cvSub(imG, imG, imG, imE);

	// Step 0
	cvSmooth(imG, imG, CV_GAUSSIAN, 17, 17);

	cvErode(imG, imE);
	cvSet(imSegm, cvScalar(50,50,50));
	cvSet(imSegmMed, cvScalar(150, 150, 150));
	cvSet(imLabel, cvScalar(0));
	cvSet(imBe, cvScalar(0));
	cvSet(imKi, cvScalar(8));
	cvSet(imMap, cvScalar(0));

	//  Step 1
	for (int x = 0; x<imBe->width; ++x) for (int y = 0; y<imBe->height; ++y)
	{
		int fp = getGray(imG, x, y);
		int q = getGray(imE, x, y);
		if (q<fp)
		{
			for (uchar irany = 0; irany<8; ++irany)
			if (x + dx[irany] >= 0 && x + dx[irany]<imBe->width && y + dy[irany] >= 0 && y + dy[irany]<imBe->height)
			{
				int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
				if (fpv == q)
				{
					setGray(imKi, x, y, irany);
					setGray(imMap, x, y, 255);
					uchar volt = getGray(imBe, x + dx[irany], y + dy[irany]);
					uchar adunk = bits[irany];
					uchar lesz = volt | adunk;
					setGray(imBe, x + dx[irany], y + dy[irany], lesz);
					break;
				}
			}
		}
	}

	cvShowImage("Ablak", imMap);
	cvWaitKey();


	// Step 2
	CvPoint* fifo = (CvPoint*)malloc(sizeof(CvPoint)*imBe->width*imBe->height);
	int nrFifo = 0;
	int readFifo = 0;

	for (int x = 0; x<imBe->width; ++x) for (int y = 0; y<imBe->height; ++y)
	{
		int fp = getGray(imG, x, y);
		int pout = getGray(imKi, x, y);
		if (pout == 8) continue;
		int added = 0;
		for (uchar irany = 0; irany<8; ++irany)
		if (x + dx[irany] >= 0 && x + dx[irany]<imBe->width && y + dy[irany] >= 0 && y + dy[irany]<imBe->height)
		{
			int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
			int pvout = getGray(imKi, x + dx[irany], y + dy[irany]);
			if (fpv == fp && pvout == 8)
			{
				if (added == 0) fifo[nrFifo++] = cvPoint(x, y);
				added++;
			}
		}
	}
	while (readFifo<nrFifo)
	{
		CvPoint p = fifo[readFifo++];
		int fp = getGray(imG, p.x, p.y);
		for (uchar irany = 0; irany<8; ++irany)
		if (p.x + dx[irany] >= 0 && p.x + dx[irany]<imBe->width && p.y + dy[irany] >= 0 && p.y + dy[irany]<imBe->height)
		{
			int fpv = getGray(imG, p.x + dx[irany], p.y + dy[irany]);
			int pvout = getGray(imKi, p.x + dx[irany], p.y + dy[irany]);
			if (fp == fpv && pvout == 8)
			{
				setGray(imKi, p.x + dx[irany], p.y + dy[irany], (irany + 4) % 8);
				setGray(imMap, p.x + dx[irany], p.y + dy[irany], 255);
				setGray(imBe, p.x, p.y, bits[(irany + 4) % 8] | getGray(imBe, p.x, p.y));
				fifo[nrFifo++] = cvPoint(p.x + dx[irany], p.y + dy[irany]);
			}
		}
	}
	cvShowImage("Ablak", imMap);
	cvWaitKey();
	// Step 3

	uint* medbuff = (uint*)malloc(sizeof(CvPoint)*imBe->width*imBe->height);
	CvPoint* stack = (CvPoint*)malloc(sizeof(CvPoint)*imBe->width*imBe->height);
	int nrStack = 0;
	for (int x = 0; x<imBe->width; ++x) for (int y = 0; y<imBe->height; ++y)
	{
		int fp = getGray(imG, x, y);
		int pout = getGray(imKi, x, y);
		if (pout != 8) continue;
		for (uchar irany = 0; irany<8; ++irany)
		if (x + dx[irany] >= 0 && x + dx[irany]<imBe->width && y + dy[irany] >= 0 && y + dy[irany]<imBe->height)
		{
			int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
			int pvout = getGray(imKi, x + dx[irany], y + dy[irany]);
			if (pvout == 8 && fp == fpv)
			{
				setGray(imKi, x + dx[irany], y + dy[irany], (irany + 4) % 8);
				setGray(imMap, x + dx[irany], y + dy[irany], 255);
				setGray(imBe, x, y, bits[(irany + 4) % 8] | getGray(imBe, x, y));
				stack[nrStack++] = cvPoint(x + dx[irany], y + dy[irany]);
			}
		}
		while (nrStack>0)
		{
			CvPoint pv = stack[--nrStack];
			int fpv = getGray(imG, pv.x, pv.y);
			int pvout = getGray(imKi, pv.x, pv.y);
			for (uchar irany = 0; irany<8; ++irany)
			if (pv.x + dx[irany] >= 0 && pv.x + dx[irany]<imBe->width && pv.y + dy[irany] >= 0 && pv.y + dy[irany]<imBe->height)
			{
				int fpvv = getGray(imG, pv.x + dx[irany], pv.y + dy[irany]);
				int pvvout = getGray(imKi, pv.x + dx[irany], pv.y + dy[irany]);
				//if (fpv==fpvv && pvvout==8 && (!(pv.x+dx[pvout]==x && pv.y+dy[pvout]==y)))
				if (fpv == fpvv && pvvout == 8 && (!(pv.x + dx[irany] == x && pv.y + dy[irany] == y)))
				{
					setGray(imMap, pv.x + dx[irany], pv.y + dy[irany], 255);
					setGray(imKi, pv.x + dx[irany], pv.y + dy[irany], (irany + 4) % 8);
					setGray(imBe, pv.x, pv.y, bits[(irany + 4) % 8] | getGray(imBe, pv.x, pv.y));
					stack[nrStack++] = cvPoint(pv.x + dx[irany], pv.y + dy[irany]);
				}
			}
		}
	}
	cvShowImage("Ablak", imMap);
	cvWaitKey();
	// Step 4
	int label = 0;
	nrFifo = 0;
	int spotSum[3];
	//printf("Hello: %d\n", label);
	//cvShowImage("Ikkuna", imSegm);
	//cvWaitKey();

	for (int x = 0; x<imBe->width; ++x) for (int y = 0; y<imBe->height; ++y)
	{
		int pout = getGray(imKi, x, y);
		if (pout != 8) continue;
		stack[nrStack++] = cvPoint(x, y);
		for (int i = 0; i<3; ++i) spotSum[i] = 0;
		while (nrStack>0)
		{
			CvPoint pv = stack[--nrStack];
			fifo[nrFifo++] = pv;
			uchar r, g, b;
			getColor(imColor, pv.x, pv.y, r, g, b);

			spotSum[0] += (int)b;
			spotSum[1] += (int)g;
			spotSum[2] += (int)r;
			uint o = (int)r * 0x10000 + (int)g * 0x100 + (int)b;
			o += (uint)(cvRound((float)r * 0.299 + 
				(float)g * 0.587 + (float)b * 0.114) * 0x1000000);
			medbuff[nrFifo - 1] = o;
			setGray(imLabel, pv.x, pv.y, label);
			int pvin = getGray(imBe, pv.x, pv.y);
			for (uchar irany = 0; irany<8; ++irany)
			{
				if ((bits[irany] & pvin)> 0)
				{
					setGray(imLabel, pv.x + dx[(irany + 4) % 8], pv.y + dy[(irany + 4) % 8], label);
					stack[nrStack++] = cvPoint(pv.x + dx[(irany + 4) % 8], pv.y + dy[(irany + 4) % 8]);
				}
			}
		}
		label++;
		if (nrFifo < 2) printf("%d", nrFifo);
		for (int i = 0; i<3; ++i)
			spotSum[i] = cvRound(spotSum[i] / nrFifo);

		qsort(medbuff, nrFifo, sizeof(uint), compare);

		int medR = (medbuff[nrFifo / 2] % 0x1000000) / 0x10000;
		int medG = (medbuff[nrFifo / 2] % 0x10000) / 0x100;
		int medB = (medbuff[nrFifo / 2] % 0x100);

		/*
		int th = 100;

		for (int i = 0; i < nrFifo; ++i) if (getGray(imMask, fifo[i].x, fifo[i].y) > 128)
		{
			//atlag
			if (spotSum[0] > th)
				setColor(imSegm, fifo[i].x, fifo[i].y, 0, 255, 0);
			else
				setColor(imSegm, fifo[i].x, fifo[i].y, 255, 0, 0);

			//median
			if (medB > th)
				setColor(imSegmMed, fifo[i].x, fifo[i].y, 0, 255, 0);
			else
				setColor(imSegmMed, fifo[i].x, fifo[i].y, 255, 0, 0);
		}
		*/
		
		for (int i = 0; i < nrFifo; ++i) //if (getGray(imMask, fifo[i].x, fifo[i].y) > 128)
		{
			//atlag
			setColor(imSegm, fifo[i].x, fifo[i].y, (uchar)spotSum[2], (uchar)spotSum[1], (uchar)spotSum[0]);
			//median
			setColor(imSegmMed, fifo[i].x, fifo[i].y, (uchar)medR, (uchar)medG, (uchar)medB);
		}
			
		nrFifo = 0;

	//	printf("Hello: %d\n", label);
	//	cvShowImage("Ikkuna", imSegm);
	//	cvWaitKey();
	}


	free(fifo);
	free(stack);
	free(medbuff);
	// no more steps
	printf("\nRegions: %d \n", label);

	cvShowImage("Ikkuna", imSegm);
	cvShowImage("Ablak", imO);
	cvWaitKey();
}

void Szabolcs1()
{
	ConfusionMatrix CMoverall(4);
	char fname[100];
	for (int patID=0; patID<220; ++patID)
	{
		ConfusionMatrix CM(4);
		if (patID%2)
			sprintf(fname,"szabolcs/y_pred_paros_10K_volume%03d.csv",patID);
		else
			sprintf(fname,"szabolcs/y_pred_paratlan_10K_volume%03d.csv",patID);
		int count = 0;
		FILE* F = fopen(fname,"rt");
		while (!feof(F))
		{
			float a, b;
			fname[0]=0;
			fgets(fname,69,F);
			if (strlen(fname)>3)
			{
				sscanf(fname,"%f,%f",&a,&b);
				CM.addItem((int)a,(int)b);
				CMoverall.addItem((int)a,(int)b);
				++count;
			}
		}
		fclose(F);
		printf("%3d -> %d\n",patID,count);
		CM.compute();
	}
	CMoverall.compute();
}


void OcvLogRegX(int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES)
{
	const int BATCH = 10000;
	const int PERMUTE[NR_FEATURES] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
	int nrFeat = USED_FEATURES;
	if (nrFeat < 2) nrFeat = 2;
	if (nrFeat > NR_FEATURES) nrFeat = NR_FEATURES;

	int depilimit = maxDepth;
	if (depilimit < 5) depilimit = 5;
	if (depilimit > 50) depilimit = 50;

	printf("\nTraining begins...\n");
// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID-1)*nrTrainPixPerTrainBaby;
	int* selForTrain = (int*)malloc(sizeof(int)*nrTrainData);
	int trainCount = 0;

	for (int patID=0; patID<MAX_PATID; ++patID)
	if (testBaby != patID)
	{
		int count = 0;
		int index = 0;
		while (count<nrTrainPixPerTrainBaby)
		{
			index = (index + rand()*rand() + rand()) % Limits[2][patID];
			selForTrain[trainCount] = Limits[0][patID] + index;
			++count;
			++trainCount;
		}
	}

	Mat_<float> trainFeatures(nrTrainData, nrFeat);
	Mat_<float> trainLabels(nrTrainData, 1);

	for (int o=0; o<nrTrainData; ++o)
	{
		for (int f=0; f<nrFeat; ++f)
		{
			float value = (float)(FeatBuffer[PERMUTE[f]][selForTrain[o]]);
			trainFeatures(o,f) = value;
		}
		trainLabels(o,0) = GTBuffer[selForTrain[o]];
	}

	Ptr<ml::LogisticRegression> logreg = ml::LogisticRegression::create();
	logreg->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
	//params
/*	rand_trees->setMaxDepth(depilimit);
	rand_trees->setMinSampleCount(0);
	rand_trees->setRegressionAccuracy(0.0f);
	rand_trees->setUseSurrogates(false);
	rand_trees->setMaxCategories(3);
	rand_trees->setCVFolds(1);
	rand_trees->setUse1SERule(false);
	rand_trees->setTruncatePrunedTree(false);
//	rand_trees->setPriors(Mat());
	rand_trees->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 45, 0.01f));
*/

// confusion matrix declaration

	ConfusionMatrix kungfu(NR_TISSUES);

// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	int pixelsToTest = Limits[2][testBaby];
	int nextTestPixel = Limits[0][testBaby];

	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, nrFeat);
		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			for (int f=0; f<nrFeat; ++f)
			{
				float value = (float)(FeatBuffer[PERMUTE[f]][pix]);
				testFeatures(pix-nextTestPixel,f) = value;
			}		
		}
		Mat response;
		logreg->predict(testFeatures, response);

		for (int pix=nextTestPixel; pix<nextTestPixel+thisBatch; ++pix)
		{
			int deci = response.at<float>(pix-nextTestPixel, 0);
			int gt = GTBuffer[pix];
			kungfu.addItem(gt,deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}

	kungfu.compute();
}



