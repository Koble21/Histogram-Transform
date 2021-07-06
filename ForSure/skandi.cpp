
#include "Transform.h"
//#include "Bratsform.h"


const int MAX_PATID = 50; // 50
const int MAX_SLICES = 256; // 155
const int WIDTH = 144; // 160
const int HEIGHT = 192;
const int VOLSIZE = WIDTH * HEIGHT * MAX_SLICES;
const int MAX_PIXELS = 72000000;
const int MILLIO = 1000000;  // 2000000
const int OBSERVED_CHANNELS = 2; // 4
const int MAX_INTENSITY = 1000;  // 32768 0x8000
const int NR_FEATURES = 21; // 36

const int FIRSTSLICES[MAX_PATID] = { 85,91,91,97,92,89,89,94,98,98 };


const int SLICES_PER_ROW = 16;
const int SLICES_PER_COL = 7;
const int NR_TISSUES = 3;
const int VERY_BIG_VALUE = MAX_PIXELS;



unsigned short** BigFootBuffer;
uchar** FeatBuffer;
uchar** PosBuffer;
uchar* GTBuffer;
PatientBasicData patLimits[MAX_PATID];


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

void CreateBuffers(int bits=8)
{
	if (bits>8)
	{
		BigFootBuffer = (unsigned short**)malloc(NR_FEATURES * sizeof(unsigned short*));
		for (int f=0; f<NR_FEATURES; ++f)
			BigFootBuffer[f] = (unsigned short*)malloc(MAX_PIXELS * sizeof(unsigned short));
	}
	else
	{
		FeatBuffer = (uchar**)malloc(NR_FEATURES * sizeof(uchar*));
		for (int f=0; f<NR_FEATURES; ++f)
			FeatBuffer[f] = (uchar*)malloc(MAX_PIXELS);
	}

	PosBuffer = (uchar**)malloc(NR_COORDS * sizeof(uchar*));
	for (int c=0; c<NR_COORDS; ++c)
		PosBuffer[c] = (uchar*)malloc(MAX_PIXELS);

	GTBuffer = (uchar*)malloc(MAX_PIXELS);
}

void ReleaseBuffers(int bits=8)
{
	if (bits>8)
	{
		for (int f=0; f<NR_FEATURES; ++f)
			free(BigFootBuffer[f]);
		free(BigFootBuffer);
	}
	else
	{
		for (int f=0; f<NR_FEATURES; ++f)
			free(FeatBuffer[f]);
		free(FeatBuffer);
	}

	for (int c=0; c<NR_COORDS; ++c)
		free(PosBuffer[c]);
	free(PosBuffer);

	free(GTBuffer);
}

void LoadFeatures(char* fname)
{
	int head[3];
	FILE* F = fopen(fname,"rb");
	fread(head,sizeof(int),3,F);
	fread(patLimits,sizeof(PatientBasicData),MAX_PATID,F);
	CreateBuffers(head[2]);
	fread(GTBuffer,sizeof(uchar),head[0],F);
	for (int c=0; c<NR_COORDS; ++c)
		fread(PosBuffer[c],sizeof(uchar),head[0],F);
	for (int f=0; f<NR_FEATURES; ++f)
		if (head[2]>8)
			fread(BigFootBuffer[f],sizeof(unsigned short),head[0],F);
		else
			fread(FeatBuffer[f],sizeof(uchar),head[0],F);
	fclose(F);
	printf("Successfully loaded %d pixel data.\n",head[0]);
	FILE* G = fopen("kingkong.csv", "at");
	fprintf(G, "%s\n", fname);
	fclose(G);
}


void OcvOneStage(int alg, int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES, int RES_IMAGE = 0)
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
				index = (index + rand() * rand() + rand()) % patLimits[patID].pixelCount;
				selForTrain[trainCount] = patLimits[patID].firstIndex + index;
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
	Ptr<cv::ml::SVM>  svm = cv::ml::SVM::create();

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
	case ALG_SVM:
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setC(1);
		svm->setKernel(cv::ml::SVM::INTER);
		svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 500, 1e-6));
		startTime();
		svm->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		trainLapse = endTime();
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
	int pixelsToTest = patLimits[testBaby].pixelCount; 
	int nextTestPixel = patLimits[testBaby].firstIndex;

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
		else if (alg == ALG_SVM)
			svm->predict(testFeatures, response);

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



void TrainAndTest(int _voltype, char* _dataFile, int alg, int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int USED_FEATURES = NR_FEATURES, int RES_IMAGE = 0)
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
				index = (index + rand() * rand() + rand()) % patLimits[patID].pixelCount;
				selForTrain[trainCount] = patLimits[patID].firstIndex + index;
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
	Ptr<cv::ml::SVM>  svm = cv::ml::SVM::create();

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
	case ALG_SVM:
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setC(1);
		svm->setKernel(cv::ml::SVM::INTER);
		svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 500, 1e-6));
		startTime();
		svm->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		trainLapse = endTime();
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
	int pixelsToTest = patLimits[testBaby].pixelCount;
	int nextTestPixel = patLimits[testBaby].firstIndex;

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

		if (alg == ALG_DTREE)
			dec_trees->predict(testFeatures, response);
		else if (alg == ALG_RF)
			rand_trees->predict(testFeatures, response);
		else if (alg == ALG_KNN)
			knn->findNearest(testFeatures, kknn, response);
		else if (alg == ALG_ANN)
			nn->predict(testFeatures, response);
		else if (alg == ALG_SVM)
			svm->predict(testFeatures, response);

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


void generateFeatures(int ALG=1, int qBitDepth = 8, float l25 = 0.4f, float pLO = 0.02f, int nrMilestones = 7)
{
	FILE* F;
	char fname[100];
	printf("Starting feature generation. \n");


	unsigned short* dataBuffers[OBSERVED_CHANNELS];
	for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
		dataBuffers[ch] = (unsigned short*)malloc(VOLSIZE*sizeof(unsigned short));
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	CreateBuffers(qBitDepth);
	int totalPixels=0;

	// adatok beolvasása és jellemzők számítása
	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		if (ALG==1)
			sprintf(fname,"babainput/baba-A1-%d-%d-%d.bab",patID,qBitDepth,cvRound(100*l25));
		else 
			sprintf(fname,"babainput/baba-A%d-%d-%d-%d-%d.bab",ALG,patID,qBitDepth,nrMilestones,cvRound(1000*pLO));

		F = fopen(fname,"rb");
		for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
			fread(dataBuffers[ch],sizeof(unsigned short),VOLSIZE,F);
		fread(bufferGT,1,VOLSIZE,F);
		fclose(F);
		int count=0;
		for (int i=0; i<VOLSIZE; ++i)
		{
			if (bufferGT[i]>0)
			{
				int index = totalPixels+count;
				PosBuffer[0][index] = patID;
				PosBuffer[1][index] = i/(WIDTH*HEIGHT) - FIRSTSLICES[patID];
				PosBuffer[2][index] = (i%(WIDTH*HEIGHT))/WIDTH;
				PosBuffer[3][index] = i%WIDTH;
				GTBuffer[index] = bufferGT[i]/100; //(bufferGT[i]==10 ? 50 : bufferGT[i]);
				if (qBitDepth>8)
				{
					for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
					{
						BigFootBuffer[ch][index] = dataBuffers[ch][i];
						int sum = dataBuffers[ch][i];
						int db = 1;
						for (int s=1; s<=5; ++s)
						{
							for (int j=-s; j<s; ++j)
							{
								if (dataBuffers[ch][i-s*WIDTH+j]>0)
								{
									sum += dataBuffers[ch][i-s*WIDTH+j];
									++db;
								}
								if (dataBuffers[ch][i+s+j*WIDTH]>0)
								{
									sum += dataBuffers[ch][i+s+j*WIDTH];
									++db;
								}
								if (dataBuffers[ch][i+s*WIDTH-j]>0)
								{
									sum += dataBuffers[ch][i+s*WIDTH-j];
									++db;
								}
								if (dataBuffers[ch][i-s-j*WIDTH]>0)
								{
									sum += dataBuffers[ch][i-s-j*WIDTH];
									++db;
								}
							}
							BigFootBuffer[2*s+ch][index] = (sum+db/2)/db;
						}
						db = 0;
						sum = 0;
						int mini = dataBuffers[ch][i];
						int maxi = dataBuffers[ch][i];
						for (int dz=-1;dz<=1;++dz) for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx)
						{
							int value = dataBuffers[ch][i+(dz*HEIGHT+dy)*WIDTH+dx];
							if (value>0)
							{
								sum += value;
								if (maxi<value) maxi = value;
								if (mini>value) mini = value;
								++db;
							}
						}
						BigFootBuffer[12+ch][index] = (sum+db/2)/db;
						BigFootBuffer[14+ch][index] = maxi;
						BigFootBuffer[16+ch][index] = mini;
					}
				}
				else
				{
					for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
					{
						FeatBuffer[ch][index] = dataBuffers[ch][i];
						int sum = dataBuffers[ch][i];
						int db = 1;
						for (int s=1; s<=5; ++s)
						{
							for (int j=-s; j<s; ++j)
							{
								if (dataBuffers[ch][i-s*WIDTH+j]>0)
								{
									sum += dataBuffers[ch][i-s*WIDTH+j];
									++db;
								}
								if (dataBuffers[ch][i+s+j*WIDTH]>0)
								{
									sum += dataBuffers[ch][i+s+j*WIDTH];
									++db;
								}
								if (dataBuffers[ch][i+s*WIDTH-j]>0)
								{
									sum += dataBuffers[ch][i+s*WIDTH-j];
									++db;
								}
								if (dataBuffers[ch][i-s-j*WIDTH]>0)
								{
									sum += dataBuffers[ch][i-s-j*WIDTH];
									++db;
								}
							}
							FeatBuffer[2*s+ch][index] = (sum+db/2)/db;
						}
						db = 0;
						sum = 0;
						int mini = dataBuffers[ch][i];
						int maxi = dataBuffers[ch][i];
						for (int dz=-1;dz<=1;++dz) for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx)
						{
							int value = dataBuffers[ch][i+(dz*HEIGHT+dy)*WIDTH+dx];
							if (value>0)
							{
								sum += value;
								if (maxi<value) maxi = value;
								if (mini>value) mini = value;
								++db;
							}
						}
						FeatBuffer[12+ch][index] = (sum+db/2)/db;
						FeatBuffer[14+ch][index] = maxi;
						FeatBuffer[16+ch][index] = mini;
					}
				}

				++count;
			}
		}
		patLimits[patID].firstIndex = totalPixels;
		patLimits[patID].lastIndex = totalPixels + count - 1;
		patLimits[patID].pixelCount = count;
		totalPixels += count;
		printf(".");
	}
	float Q=1.0;
	for (int i=0; i<qBitDepth; ++i) Q=2.0f*Q;
	Q=Q-2.0f;
	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		int minX=WIDTH/2, maxX=WIDTH/2, minY=HEIGHT/2, maxY=HEIGHT/2, minZ=MAX_SLICES-1, maxZ=0;  
		for (int i=patLimits[patID].firstIndex; i<=patLimits[patID].lastIndex; ++i)
		{
			if (PosBuffer[1][i]>maxZ) maxZ=PosBuffer[1][i];
			if (PosBuffer[1][i]<minZ) minZ=PosBuffer[1][i];
			if (PosBuffer[2][i]>maxY) maxY=PosBuffer[2][i];
			if (PosBuffer[2][i]<minY) minY=PosBuffer[2][i];
			if (PosBuffer[3][i]>maxX) maxX=PosBuffer[3][i];
			if (PosBuffer[3][i]<minX) minX=PosBuffer[3][i];
		}
		for (int i=patLimits[patID].firstIndex; i<=patLimits[patID].lastIndex; ++i)
		{
			if (qBitDepth>8)
			{
				BigFootBuffer[18][i] = 1 + cvRound(Q*((float)(PosBuffer[1][i]-minZ) / (float)(maxZ-minZ)));
				BigFootBuffer[19][i] = 1 + cvRound(Q*((float)(PosBuffer[2][i]-minY) / (float)(maxY-minY)));
				BigFootBuffer[20][i] = 1 + cvRound(Q*((float)(PosBuffer[3][i]-minX) / (float)(maxX-minX)));
			}
			else
			{
				FeatBuffer[18][i] = 1 + cvRound(Q*((float)(PosBuffer[1][i]-minZ) / (float)(maxZ-minZ)));
				FeatBuffer[19][i] = 1 + cvRound(Q*((float)(PosBuffer[2][i]-minY) / (float)(maxY-minY)));
				FeatBuffer[20][i] = 1 + cvRound(Q*((float)(PosBuffer[3][i]-minX) / (float)(maxX-minX)));
			}
		}
	}
	// Mentés
	if (ALG==1)
		sprintf(fname,"bigbaba-A1-%d-%d.dat",qBitDepth,cvRound(100*l25));
	else 
		sprintf(fname,"bigbaba-A%d-%d-%d-%d.dat",ALG,qBitDepth,nrMilestones,cvRound(1000*pLO));
	F = fopen(fname,"wb");
	fwrite(&totalPixels,sizeof(int),1,F);
	fwrite(&NR_FEATURES,sizeof(int),1,F);
	fwrite(&qBitDepth,sizeof(int),1,F);
	fwrite(patLimits,sizeof(PatientBasicData),MAX_PATID,F);
	fwrite(GTBuffer,sizeof(uchar),totalPixels,F);
	for (int c = 0; c < NR_COORDS; ++c)
	{
		fwrite(PosBuffer[c], sizeof(uchar), totalPixels, F);
		free(PosBuffer[c]);
	}
	for (int f=0; f<NR_FEATURES; ++f)
		if (qBitDepth > 8)
		{
			fwrite(BigFootBuffer[f], sizeof(unsigned short), totalPixels, F);
			free(BigFootBuffer[f]);
		}
		else
		{
			fwrite(FeatBuffer[f], sizeof(uchar), totalPixels, F);
			free(FeatBuffer[f]);
		}
	fclose(F);
	printf("Done. \n");

	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);
}

void HistNormLinearTransformX(int qBitDepth = 8, float l25 = 0.4f)
{
	float lambda25 = l25;
	if (lambda25 < 0.25f) lambda25 = 0.25;
	if (lambda25 > 0.45f) lambda25 = 0.45;
	float lambda75 = 1.0f - lambda25;

	char fname[100];
	unsigned short* dataBuffers[OBSERVED_CHANNELS];
	for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
		dataBuffers[ch] = (unsigned short*)malloc(VOLSIZE*sizeof(unsigned short));
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);
	unsigned short* flat = (unsigned short*)malloc(MILLIO*sizeof(unsigned short));

	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* F = fopen(fname,"rb");
		int nrBytes = 0;
		for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
			nrBytes += fread(dataBuffers[ch],sizeof(unsigned short),VOLSIZE,F);
		nrBytes += fread(bufferGT,1,VOLSIZE,F);
		fclose(F);
		// if (nrBytes<3*VOLSIZE) printf("Data reading failed");
		int hist[OBSERVED_CHANNELS][MAX_INTENSITY]={0};

		int pixCount = 0;
		for (int pix=0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
					hist[ch][ dataBuffers[ch][pix]-1 ]++;
				++pixCount;
			}
	
		int index, p25, p75;
		float a, b;
		float coltrans[OBSERVED_CHANNELS][MAX_INTENSITY]={0};

		for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
		{
			index = 0;
			for (int value=0; value<MAX_INTENSITY; ++value)
			{
				for (int i=0; i<hist[ch][value]; ++i)
					flat[index++] = value+1;
			}

			p25 = flat[pixCount/4];
			p75 = flat[pixCount*3/4];

			a = (lambda75-lambda25)/((float)(p75-p25));
			b = lambda25 - a * (float)p25;

			for (int value=0; value<MAX_INTENSITY; ++value)
			{
				float newValue = a * (value+1) + b;
				if (newValue < 0.0f) newValue = 0.0f;
				if (newValue > 1.0f) newValue = 1.0f;
				coltrans[ch][value] = newValue;
			}
		}
		
		float Q = 1.0f;
		for (int i=0; i<qBitDepth; ++i) Q = 2.0f*Q;
		for (int pix=0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
				for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
				{
					int value = dataBuffers[ch][pix];
					int newValue = 1 + cvRound((Q-2.0) * coltrans[ch][value]);
					dataBuffers[ch][pix] = newValue;
				}

		sprintf(fname,"babainput/baba-A1-%d-%d-%d.bab",patID,qBitDepth,cvRound(100*lambda25));
		F = fopen(fname,"wb");
		for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
			fwrite(dataBuffers[ch],2,VOLSIZE,F);
		fwrite(bufferGT,1,VOLSIZE,F);
		fclose(F);
	}
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);
	free(flat);

	generateFeatures(1,qBitDepth,lambda25);
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);
}



void HistNormLinearTransform(int qBitDepth = 8, float l25 = 0.4f)
{
	float lambda25 = l25;
	if (lambda25 < 0.25f) lambda25 = 0.25;
	if (lambda25 > 0.45f) lambda25 = 0.45;
	float lambda75 = 1.0f - lambda25;

	char fname[100];
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);
	
	unsigned short* flat = (unsigned short*)malloc(MILLIO*2);

	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* F = fopen(fname,"rb");
		int nrBytes = 0;
		nrBytes += fread(bufferT1,2,VOLSIZE,F);
		nrBytes += fread(bufferT2,2,VOLSIZE,F);
		nrBytes += fread(bufferGT,1,VOLSIZE,F);
		fclose(F);
		// if (nrBytes<3*VOLSIZE) printf("Data reading failed");
		int histT1[MAX_INTENSITY]={0};
		int histT2[MAX_INTENSITY]={0};

		int pixCount = 0;
		for (int pix=0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				histT1[ bufferT1[pix]-1 ]++;
				histT2[ bufferT2[pix]-1 ]++;
				++pixCount;
			}
	
		int index, p25, p75;
		float a, b;

		index = 0;
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			for (int i=0; i<histT1[value]; ++i)
				flat[index++] = value+1;
		}

		p25 = flat[pixCount/4];
		p75 = flat[pixCount*3/4];

		a = (lambda75-lambda25)/((float)(p75-p25));
		b = lambda25 - a * (float)p25;

		float coltransT1[MAX_INTENSITY]={0};
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			float newValue = a * (value+1) + b;
			if (newValue < 0.0f) newValue = 0.0f;
			if (newValue > 1.0f) newValue = 1.0f;
			coltransT1[value] = newValue;
		}
		
		index = 0;
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			for (int i=0; i<histT2[value]; ++i)
				flat[index++] = value+1;
		}

		p25 = flat[pixCount/4];
		p75 = flat[pixCount*3/4];

		a = (lambda75-lambda25)/((float)(p75-p25));
		b = lambda25 - a * (float)p25;

		float coltransT2[MAX_INTENSITY]={0};
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			float newValue = a * (value+1) + b;
			if (newValue < 0.0f) newValue = 0.0f;
			if (newValue > 1.0f) newValue = 1.0f;
			coltransT2[value] = newValue;
		}		
		
		float Q = 1.0f;
		for (int i=0; i<qBitDepth; ++i) Q = 2.0f*Q;
		for (int pix=0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				int value = bufferT1[pix];
				int newValue = 1 + cvRound((Q-2.0) * coltransT1[value]);
				bufferT1[pix] = newValue;

				value = bufferT2[pix];
				newValue = 1 + cvRound((Q-2.0) * coltransT2[value]);
				bufferT2[pix] = newValue;
			}

		sprintf(fname,"babainput/baba-A1-%d-%d-%d.bab",patID,qBitDepth,cvRound(100*lambda25));
		F = fopen(fname,"wb");
		fwrite(bufferT1,2,VOLSIZE,F);
		fwrite(bufferT2,2,VOLSIZE,F);
		fwrite(bufferGT,1,VOLSIZE,F);
		fclose(F);
	}
}


void HistNormNyulTransformX(int qBitDepth = 8, float pLO = 0.02f, int nrMilestones = 7)
{
	const int mileStones[12][12] =
	{ {-1,0,0,0,0,0,0,0,0,0,0,0},
	 {-1,0,0,0,0,0,0,0,0,0,0,0},
	 {-1,0,0,0,0,0,0,0,0,0,0,0},
	 {50,-1,0,0,0,0,0,0,0,0,0,0},
	 {25,75,-1,0,0,0,0,0,0,0,0,0},
	 {25,50,75,-1,0,0,0,0,0,0,0,0},
	 {10,25,75,90,-1,0,0,0,0,0,0,0},
	 {10,25,50,75,90,-1,0,0,0,0,0,0},
	 {10,25,40,60,75,90,-1,0,0,0,0,0},
	 {10,25,40,50,60,75,90,-1,0,0,0,0},
	 {10,20,30,40,60,70,80,90,-1,0,0,0},
	{10,20,30,40,50,60,70,80,90,-1,0,0} };

	int M = nrMilestones;
	if (M < 3) M = 3;
	if (M > 11) M = 11;

	float p_lo = pLO;
	if (p_lo < 0.01f) p_lo = 0.01;
	if (p_lo > 0.05f) p_lo = 0.05;
	float p_hi = 1.0f - p_lo;

	char fname[100];
	unsigned short* dataBuffers[OBSERVED_CHANNELS];
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		dataBuffers[ch] = (unsigned short*)malloc(VOLSIZE * sizeof(unsigned short));

	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	unsigned short* flat = (unsigned short*)malloc(MILLIO * sizeof(unsigned short));

	float a[OBSERVED_CHANNELS][MAX_PATID], b[OBSERVED_CHANNELS][MAX_PATID];
	int p_kh[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	float y_kh[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	float y_k_avg[OBSERVED_CHANNELS][MAX_MILESTONES] = { 0 };

	for (int patID = 0; patID < MAX_PATID; ++patID)
	{
		sprintf(fname, "babainput/baba-%d.bab", patID);
		FILE* F = fopen(fname, "rb");
		int nrBytes = 0;
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
			nrBytes += fread(dataBuffers[ch], 2, VOLSIZE, F);
		nrBytes += fread(bufferGT, 1, VOLSIZE, F);
		fclose(F);
		// if (nrBytes<3*VOLSIZE) printf("Data reading failed");
		int hist[OBSERVED_CHANNELS][MAX_INTENSITY] = { 0 };

		int pixCount = 0;
		for (int pix = 0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
					hist[ch][dataBuffers[ch][pix] - 1]++;
				++pixCount;
			}

		int index, p1, p99, p_ms;

		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		{
			index = 0;
			for (int value = 0; value < MAX_INTENSITY; ++value)
			{
				for (int i = 0; i < hist[ch][value]; ++i)
					flat[index++] = value + 1;
			}

			p1 = flat[cvRound((float)pixCount * p_lo)];
			p99 = flat[cvRound((float)pixCount * p_hi)];

			a[ch][patID] = 1.0f / ((float)(p99 - p1));
			b[ch][patID] = -a[ch][patID] * (float)p1;

			y_kh[ch][patID][0] = 0.0f;
			y_kh[ch][patID][M - 1] = 1.0f;
			p_kh[ch][patID][0] = p1;
			p_kh[ch][patID][M - 1] = p99;
			for (int i = 0; i < M - 2; ++i)
			{
				int ms = mileStones[M][i];
				p_ms = flat[cvRound((float)pixCount * (float)ms * 0.01f)];
				y_kh[ch][patID][i + 1] = a[ch][patID] * p_ms + b[ch][patID];
				p_kh[ch][patID][i + 1] = p_ms;
			}
		}
	}

	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		for (int patID = 0; patID < MAX_PATID; ++patID)
		{
			for (int m = 0; m < M; ++m)
				y_k_avg[ch][m] += y_kh[ch][patID][m] / (float)MAX_PATID;
		}

	for (int patID = 0; patID < MAX_PATID; ++patID)
	{
		float coltrans[OBSERVED_CHANNELS][MAX_INTENSITY] = { 0 };
	
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		{
			for (int value = 0; value < p_kh[ch][patID][0]; ++value)
				coltrans[ch][value] = 0.0f;
			for (int value = p_kh[ch][patID][0]; value < MAX_INTENSITY; ++value)
				coltrans[ch][value] = 1.0f;
			for (int m = 0; m < M - 1; ++m)
			{
				for (int value = p_kh[ch][patID][m]; value <= p_kh[ch][patID][m + 1]; ++value)
				{
					float hanyadreszettettukmegazutnak = (float)(value - p_kh[ch][patID][m]) / (float)(p_kh[ch][patID][m + 1] - p_kh[ch][patID][m]);
					coltrans[ch][value - 1] = y_k_avg[ch][m] + (y_k_avg[ch][m + 1] - y_k_avg[ch][m]) * hanyadreszettettukmegazutnak;
				}
			}
		}

		sprintf(fname, "babainput/baba-%d.bab", patID);
		FILE* F = fopen(fname, "rb");
		int nrBytes = 0;
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
			nrBytes += fread(dataBuffers[ch], 2, VOLSIZE, F);
		nrBytes += fread(bufferGT, 1, VOLSIZE, F);
		fclose(F);

		float Q = 1.0f;
		for (int i = 0; i < qBitDepth; ++i) Q = 2.0f * Q;
		for (int pix = 0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
				{
					int value = dataBuffers[ch][pix];
					int newValue = 1 + cvRound((Q - 2.0) * coltrans[ch][value]);
					dataBuffers[ch][pix] = newValue;
				}
			}

		sprintf(fname, "babainput/baba-A2-%d-%d-%d-%d.bab", patID, qBitDepth, M, cvRound(1000 * p_lo));
		F = fopen(fname, "wb");
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
			fwrite(dataBuffers[ch], 2, VOLSIZE, F);
		fwrite(bufferGT, 1, VOLSIZE, F);
		fclose(F);
	}
	generateFeatures(2,qBitDepth,0,pLO,nrMilestones);// (1, qBitDepth, lambda25);
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);
}


void HistNormNyulTransform(int qBitDepth = 8, float pLO = 0.02f, int nrMilestones = 7)
{
	const int MAX_MILESTONES = 11;
	const int mileStones[12][12] =
	{{-1,0,0,0,0,0,0,0,0,0,0,0},
  	 {-1,0,0,0,0,0,0,0,0,0,0,0},
	 {-1,0,0,0,0,0,0,0,0,0,0,0},
	 {50,-1,0,0,0,0,0,0,0,0,0,0},
	 {25,75,-1,0,0,0,0,0,0,0,0,0},
	 {25,50,75,-1,0,0,0,0,0,0,0,0},
	 {10,25,75,90,-1,0,0,0,0,0,0,0},
	 {10,25,50,75,90,-1,0,0,0,0,0,0},
	 {10,25,40,60,75,90,-1,0,0,0,0,0},
	 {10,25,40,50,60,75,90,-1,0,0,0,0},
	 {10,20,30,40,60,70,80,90,-1,0,0,0},
	{10,20,30,40,50,60,70,80,90,-1,0,0}};

	int M = nrMilestones;
	if (M < 3) M = 3;
	if (M > 11) M = 11;

	float p_lo = pLO;
	if (p_lo < 0.01f) p_lo = 0.01;
	if (p_lo > 0.05f) p_lo = 0.05;
	float p_hi = 1.0f - p_lo;

	char fname[100];
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);
	
	unsigned short* flat = (unsigned short*)malloc(MILLIO*2);
	
	float a[OBSERVED_CHANNELS][MAX_PATID], b[OBSERVED_CHANNELS][MAX_PATID];
	int p_kh[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	float y_kh[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	float y_k_avg[OBSERVED_CHANNELS][MAX_MILESTONES] = {0};

	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* F = fopen(fname,"rb");
		int nrBytes = 0;
		nrBytes += fread(bufferT1,2,VOLSIZE,F);
		nrBytes += fread(bufferT2,2,VOLSIZE,F);
		nrBytes += fread(bufferGT,1,VOLSIZE,F);
		fclose(F);
		// if (nrBytes<3*VOLSIZE) printf("Data reading failed");
		int histT1[MAX_INTENSITY]={0};
		int histT2[MAX_INTENSITY]={0};

		int pixCount = 0;
		for (int pix=0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				histT1[ bufferT1[pix]-1 ]++;
				histT2[ bufferT2[pix]-1 ]++;
				++pixCount;
			}
	
		int index, p1, p99, p_ms;

		index = 0;
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			for (int i=0; i<histT1[value]; ++i)
				flat[index++] = value+1;
		}

		p1 = flat[cvRound((float)pixCount*p_lo)];
		p99 = flat[cvRound((float)pixCount*p_hi)];

		a[0][patID] = 1.0f/((float)(p99-p1));
		b[0][patID] = -a[0][patID] * (float)p1;

		y_kh[0][patID][0] = 0.0f;
		y_kh[0][patID][M-1] = 1.0f;
		p_kh[0][patID][0] = p1;
		p_kh[0][patID][M-1] = p99;
		for (int i=0; i<M-2; ++i)
		{
			int ms = mileStones[M][i];
			p_ms = flat[cvRound((float)pixCount*ms*0.01)];
			y_kh[0][patID][i+1] = a[0][patID] * p_ms + b[0][patID];
			p_kh[0][patID][i+1] = p_ms;
		}

		index = 0;
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			for (int i=0; i<histT2[value]; ++i)
				flat[index++] = value+1;
		}

		p1 = flat[cvRound((float)pixCount*p_lo)];
		p99 = flat[cvRound((float)pixCount*p_hi)];

		a[1][patID] = 1.0f/((float)(p99-p1));
		b[1][patID] = -a[1][patID] * (float)p1;

		y_kh[1][patID][0] = 0.0f;
		y_kh[1][patID][M-1] = 1.0f;
		p_kh[1][patID][0] = p1;
		p_kh[1][patID][M-1] = p99;		
		for (int i=0; i<M-2; ++i)
		{
			int ms = mileStones[M][i];
			p_ms = flat[cvRound((float)pixCount*ms*0.01)];
			y_kh[1][patID][i+1] = a[1][patID] * p_ms + b[1][patID];
			p_kh[1][patID][i+1] = p_ms;
		}
	}

	for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
		for (int patID = 0; patID<MAX_PATID; ++patID)	
		{
			for (int m = 0; m<M; ++m)
				y_k_avg[ch][m] += y_kh[ch][patID][m] / (float)MAX_PATID;
		}



	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		float coltransT1[MAX_INTENSITY]={0};
		for (int value=0; value<p_kh[0][patID][0]; ++value)
			coltransT1[value] = 0.0f;
		for (int value=p_kh[0][patID][0]; value<MAX_INTENSITY; ++value)
			coltransT1[value] = 1.0f;
		for (int m=0; m<M-1; ++m)
		{
			// [m,m+1]:   [p_kh[0][patID][m] --> p_kh[0][patID][m+1]]  [y_k_avg[0][m] --> y_k_avg[0][m+1]]
			for (int value=p_kh[0][patID][m]; value<=p_kh[0][patID][m+1]; ++value)
			{
				float hanyadreszettettukmegazutnak = (float)(value-p_kh[0][patID][m]) / (float)(p_kh[0][patID][m+1]-p_kh[0][patID][m]);
				coltransT1[value-1] = y_k_avg[0][m] + (y_k_avg[0][m+1]-y_k_avg[0][m]) * hanyadreszettettukmegazutnak;
			}
		}

		float coltransT2[MAX_INTENSITY]={0};
		for (int value=0; value<p_kh[1][patID][0]; ++value)
			coltransT2[value] = 0.0f;
		for (int value=p_kh[1][patID][0]; value<MAX_INTENSITY; ++value)
			coltransT2[value] = 1.0f;
		for (int m=0; m<M-1; ++m)
		{
			for (int value=p_kh[1][patID][m]; value<=p_kh[1][patID][m+1]; ++value)
			{
				float hanyadreszettettukmegazutnak = (float)(value-p_kh[1][patID][m]) / (float)(p_kh[1][patID][m+1]-p_kh[1][patID][m]);
				coltransT2[value-1] = y_k_avg[1][m] + (y_k_avg[1][m+1]-y_k_avg[1][m]) * hanyadreszettettukmegazutnak;
			}
		}

		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* F = fopen(fname,"rb");
		int nrBytes = 0;
		nrBytes += fread(bufferT1,2,VOLSIZE,F);
		nrBytes += fread(bufferT2,2,VOLSIZE,F);
		nrBytes += fread(bufferGT,1,VOLSIZE,F);
		fclose(F);

		float Q = 1.0f;
		for (int i=0; i<qBitDepth; ++i) Q = 2.0f*Q;
		for (int pix=0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				int value = bufferT1[pix];
				int newValue = 1 + cvRound((Q-2.0) * coltransT1[value]);
				bufferT1[pix] = newValue;

				value = bufferT2[pix];
				newValue = 1 + cvRound((Q-2.0) * coltransT2[value]);
				bufferT2[pix] = newValue;
			}

		sprintf(fname,"babainput/baba-A2-%d-%d-%d-%d.bab",patID,qBitDepth,M,cvRound(1000*p_lo));
		F = fopen(fname,"wb");
		fwrite(bufferT1,2,VOLSIZE,F);
		fwrite(bufferT2,2,VOLSIZE,F);
		fwrite(bufferGT,1,VOLSIZE,F);
		fclose(F);


		

	}

/*
		for (int value=0; value<MAX_INTENSITY; ++value)
		{


			float newValue = a * (value+1) + b;
			if (newValue < 0.0f) newValue = 0.0f;
			if (newValue > 1.0f) newValue = 1.0f;
			coltransT1[value] = newValue;
		}
		
		index = 0;
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			for (int i=0; i<histT2[value]; ++i)
				flat[index++] = value+1;
		}

		p25 = flat[pixCount/4];
		p75 = flat[pixCount*3/4];

		a = (lambda75-lambda25)/((float)(p75-p25));
		b = lambda25 - a * (float)p25;

		float coltransT2[MAX_INTENSITY]={0};
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			float newValue = a * (value+1) + b;
			if (newValue < 0.0f) newValue = 0.0f;
			if (newValue > 1.0f) newValue = 1.0f;
			coltransT2[value] = newValue;
		}		
		
		

	}*/
}

int* HistFuzzyCMeans(int* hist, int len, int lo, int hi, int c)
{
	float v[MAX_MILESTONES];
	float u[MAX_MILESTONES];
	float sumUp[MAX_MILESTONES];
	float sumDn[MAX_MILESTONES];
	int* res = (int*)malloc(c*sizeof(int));

	for (int i=0; i<c; ++i)
		v[i] = (float)lo + (float)(hi-lo) * (float)(i+1)/(float)(c+1);
	// fuzzy c-means: m=2.0f; -2.0f/(m-1.0f)=-2.0f
	for (int cycle=0; cycle<20; ++cycle)
	{
		for (int i=0; i<c; ++i) 
		{
			sumUp[i] = 0.0f;
			sumDn[i] = 0.0f;
		}
		for (int value=lo; value<=hi; ++value)
		{
			int match = -1;
			for (int i=0; i<c; ++i) 
			{
				u[i]=(v[i]-value)*(v[i]-value);
				if (u[i] < 0.00001f) match = i; 
			}
			if (match >= 0) // ritka eset
			{
				for (int i=0; i<c; ++i) u[i]=0.0f;
				u[match] = 1.0f;
			}
			else
			{
				float sum = 0.0f;
				for (int i=0; i<c; ++i)
				{
					u[i] = 1.0f/u[i];
					sum += u[i];
				}
				for (int i=0; i<c; ++i)
					u[i] /= sum;
			}

			for (int i=0; i<c; ++i)
			{
				sumUp[i] += hist[value]*u[i]*u[i]*value;
				sumDn[i] += hist[value]*u[i]*u[i];
			}
		}

		for (int i=0; i<c; ++i)
			v[i] = sumUp[i] / sumDn[i]; 	
	}

	for (int i=0; i<c; ++i)
		res[i] = cvRound(v[i]);
	return res;
}

void HistNormRokaTransformX(int qBitDepth = 8, float pLO = 0.02f, int nrMilestones = 7)
{
	int fuzzyMileStones[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	int M = nrMilestones;
	if (M < 3) M = 3;
	if (M > 11) M = 11;

	float p_lo = pLO;
	if (p_lo < 0.01f) p_lo = 0.01;
	if (p_lo > 0.05f) p_lo = 0.05;
	float p_hi = 1.0f - p_lo;

	char fname[100];
	unsigned short* dataBuffers[OBSERVED_CHANNELS];
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		dataBuffers[ch] = (unsigned short*)malloc(VOLSIZE * sizeof(unsigned short));

	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	unsigned short* flat = (unsigned short*)malloc(MILLIO * sizeof(unsigned short));

	float a[OBSERVED_CHANNELS][MAX_PATID], b[OBSERVED_CHANNELS][MAX_PATID];
	int p_kh[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	float y_kh[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	float y_k_avg[OBSERVED_CHANNELS][MAX_MILESTONES] = { 0 };

	for (int patID = 0; patID < MAX_PATID; ++patID)
	{
		sprintf(fname, "babainput/baba-%d.bab", patID);
		FILE* F = fopen(fname, "rb");
		int nrBytes = 0;
		for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
			nrBytes += fread(dataBuffers[ch], 2, VOLSIZE, F);
		nrBytes += fread(bufferGT, 1, VOLSIZE, F);
		fclose(F);
		// if (nrBytes<3*VOLSIZE) printf("Data reading failed");
		int hist[OBSERVED_CHANNELS][MAX_INTENSITY] = { 0 };

		int pixCount = 0;
		for (int pix = 0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
					hist[ch][dataBuffers[ch][pix] - 1]++;
				++pixCount;
			}

		int index, p1, p99, p_ms;

		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		{
			index = 0;
			for (int value = 0; value < MAX_INTENSITY; ++value)
			{
				for (int i = 0; i < hist[ch][value]; ++i)
					flat[index++] = value + 1;
			}

			p1 = flat[cvRound((float)pixCount * p_lo)];
			p99 = flat[cvRound((float)pixCount * p_hi)];

			int* res = HistFuzzyCMeans(hist[ch], MAX_INTENSITY, p1, p99, M - 2);
			for (int i = 0; i < M - 2; ++i)
				fuzzyMileStones[ch][patID][i] = res[i];
			free(res);

			a[ch][patID] = 1.0f / ((float)(p99 - p1));
			b[ch][patID] = -a[ch][patID] * (float)p1;

			y_kh[ch][patID][0] = 0.0f;
			y_kh[ch][patID][M - 1] = 1.0f;
			p_kh[ch][patID][0] = p1;
			p_kh[ch][patID][M - 1] = p99;
			for (int i = 0; i < M - 2; ++i)
			{
				//int ms = mileStones[M][i];
				p_ms = fuzzyMileStones[ch][patID][i];
				y_kh[ch][patID][i + 1] = a[ch][patID] * p_ms + b[ch][patID];
				p_kh[ch][patID][i + 1] = p_ms;
			}
		}
	}

	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		for (int patID = 0; patID < MAX_PATID; ++patID)
		{
			for (int m = 0; m < M; ++m)
				y_k_avg[ch][m] += y_kh[ch][patID][m] / (float)MAX_PATID;
		}

	for (int patID = 0; patID < MAX_PATID; ++patID)
	{
		float coltrans[OBSERVED_CHANNELS][MAX_INTENSITY] = { 0 };
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		{
			for (int value = 0; value < p_kh[ch][patID][0]; ++value)
				coltrans[ch][value] = 0.0f;
			for (int value = p_kh[ch][patID][0]; value < MAX_INTENSITY; ++value)
				coltrans[ch][value] = 1.0f;
			for (int m = 0; m < M - 1; ++m)
			{
				// [m,m+1]:   [p_kh[0][patID][m] --> p_kh[0][patID][m+1]]  [y_k_avg[0][m] --> y_k_avg[0][m+1]]
				for (int value = p_kh[ch][patID][m]; value <= p_kh[ch][patID][m + 1]; ++value)
				{
					float hanyadreszettettukmegazutnak = (float)(value - p_kh[ch][patID][m]) / (float)(p_kh[ch][patID][m + 1] - p_kh[ch][patID][m]);
					coltrans[ch][value - 1] = y_k_avg[ch][m] + (y_k_avg[ch][m + 1] - y_k_avg[ch][m]) * hanyadreszettettukmegazutnak;
				}
			}
		}

		sprintf(fname, "babainput/baba-%d.bab", patID);
		FILE* F = fopen(fname, "rb");
		int nrBytes = 0;
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
			nrBytes += fread(dataBuffers[ch], 2, VOLSIZE, F);
		nrBytes += fread(bufferGT, 1, VOLSIZE, F);
		fclose(F);

		float Q = 1.0f;
		for (int i = 0; i < qBitDepth; ++i) Q = 2.0f * Q;
		for (int pix = 0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
				{
					int value = dataBuffers[ch][pix];
					int newValue = 1 + cvRound((Q - 2.0) * coltrans[ch][value]);
					dataBuffers[ch][pix] = newValue;
				}
			}

		sprintf(fname, "babainput/baba-A3-%d-%d-%d-%d.bab", patID, qBitDepth, M, cvRound(1000 * p_lo));
		F = fopen(fname, "wb");
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
			fwrite(dataBuffers[ch], 2, VOLSIZE, F);
		fwrite(bufferGT, 1, VOLSIZE, F);
		fclose(F);
	}
	generateFeatures(3, qBitDepth, 0, pLO, nrMilestones);
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);
}


void HistNormRokaTransform(int qBitDepth = 8, float pLO = 0.02f, int nrMilestones = 7)
{
	int fuzzyMileStones[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	int M = nrMilestones;
	if (M < 3) M = 3;
	if (M > 11) M = 11;

	float p_lo = pLO;
	if (p_lo < 0.01f) p_lo = 0.01;
	if (p_lo > 0.05f) p_lo = 0.05;
	float p_hi = 1.0f - p_lo;

	char fname[100];
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);
	
	unsigned short* flat = (unsigned short*)malloc(MILLIO*2);
	
	float a[OBSERVED_CHANNELS][MAX_PATID], b[OBSERVED_CHANNELS][MAX_PATID];
	int p_kh[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	float y_kh[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	float y_k_avg[OBSERVED_CHANNELS][MAX_MILESTONES] = {0};

	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* F = fopen(fname,"rb");
		int nrBytes = 0;
		nrBytes += fread(bufferT1,2,VOLSIZE,F);
		nrBytes += fread(bufferT2,2,VOLSIZE,F);
		nrBytes += fread(bufferGT,1,VOLSIZE,F);
		fclose(F);
		// if (nrBytes<3*VOLSIZE) printf("Data reading failed");
		int histT1[MAX_INTENSITY]={0};
		int histT2[MAX_INTENSITY]={0};

		int pixCount = 0;
		for (int pix=0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				histT1[ bufferT1[pix]-1 ]++;
				histT2[ bufferT2[pix]-1 ]++;
				++pixCount;
			}
	
		int index, p1, p99, p_ms;

		index = 0;
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			for (int i=0; i<histT1[value]; ++i)
				flat[index++] = value+1;
		}

		p1 = flat[cvRound((float)pixCount*p_lo)];
		p99 = flat[cvRound((float)pixCount*p_hi)];

		int* res = HistFuzzyCMeans(histT1,MAX_INTENSITY,p1,p99,M-2);
		for (int i=0; i<M-2; ++i)
			fuzzyMileStones[0][patID][i] = res[i];
		free(res);

		a[0][patID] = 1.0f/((float)(p99-p1));
		b[0][patID] = -a[0][patID] * (float)p1;

		y_kh[0][patID][0] = 0.0f;
		y_kh[0][patID][M-1] = 1.0f;
		p_kh[0][patID][0] = p1;
		p_kh[0][patID][M-1] = p99;
		for (int i=0; i<M-2; ++i)
		{
			//int ms = mileStones[M][i];
			p_ms = fuzzyMileStones[0][patID][i];
			y_kh[0][patID][i+1] = a[0][patID] * p_ms + b[0][patID];
			p_kh[0][patID][i+1] = p_ms;
		}

		index = 0;
		for (int value=0; value<MAX_INTENSITY; ++value)
		{
			for (int i=0; i<histT2[value]; ++i)
				flat[index++] = value+1;
		}

		p1 = flat[cvRound((float)pixCount*p_lo)];
		p99 = flat[cvRound((float)pixCount*p_hi)];

		res = HistFuzzyCMeans(histT2,MAX_INTENSITY,p1,p99,M-2);
		for (int i=0; i<M-2; ++i)
			fuzzyMileStones[1][patID][i] = res[i];
		free(res);

		a[1][patID] = 1.0f/((float)(p99-p1));
		b[1][patID] = -a[1][patID] * (float)p1;

		y_kh[1][patID][0] = 0.0f;
		y_kh[1][patID][M-1] = 1.0f;
		p_kh[1][patID][0] = p1;
		p_kh[1][patID][M-1] = p99;		
		for (int i=0; i<M-2; ++i)
		{
			//int ms = mileStones[M][i];
			p_ms = fuzzyMileStones[1][patID][i];
			y_kh[1][patID][i+1] = a[1][patID] * p_ms + b[1][patID];
			p_kh[1][patID][i+1] = p_ms;
		}
	}

	for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
		for (int patID = 0; patID<MAX_PATID; ++patID)	
		{
			for (int m = 0; m<M; ++m)
				y_k_avg[ch][m] += y_kh[ch][patID][m] / (float)MAX_PATID;
		}

	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		float coltransT1[MAX_INTENSITY]={0};
		for (int value=0; value<p_kh[0][patID][0]; ++value)
			coltransT1[value] = 0.0f;
		for (int value=p_kh[0][patID][0]; value<MAX_INTENSITY; ++value)
			coltransT1[value] = 1.0f;
		for (int m=0; m<M-1; ++m)
		{
			// [m,m+1]:   [p_kh[0][patID][m] --> p_kh[0][patID][m+1]]  [y_k_avg[0][m] --> y_k_avg[0][m+1]]
			for (int value=p_kh[0][patID][m]; value<=p_kh[0][patID][m+1]; ++value)
			{
				float hanyadreszettettukmegazutnak = (float)(value-p_kh[0][patID][m]) / (float)(p_kh[0][patID][m+1]-p_kh[0][patID][m]);
				coltransT1[value-1] = y_k_avg[0][m] + (y_k_avg[0][m+1]-y_k_avg[0][m]) * hanyadreszettettukmegazutnak;
			}
		}

		float coltransT2[MAX_INTENSITY]={0};
		for (int value=0; value<p_kh[1][patID][0]; ++value)
			coltransT2[value] = 0.0f;
		for (int value=p_kh[1][patID][0]; value<MAX_INTENSITY; ++value)
			coltransT2[value] = 1.0f;
		for (int m=0; m<M-1; ++m)
		{
			for (int value=p_kh[1][patID][m]; value<=p_kh[1][patID][m+1]; ++value)
			{
				float hanyadreszettettukmegazutnak = (float)(value-p_kh[1][patID][m]) / (float)(p_kh[1][patID][m+1]-p_kh[1][patID][m]);
				coltransT2[value-1] = y_k_avg[1][m] + (y_k_avg[1][m+1]-y_k_avg[1][m]) * hanyadreszettettukmegazutnak;
			}
		}

		sprintf(fname,"babainput/baba-%d.bab",patID);
		FILE* F = fopen(fname,"rb");
		int nrBytes = 0;
		nrBytes += fread(bufferT1,2,VOLSIZE,F);
		nrBytes += fread(bufferT2,2,VOLSIZE,F);
		nrBytes += fread(bufferGT,1,VOLSIZE,F);
		fclose(F);

		float Q = 1.0f;
		for (int i=0; i<qBitDepth; ++i) Q = 2.0f*Q;
		for (int pix=0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
			{
				int value = bufferT1[pix];
				int newValue = 1 + cvRound((Q-2.0) * coltransT1[value]);
				bufferT1[pix] = newValue;

				value = bufferT2[pix];
				newValue = 1 + cvRound((Q-2.0) * coltransT2[value]);
				bufferT2[pix] = newValue;
			}

		sprintf(fname,"babainput/baba-A3-%d-%d-%d-%d.bab",patID,qBitDepth,M,cvRound(1000*p_lo));
		F = fopen(fname,"wb");
		fwrite(bufferT1,2,VOLSIZE,F);
		fwrite(bufferT2,2,VOLSIZE,F);
		fwrite(bufferGT,1,VOLSIZE,F);
		fclose(F);
	}
}


void xSamiBaba()
{
	int hist[20000]={0};
	char fname[100];
	unsigned short* bufferT1 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned short* bufferT2 = (unsigned short*)malloc(VOLSIZE*2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	IplImage* imT1 = cvCreateImage(cvSize(5*WIDTH,2*HEIGHT),IPL_DEPTH_8U,1);
	IplImage* imT2 = cvCreateImage(cvSize(5*WIDTH,2*HEIGHT),IPL_DEPTH_8U,1);
	cvSet(imT1,cvScalar(0));
	cvSet(imT2,cvScalar(0));
	for (int patID = 0; patID<MAX_PATID; ++patID)
	{
		int sliceCount[MAX_SLICES] = {0};
		int maxSliceCount = 0;
		int maxSlice = 0;

		//sprintf(fname,"babainput/baba-A3-%d-8-7-20.bab",patID);
		sprintf(fname,"babainput/baba-A1X-%d-8-38.bab",patID);
		FILE* F = fopen(fname,"rb");
		fread(bufferT1,2,VOLSIZE,F);
		fread(bufferT2,2,VOLSIZE,F);
		fread(bufferGT,1,VOLSIZE,F);
		fclose(F);

/*		int count = 0;
		int count1 = 0;
		int count2 = 0;
		int maxT1 = 0;
		int maxT2 = 0;
		int minT1 = 9999;
		int minT2 = 9999;
		int sumT1 = 0;
		int sumT2 = 0;*/
		for (int i=0; i<VOLSIZE; ++i)
		{
//			bufferT1[i] = ((bufferT1[i]%256)*256 + (bufferT1[i]/256))%32768;
//			bufferT2[i] = ((bufferT2[i]%256)*256 + (bufferT2[i]/256))%32768;
			if (bufferT1[i]>0) sliceCount[i/(WIDTH*HEIGHT)]++;
		}
		for (int i=1; i<MAX_SLICES; ++i)
			if (maxSliceCount < sliceCount[i])
			{
				maxSliceCount = sliceCount[i];
				maxSlice = i;
			}
		printf("MaxSlice: %d %d %d \n",patID,maxSlice,maxSliceCount);

		for (int x=0; x<WIDTH; ++x) for (int y=0; y<HEIGHT; ++y)
		{
			int i = WIDTH*HEIGHT*maxSlice + y*WIDTH + x;
			if (bufferT1[i]>0)
			{
				int val = bufferT1[i];  //1 + cvRound((float)(bufferT1[i]) * 0.25);
				setGray(imT1,(patID%5)*WIDTH+x, (patID/5)*HEIGHT+y, val);
			}
			if (bufferT2[i]>0)
			{
				int val = bufferT2[i];  //1 + cvRound((float)(bufferT1[i]) * 0.25);
				setGray(imT2,(patID%5)*WIDTH+x, (patID/5)*HEIGHT+y, val);
			}
		}

	}

	cvSaveImage("kimenetT1.png",imT1);
	cvSaveImage("kimenetT2.png",imT2);
	cvShowImage("T1",imT1);
	cvShowImage("T2",imT2);
	cvWaitKey();
	//getch();
	free(bufferT1);
	free(bufferT2);
	free(bufferGT);

}


void xSamiTumor()
{
	const int SZAZHATVAN = 160;
//	const int HEIGHT = 192;
	const int VOLTAREN = 155 * 160 * 192;
	const int NRCH = 4;
//	int hist[20000] = { 0 };
	char fname[100];
	unsigned short* buffer[4];
	for (int ch=0; ch<NRCH; ++ch)
		buffer[ch] = (unsigned short*)malloc(VOLTAREN * 2);
	unsigned char* bufferGT = (unsigned char*)malloc(VOLTAREN);

	IplImage* im[NRCH];
	for (int ch = 0; ch < NRCH; ++ch)
	{
		im[ch] = cvCreateImage(cvSize(10 * SZAZHATVAN, 5 * HEIGHT), IPL_DEPTH_8U, 1);
		cvSet(im[ch], cvScalar(0));
	}
	for (int patID = 0; patID < 50; ++patID)
	{
		int sliceCount[MAX_SLICES] = { 0 };
		int maxSliceCount = 0;
		int maxSlice = 0;

		//sprintf(fname,"babainput/baba-A3-%d-8-7-20.bab",patID);
		sprintf(fname, "tumorinput/lg19-A1-%02d-8-38.tum", patID);
		FILE* F = fopen(fname, "rb");
		for (int ch = 0; ch < NRCH; ++ch)
			fread(buffer[ch], 2, VOLTAREN, F);
		fread(bufferGT, 1, VOLTAREN, F);
		fclose(F);

		for (int i = 0; i < VOLTAREN; ++i)
		{
			//			bufferT1[i] = ((bufferT1[i]%256)*256 + (bufferT1[i]/256))%32768;
			//			bufferT2[i] = ((bufferT2[i]%256)*256 + (bufferT2[i]/256))%32768;
			if (bufferGT[i] > 0) sliceCount[i / (160 * 192)]++;
		}
		for (int i = 1; i < MAX_SLICES; ++i)
			if (maxSliceCount < sliceCount[i])
			{
				maxSliceCount = sliceCount[i];
				maxSlice = i;
			}
		printf("MaxSlice: %d %d %d \n", patID, maxSlice, maxSliceCount);

		for (int x = 0; x < SZAZHATVAN; ++x) for (int y = 0; y < HEIGHT; ++y)
		{
			int i = SZAZHATVAN * HEIGHT * maxSlice + y * SZAZHATVAN + x;
			for (int ch = 0; ch < NRCH; ++ch)
			if (buffer[ch][i] > 0)
			{
				int val = buffer[ch][i];  //1 + cvRound((float)(bufferT1[i]) * 0.25);
				setGray(im[ch], (patID % 10) * SZAZHATVAN + x, (patID / 10) * HEIGHT + y, val);
			}
		}

	}

//	cvSaveImage("kimenetT1.png", imT1);
//	cvSaveImage("kimenetT2.png", imT2);
	cvShowImage("T1", im[0]);
	cvShowImage("T2", im[1]);
	cvShowImage("T1C", im[2]);
	cvShowImage("FLAIR", im[3]);
	cvWaitKey();
	//getch();
	for (int ch = 0; ch < NRCH; ++ch)
		free(buffer[ch]);
	free(bufferGT);

}



int selectedVolume(int o)
{
	if (o < 10) return 0;
	if (o == 15) return 0;
	if (o == 20) return 0;
	if (o == 48) return 0;
	if (o == 54) return 0;
	if (o == 55) return 0;
	if (o == 56) return 0;
	if (o == 62) return 0;
	if (o == 63) return 0;
	if (o == 64) return 0;
	if (o == 67) return 0;
	if (o == 69) return 0;
	if (o == 70) return 0;
	if (o == 71) return 0;
	if (o > 72) return 0;
	return 1;
}

void brats1()
{
	const int HETVENHAT = 259;
	const int NEGY = 4;
	const int BUFFSIZE = 240 * 240 * 155;

	short avgVals[NEGY][HETVENHAT];
	char fname[100];
	unsigned short* buffer = (unsigned short*)malloc(sizeof(short) * BUFFSIZE);
	unsigned short* flat = (unsigned short*)malloc(sizeof(short) * MILLIO * 2);
	short* bufferData[NEGY];
	for (int ch = 0; ch < NEGY; ++ch)
	{
		bufferData[ch] = (short*)malloc(sizeof(short) * BUFFSIZE);
	}
	char* bufferGT = (char*)malloc(sizeof(char) * BUFFSIZE);
	short* saveData[NEGY];
	for (int ch = 0; ch < NEGY; ++ch)
	{
		saveData[ch] = (short*)malloc(sizeof(short) * 200 * 180 * 155);
	}
	char* saveGT = (char*)malloc(sizeof(char) * 200 * 180 * 155);

	int maxX = 0;
	int maxY = 0;
	int minX = 240;
	int minY = 240;

	int realPaci = -1;
	for (int paci = 0; paci < HETVENHAT; ++paci) //if (selectedVolume(paci))
	{
		int replacement[NEGY];
		++realPaci;
		int validPix = 0;
		sprintf(fname, "input-hg-2019/input_vol%03d_ch%d.vol", paci, 4);
		FILE* F = fopen(fname, "rb");
		fread(buffer, sizeof(short), BUFFSIZE, F);
		fclose(F);
		for (int i = 0; i < BUFFSIZE; ++i)
		{
			bufferGT[i] = -1;
			if (buffer[i] > 0)
			{
				bufferGT[i] = buffer[i];
				++validPix;
			}
		}
		for (int ch = 0; ch < NEGY; ++ch)
		{
			sprintf(fname, "input-hg-2019/input_vol%03d_ch%d.vol", paci, ch);
			FILE* F = fopen(fname, "rb");
			fread(bufferData[ch], sizeof(short), BUFFSIZE, F);
			fclose(F);

			int count = 0;
			int maxi = 0;
			float sum = 0.0f;
			int mini = 32767;
			int hist[0x8000] = { 0 };
			for (int i = 0; i < BUFFSIZE; ++i)
			{
				int value = bufferData[ch][i];
				if (value > 0)
				{
					count++;
					hist[value]++;
					sum += value;
					if (maxi < value) maxi = value;
					if (mini > value) mini = value;
					if (bufferGT[i] < 0)
					{
						bufferGT[i] = 0;
						++validPix;
					}
				}
			}
			int index = 0;
			for (int h = 0; h < 0x8000; ++h)
			{
				for (int o = 0; o < hist[h]; ++o)
				{
					flat[index++] = h;
				}
			}

			int p1 = flat[index / 100];
			int p99 = flat[index - index / 100];
			replacement[ch] = flat[index / 2];

			avgVals[ch][paci] = cvRound(sum / count);
			F = fopen("min-max.csv", "at");
			fprintf(F, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", paci, ch, count, index, validPix, mini, p1, cvRound(sum / count), p99, maxi);
		//	printf("%d,%d,%d,%d,[%d],%d,%d,[%d],%d,%d\n", paci, ch, count, index, validPix, mini, p1, cvRound(sum / count), p99, maxi);
			fclose(F);
		}

		int missing = 0;
		int wrong = 0;
		for (int i = 0; i < BUFFSIZE; ++i)
			if (bufferGT[i] >= 0)
			{
				if ((i % 240) > maxX) maxX = (i % 240);
				if ((i % 240) < minX) minX = (i % 240);
				if ((i%57600) / 240 > maxY) maxY = (i % 57600) / 240;
				if ((i%57600) / 240 < minY) minY = (i % 57600) / 240;

				int q = 0;
				int qq = 0;
				for (int ch = 0; ch < NEGY; ++ch)
				{
					if (bufferData[ch][i] <= 0)
					{
						int sum = 0;
						++q;
						int o = 0;
						for (int dx = -3; dx <= 3; ++dx) for (int dy = -3; dy <= 3; ++dy)
							if (bufferData[ch][i + dx + 240 * dy] > 0)
							{
								++o;
								sum += bufferData[ch][i + dx + 240 * dy];
							}
						if (!o)
						{
							++qq;
							bufferData[ch][i] = -replacement[ch];
						}
						else bufferData[ch][i] = -cvRound(sum / o);
					}
				}
				if (q > 0) ++missing;
				if (qq > 0) ++wrong;
			}

		printf("[%d,%d]", missing, wrong);
		
		int db = 0;
		for (int i = 0; i < BUFFSIZE; ++i)
			if (bufferGT[i] >= 0)
				for (int ch = 0; ch < NEGY; ++ch)
					if (bufferData[ch][i] < 0)
					{
						++db;
						bufferData[ch][i] = -bufferData[ch][i];
					}
		printf("<%6d>\n", db);

		int index = 0;
		for (int i = 0; i < BUFFSIZE; ++i)
		{
			int x = i % 240;
			int y = (i % 57600) / 240;
			if (x >= 36 && x < 216 && y >= 24 && y < 224)
			{
				for (int ch = 0; ch < NEGY; ++ch)	
					saveData[ch][index] = bufferData[ch][i];
				saveGT[index] = 1 + bufferGT[i];
				++index;
			}
		}
		printf("%d",index);
		sprintf(fname, "input-hg-2019/hg19-%03d.tum", realPaci);
		F = fopen(fname, "wb");
		for (int ch = 0; ch < NEGY; ++ch)
			fwrite(saveData[ch], sizeof(short), 200 * 180 * 155, F);
		fwrite(saveGT, sizeof(char), 200 * 180 * 155, F);
		fclose(F);
	}
	printf("<%d,%d -- %d,%d>", minX, minY, maxX, maxY);
}
//40,30,160,192  
//36,24,180,200

void brats1_5()
{
	const int HETVENHAT = 259;
	const int NEGY = 4;
	const int BUFFSIZE = 180 * 200 * 155;

	char fname[100];
	short* bufferData[NEGY];
	for (int ch = 0; ch < NEGY; ++ch)
		bufferData[ch] = (short*)malloc(sizeof(short) * BUFFSIZE);
	char* bufferGT = (char*)malloc(sizeof(char) * BUFFSIZE);

	for (int paci = 0; paci < HETVENHAT; ++paci)
	{
		sprintf(fname, "input-hg-2019/hg19-%03d.tum", paci);
		FILE* F = fopen(fname, "rb");
		for (int ch = 0; ch < NEGY; ++ch)
			fread(bufferData[ch], sizeof(short), BUFFSIZE, F);
		fread(bufferGT, sizeof(char), BUFFSIZE, F);
		fclose(F);

		int count = 0; 
		int wrong = 0;
		for (int i = 0; i < BUFFSIZE; ++i)
		{
			if (bufferGT[i] > 0)
			{
				++count;
				for (int ch = 0; ch < NEGY; ++ch)
					if (bufferData[ch][i] == 0) ++wrong;
			}
		}
		printf("Volume %03d(%d)...",paci,count);
		if (wrong == 0) printf("OK\n");
		else printf("missing %d values\n", wrong);
	}
}

void brats2()
{
	const int MAXPACI = 259;
	const int NEGY = 4;
	const int BUFFSIZE = 180 * 200 * 155;

	char fname[100];
	short* bufferData[NEGY];
	for (int ch = 0; ch < NEGY; ++ch)
		bufferData[ch] = (short*)malloc(sizeof(short) * BUFFSIZE);
	char* bufferGT = (char*)malloc(sizeof(char) * BUFFSIZE);

	for (int paci = 0; paci < MAXPACI; ++paci) 
	{
		sprintf(fname, "input-hg-2019/hg19-%03d.tum", paci);
		FILE* F = fopen(fname, "rb");
		for (int ch = 0; ch < NEGY; ++ch)
			fread(bufferData[ch], sizeof(short), BUFFSIZE, F);
		fread(bufferGT, sizeof(char), BUFFSIZE, F);
		fclose(F);

		int db = 0;
		int missing = 0;
		for (int i = 0; i < BUFFSIZE; ++i)
		{
			if (bufferGT[i] >= 0)
			{
				++db;
				for (int ch = 0; ch < NEGY; ++ch)
					if (bufferData[ch][i] <= 0) ++missing;
			}
		}
		printf("<%d,%d>\n", db, missing);
	}
}



void buildData(int which, int bits)
{
	int tails[6] = { 1,3,5,10,20,30 };

	if (which == INFANT_DATA)
	{
		for (int i = 36; i <= 48; ++i)
		{
			HistNormLinear tr(INFANT_DATA, bits, 0.01f * i);
			tr.run();
		}
		for (int i = 0; i <= 5; ++i)
			for (int j = 0; j <= 11; j += 1)
			{
				HistNormNyul tr(INFANT_DATA, bits, 0.001f * tails[i], j);
				tr.run();
//				HistNormRoka tr2(INFANT_DATA, bits, 0.001f * i, j);
//				tr2.run();
			}
	}
	else
	{
/*		for (int i=36; i<=48; ++i)
		{ 
			HistNormLinear tr(TUMOR_DATA, bits, 0.01f * i);
			tr.run();
		}*/
		for (int i = 0; i <= 5; ++i)
			for (int j = 0; j <= 11; j += 1)
			{
				HistNormNyul tr(TUMOR_DATA, bits, 0.001f * tails[i], j);
				tr.run();
//				HistNormRoka tr2(TUMOR_DATA, bits, 0.001f * i, j);
//				tr2.run();
			}
	}
}


void slicer()
{
	const int NRPIX = 0x4400000;
	const int NR_FEAT = 36;
	const int NR_PACI = 50;
	const int W = 160;
	const int H = 192;
	const int S = H*W;

	FILE* F = fopen("tumor/tumor-A1-8-38.dat", "rb");
	int head[3];
	fread(head, sizeof(int), 3, F);

	PatientBasicData pbd[50];
	fread(pbd, sizeof(PatientBasicData), NR_PACI, F);

	uchar* GTBuffer = (uchar*)malloc(NRPIX);
	fread(GTBuffer, sizeof(uchar), head[0], F);

	uchar** PosBuffer = (uchar**)malloc(NR_COORDS * sizeof(uchar*));
	for (int c = 0; c < NR_COORDS; ++c)
	{
		PosBuffer[c] = (uchar*)malloc(NRPIX);
		fread(PosBuffer[c], sizeof(uchar), head[0], F);
	}
	uchar** FeatBuffer = (uchar**)malloc(NR_FEAT * sizeof(uchar*));
	for (int f = 0; f < NR_FEATURES; ++f)
	{
		FeatBuffer[f] = (uchar*)malloc(NRPIX);
		fread(FeatBuffer[f], sizeof(uchar), head[0], F);
	}

	IplImage* imFrame;
	IplImage* gtImage;
	IplImage* featImages[NR_FEAT];
	for (int f = 0; f < NR_FEATURES; ++f)
	{
		featImages[f] = cvCreateImage(cvSize(SLICES_PER_ROW * W, 10 * H), IPL_DEPTH_8U, 1);
	}
	gtImage = cvCreateImage(cvSize(SLICES_PER_ROW * W, 10 * H), IPL_DEPTH_8U, 1);
	imFrame = cvCreateImage(cvSize(5 * W, H), IPL_DEPTH_8U, 1);

	for (int paci = 0; paci < NR_PACI; ++paci)
	{
		for (int f = 0; f < NR_FEATURES; ++f)
			cvSet(featImages[f], cvScalar(0));
		cvSet(gtImage, cvScalar(0));

		for (int pix = pbd[paci].firstIndex; pix <= pbd[paci].lastIndex; ++pix)
		{
			int x = PosBuffer[3][pix];
			int y = PosBuffer[2][pix];
			int s = PosBuffer[1][pix];
			int X = (s % SLICES_PER_ROW) * W + x;
			int Y = (s / SLICES_PER_ROW) * H + y;

			for (int f = 0; f < NR_FEATURES; ++f)
				setGray(featImages[f], X, Y, FeatBuffer[f][pix]);

			setGray(gtImage, X, Y, (GTBuffer[pix]>1)*128+127);
		}
	//	cvShowImage("T2", gtImage);
	//	cvWaitKey();
		char fname[100];
		for (int slice = 0; slice < 155; ++slice)
		{
			for (int f = 0; f < 4; ++f)
			{
				cvSetImageROI(featImages[f], cvRect((slice % SLICES_PER_ROW) * W, (slice / SLICES_PER_ROW) * H, W, H));
				cvSetImageROI(imFrame, cvRect(f * W, 0, W, H));
				cvCopy(featImages[f], imFrame);
				cvResetImageROI(featImages[f]);
			}
			cvSetImageROI(gtImage, cvRect((slice % SLICES_PER_ROW) * W, (slice / SLICES_PER_ROW) * H, W, H));
			cvSetImageROI(imFrame, cvRect(4 * W, 0, W, H));
			cvCopy(gtImage, imFrame);
			cvResetImageROI(gtImage);
			cvResetImageROI(imFrame);

			sprintf(fname,"gifdata/paci%02d-%03d.png", paci, slice);
			cvSaveImage(fname, imFrame);
		}
		cvCmpS(featImages[0], 1, featImages[0], CV_CMP_GE);
		sprintf(fname, "gifdata/mask%02d.png", paci);
		cvSaveImage(fname, featImages[0]);
		printf("[%d]", paci);
	}
}


int howManyWhites(IplImage* im)
{
	CvScalar avg, sdv;
	cvAvgSdv(im, &avg, &sdv);
	return cvRound(avg.val[0] * (float)(im->width * im->height) / 255.0f);
}

float postproc(IplImage* im, IplImage* imM, int& xtp, int& xfp, int& xfn)
{
	const int NEIGH = 5;
	IplImage* imRe = cvCloneImage(im);
	cvCmpS(im, 160, imRe, CV_CMP_GE);
	int tp = howManyWhites(imRe);
	cvCmpS(im, 96, imRe, CV_CMP_GE);
	int fp = howManyWhites(imRe) - tp;
	cvCmpS(im, 32, imRe, CV_CMP_GE);
	int fn = howManyWhites(imRe) - tp - fp;
//	cvCmpS(im, 96, imRe, CV_CMP_GE);

	xtp = 0;
	xfp = 0; 
	xfn = 0;
	for (int s=0; s<155; ++s)
		for (int y = 0; y < 192; ++y)
			for (int x = 0; x < 160; ++x)
			{
				int X = x + (s % 16) * 160;
				int Y = y + (s / 16) * 192;
				if (getGray(imM, X, Y) > 128)
				{
					int all = 0;
					int poz = 0;
					int gt = (getGray(im, X, Y) % 128 > 32);
					for (int dx = -NEIGH; dx <= NEIGH; ++dx)
						for (int dy = -NEIGH; dy <= NEIGH; ++dy)
							for (int ds = -NEIGH; ds <= NEIGH; ++ds) 
								if(x + dx >= 0 && x + dx < 160 && y + dy >= 0 && y + dy < 192 && s + ds >= 0 && s + ds < 155)
								{
									int XX = (x+dx) + ((s+ds) % 16) * 160;
									int YY = (y+dy) + ((s+ds) / 16) * 192;
									if (getGray(imM, XX, YY) > 128)
									{
										++all;
										if (getGray(im, XX, YY) > 96) ++poz;
									}
								}
					int deci = ((float)poz / (float)all > 0.33333333f);
					if (deci)
					{ 
						if (gt) ++xtp;
						else ++xfp;
					}
					else
					{ 
						if (gt) ++xfn;
					}
				}
			}
	cvReleaseImage(&imRe);
	printf("%d,%d,%d  %d,%d,%d\n", tp, fp, fn, xtp, xfp, xfn);
	return (float)(2*xtp)/(float)(2*xtp+xfp+xfn);
}

void postprocenko()
{
	char fname[100];
	for (int t = 410; t <= 417; ++t)
	{
//		if (t == 211) continue;
//		if (t >= 212 && ((t-212)%12>=5)) continue;

		for (int p = 0; p < 50; ++p)
		{
			int tp, fp, fn;
			sprintf(fname, "images/testres%03d-%02d.png", t, p);
			printf("%s\n", fname);
			IplImage* im = cvLoadImage(fname, 0);
			sprintf(fname, "tumor/mask%02d.png", p);
			IplImage* imM = cvLoadImage(fname, 0);
			float res = postproc(im,imM,tp,fp,fn);
			FILE* F = fopen("postproc.csv", "at");
			fprintf(F, "%d,%d,%.5f,,%d,%d,%d\n", t, p, res, tp, fp, fn);
			fclose(F);
			cvReleaseImage(&im);
			printf(".");
		}
	}
}


void spain()
{
	HistNormNyul tr(TUMOR_DATA, 8, 0.005f, 2);
	tr.run();
	//	brats1_5();
//	buildData(INFANT_DATA, 8);
//		postprocenko();
//	buildData(TUMOR_DATA, 8);
	//	slicer();
	return;

	const int which = 2;
	const int half = 0;

	char fname[100];
	int test = 600;
	/*
		for (int i = 40+2*which; i <= 44; i += 2)
		{
			sprintf(fname, "tumor/tumor-A1-8-%d.dat", i);
			//		sprintf(fname, "babainput/bigbaba-A1-8-%d.dat",86-i);
			Tester tst(fname, test, TUMOR_DATA);
			//		Tester tst(fname, test, INFANT_DATA);
					//	Tester tst("bigbaba-A1-8-38.dat",INFANT_DATA);
			for (int paci = 0; paci < MAX_PATID; ++paci)
				//tst.MultiSVM(paci, 30, 0, 0);
				// tst.TrainAndTest(ALG_KNN, paci, 5000, 11, 0);
					tst.TrainAndTest(ALG_RF, paci, 50000, 24, 0);
			++test;
		}
		return;*/
		const int tails[4] = { 1,3,5,10 };//, 20, 30

		const int schemes[6] = { 0,1,2,4,6,11 };

		//for (int i = 3; i < 4; i++)
		//for (int j = 3*half; j < 3+3*half; j += 1)
		{
			//sprintf(fname, "tumor/tumor-A2-8-%d-%d.dat", schemes[4], tails[2]);
				sprintf(fname, "babainput/bigbaba-A2-8-%d-%d.dat",2,5);
			//Tester tst(fname, test, TUMOR_DATA);
			Tester tst(fname, test, INFANT_DATA, 1);
			for (int paci = 0; paci < MAX_PATID; ++paci)
			{
			//	tst.MultiKNN(paci, 1000, 11, 0);
			//	tst.TrainAndTest(ALG_KNN, paci, 50000, 11, 0);
			//	tst.TrainAndTest(ALG_RF, paci, 1000, 15, 0);
				tst.OcvBoost(paci, 1000, 15, 21);
			}
				//tst.TrainAndTest(ALG_KNN, paci, 5000, 11, 0);
				//tst.MultiSVM(paci, 30, 0, 0);
			++test;
		}

	return;
}

void csongor1()
{
	int maxX = 0;
	int minX = 240;
	int maxY = 0;
	int minY = 240;
	LoadFeatures("tumor/tumor-A2-8-3-5.dat");
	for (int i = 0; i <= patLimits[49].lastIndex; ++i)
	{
		int value = PosBuffer[3][i];
		if (value > maxX) maxX = value;
		if (value < minX) minX = value;
		value = PosBuffer[2][i];
		if (value > maxY) maxY = value;
		if (value < minY) minY = value;
	}

	printf("%d %d\n", minX, maxX);
	printf("%d %d\n", minY, maxY);

	for (int p = 0; p < MAX_PATID; ++p)
		printf("[%d %d]", PosBuffer[1][patLimits[p].firstIndex], PosBuffer[1][patLimits[p].lastIndex]);

	const int TIZENHAT = 24;
	const int KILENC = 144 / TIZENHAT;

	IplImage* imRe = cvCreateImage(cvSize(TIZENHAT * WIDTH, KILENC * HEIGHT), IPL_DEPTH_8U, 1);
	for (int p = 0; p < MAX_PATID; ++p)
	{
		char fname[100];
		for (int ch = 0; ch < 4; ++ch)
		{
			cvSet(imRe, cvScalar(0));
			for (int i = patLimits[p].firstIndex; i <= patLimits[p].lastIndex; ++i)
			{
				int X = PosBuffer[3][i] + (PosBuffer[1][i] % TIZENHAT) * WIDTH;
				int Y = PosBuffer[2][i] + (PosBuffer[1][i] / TIZENHAT) * HEIGHT;
				setGray(imRe, X, Y, FeatBuffer[ch][i]);
			}
			sprintf(fname, "csongor-input/pat%02d-ch%d.png", p, ch);
			cvSaveImage(fname, imRe);
		}
		cvSet(imRe, cvScalar(0));
		for (int i = patLimits[p].firstIndex; i <= patLimits[p].lastIndex; ++i)
		{
			int X = PosBuffer[3][i] + (PosBuffer[1][i] % TIZENHAT) * WIDTH;
			int Y = PosBuffer[2][i] + (PosBuffer[1][i] / TIZENHAT) * HEIGHT;
			setGray(imRe, X, Y, GTBuffer[i] > 1 ? 255 : 128);
		}
		sprintf(fname, "csongor-input/pat%02d-gt.png", p);
		cvSaveImage(fname, imRe);
	}
	cvReleaseImage(&imRe);
	return;
}



void main()
{
//	timestamp();
	char fname[100];
	sprintf(fname, "tumor/tumor-A2-8-3-5.dat", 0);
	Tester tst(fname, 555, TUMOR_DATA, 0);
	for (int paci = 0; paci < 10; ++paci)
	{
		tst.TrainAndTest(ALG_RF, paci, 1000, 15);
	}


	//	buildData(INFANT_DATA, 8);
	//		postprocenko();
	//	buildData(TUMOR_DATA, 8);
		//	slicer();
	//	return;

	const int which = 5;

	//char fname[100];
	int test = 200 + 12 * which;
/*
	if (which == 0)
	{
		for (int i = 38; i <= 48; i += 1)
		{
			sprintf(fname, "tumor/tumor-A1-8-%d.dat", i);
			//		sprintf(fname, "babainput/bigbaba-A1-8-%d.dat",86-i);
			Tester tst(fname, test, TUMOR_DATA, 0);
			//		Tester tst(fname, test, INFANT_DATA);
					//	Tester tst("bigbaba-A1-8-38.dat",INFANT_DATA);
			for (int paci = 0; paci < MAX_PATID; ++paci)
				//		tst.MultiSVM(paci, 30, 0, 0);
				//tst.TrainAndTest(ALG_KNN, paci, 200, 11, 0);
					tst.TrainAndTest(ALG_RF, paci, 10000, 21, 0);
			++test;
		}
	}
	//	return;
	else*/
	{
		const int tails[6] = { 1,3,5,10,20,30 };
		const int schemes[5] = { 0,1,2,4,6 };

		//for (int i = 0; i < 6; i++)
		//for (int j = 0; j < 12; j += 1)
		{
			//sprintf(fname, "tumor/tumor-A2-8-%d-%d.dat", 2, 5);
				sprintf(fname, "babainput/bigbaba-A2-8-%d-%d.dat",2,5);
			Tester tst(fname, test, INFANT_DATA,1);
			//			Tester tst(fname, test, INFANT_DATA);
			//for (int paci = 0; paci < MAX_PATID; ++paci)
			//		tst.TrainAndTest(ALG_RF, paci, 10000, 21, 0);
			//	tst.TrainAndTest(ALG_KNN, paci, 200, 11, 0);
			//	tst.MultiSVM(paci, 30, 0, 0);
			tst.OcvBoost(7,5000,16,21);
/*			sprintf(fname, "tumor/tumor-A3-8-%d-%d.dat", j, i);
			Tester tst2(fname, test+81, TUMOR_DATA);
			//	Tester tst("bigbaba-A1-8-38.dat",INFANT_DATA);
			for (int paci = 0; paci < MAX_PATID; ++paci)
				tst2.TrainAndTest(ALG_RF, paci, 2000, 18, 0);
				*/
			++test;
		}
	}

	return;
	//	xSamiTumor();
	//	return;
	srand(time(0));
	//	LoadFeatures("bigbaba-A1-8-40.dat");
	//	for (int paci = 0; paci < MAX_PATID; ++paci)
	//		OcvOneStage(ALG_RF, paci, 5000, 12, 21);

	HistNormLinear tr(2, 8, 0.38f);
	tr.run();
	//	brats1();
	return;


	//for (int p = 38; p < 46; ++p)
	{
		sprintf(fname, "bigbaba-A3-8-3-30.dat");
		LoadFeatures(fname);
		for (int paci = 0; paci < MAX_PATID; ++paci)
			OcvOneStage(ALG_RF, paci, 20000, 26, 21);
		ReleaseBuffers();
	}
	/*	for (int p = 1; p < 6; ++p)
			for (int q = 4; q < 12; ++q)
			{
				sprintf(fname, "bigbaba-A3-8-%d-%d0.dat", q, p);
				LoadFeatures(fname);
				for (int paci = 0; paci < MAX_PATID; ++paci)
					OcvOneStage(ALG_RF, paci, 5000, 12, 21);
				ReleaseBuffers();
			}*/
			//	for (int p = 30; p < 46; ++p)
		//		HistNormLinearTransformX(8,0.01f * p);

		//	for (int p = 1; p < 6; ++p)
		//		for (int q = 4; q < 12; ++q)
		//			HistNormRokaTransformX(8, 0.01f * p, q);

		//	HistNormRokaTransformX(8, 0.02f, 7);

				//	Watershed();
		//	xSamiBela();
		//	ReleaseBuffers();
	getch();
}
