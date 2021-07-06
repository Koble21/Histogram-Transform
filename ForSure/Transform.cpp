#include "Transform.h"

Tester::Tester(char* fname, int _testID, int _whichData = INFANT_DATA, int _atlas = 1)
{
	VOLTYPE = _whichData;
	if (VOLTYPE != INFANT_DATA && VOLTYPE != TUMOR_DATA)
		VOLTYPE = INFANT_DATA;

	ALG = 3;
	testID = _testID;

	if (VOLTYPE == INFANT_DATA)
	{
		MAX_PATID = 10;
		MAX_SLICES = 256;
		WIDTH = 144;
		HEIGHT = 192;
		MAX_PIXELS = 8000000;
		OBSERVED_CHANNELS = 2;
		MAX_INTENSITY = 1000;
		NR_FEATURES = (_atlas > 0 ? 21 : 18);
		NR_TISSUES = 3;

		SLICES_PER_ROW = 16;
		SLICES_PER_COL = 7;
	}
	else
	{
		MAX_PATID = 50;
		MAX_SLICES = 155;
		WIDTH = 160;
		HEIGHT = 192;
		MAX_PIXELS = 0x4400000;
		OBSERVED_CHANNELS = 4;
		MAX_INTENSITY = 0x8000;
		NR_FEATURES = 36;
		NR_TISSUES = 2;

		SLICES_PER_ROW = 16;
		SLICES_PER_COL = 9;
	}

	BATCH = 10;
	VOLSIZE = WIDTH * HEIGHT * MAX_SLICES;

	FILE* F = fopen(fname, "rb");
	int head[4];
	fread(head, sizeof(int), 4, F);
	BITS = head[2];
	patLimits = (PatientBasicData*)malloc(sizeof(PatientBasicData) * MAX_PATID);
	fread(patLimits, sizeof(PatientBasicData), MAX_PATID, F);
	GTBuffer = (uchar*)malloc(sizeof(uchar) * MAX_PIXELS);
	fread(GTBuffer, sizeof(uchar), head[0], F);
	//	int db[2] = { 0 };
	//	for (int i = 0; i < head[0]; ++i)
	//		++db[GTBuffer[i]>0];
	//	printf("%d,%d\n", db[0], db[1]);
	PosBuffer = (uchar**)malloc(sizeof(uchar*) * NR_COORDS);
	for (int i = 0; i < NR_COORDS; ++i)
	{
		PosBuffer[i] = (uchar*)malloc(sizeof(uchar) * MAX_PIXELS);
		fread(PosBuffer[i], sizeof(uchar), head[0], F);
	}
	if (BITS > 8)
	{
		BigFootBuffer = (unsigned short**)malloc(sizeof(unsigned short*) * NR_FEATURES);
		for (int i = 0; i < NR_FEATURES; ++i)
		{
			BigFootBuffer[i] = (unsigned short*)malloc(sizeof(unsigned short) * MAX_PIXELS);
			if (i < OBSERVED_CHANNELS)
				fread(BigFootBuffer[i], sizeof(unsigned short), head[0], F);
		}
	}
	else
	{
		FeatBuffer = (uchar**)malloc(sizeof(uchar*) * NR_FEATURES);
		for (int i = 0; i < NR_FEATURES; ++i)
		{
			FeatBuffer[i] = (uchar*)malloc(sizeof(uchar) * MAX_PIXELS);
			if (i < OBSERVED_CHANNELS)
				fread(FeatBuffer[i], sizeof(uchar), head[0], F);
		}
	}
	fclose(F);

	GenerateFeatures();
}

Tester::~Tester()
{
	free(patLimits);
	free(GTBuffer);
	for (int i = 0; i < NR_COORDS; ++i) free(PosBuffer[i]);
	free(PosBuffer);
	if (BITS > 8)
	{
		for (int i = 0; i < NR_FEATURES; ++i) free(BigFootBuffer[i]);
		free(BigFootBuffer);
	}
	else
	{
		for (int i = 0; i < NR_FEATURES; ++i) free(FeatBuffer[i]);
		free(FeatBuffer);
	}
}

void Tester::GenerateFeatures()
{
	FILE* F;
	char fname[100];
	printf("Starting feature generation. \n");

	float Q = 1.0;
	for (int i = 0; i < BITS; ++i) Q = 2.0f * Q;
	Q = Q - 2.0f;

	// adatok beolvasása és jellemzők számítása
	for (int patID = 0; patID < MAX_PATID; ++patID) for (int ch=0; ch<OBSERVED_CHANNELS; ++ch)
	{
		unsigned short* buffer = (unsigned short*)calloc(VOLSIZE, sizeof(unsigned short));
		for (int pix = patLimits[patID].firstIndex; pix <= patLimits[patID].lastIndex; ++pix)
		{
			int x = PosBuffer[3][pix];
			int y = PosBuffer[2][pix];
			int z = PosBuffer[1][pix];
			int addr = x + y * WIDTH + z * WIDTH * HEIGHT;
			if (BITS>8)
				buffer[addr] = BigFootBuffer[ch][pix];
			else
				buffer[addr] = FeatBuffer[ch][pix];
		}

		int count = 0;

		int firstSlice = PosBuffer[0][patLimits[patID].firstIndex];

		for (int pix = patLimits[patID].firstIndex; pix <= patLimits[patID].lastIndex; ++pix)
		{
			int x = PosBuffer[3][pix];
			int y = PosBuffer[2][pix];
			int z = PosBuffer[1][pix];
			int addr = x + y * WIDTH + z * WIDTH * HEIGHT;

			int sum = buffer[addr];
			int db = 1;

			for (int s = 1; s <= 5; ++s)
			{
				for (int j = -s; j < s; ++j)
				{
					if (buffer[addr - s * WIDTH + j] > 0)
					{
						sum += buffer[addr - s * WIDTH + j];
						++db;
					}
					if (buffer[addr + s + j * WIDTH] > 0)
					{
						sum += buffer[addr + s + j * WIDTH];
						++db;
					}
					if (buffer[addr + s * WIDTH - j] > 0)
					{
						sum += buffer[addr + s * WIDTH - j];
						++db;
					}
					if (buffer[addr - s - j * WIDTH] > 0)
					{
						sum += buffer[addr - s - j * WIDTH];
						++db;
					}
				}
				if (BITS > 8)
					BigFootBuffer[OBSERVED_CHANNELS * s + ch][pix] = (sum + db / 2) / db;
				else
					FeatBuffer[OBSERVED_CHANNELS * s + ch][pix] = (sum + db / 2) / db;
			}

			db = 0;
			sum = 0;
			int mini = buffer[addr];
			int maxi = buffer[addr];
			for (int dz = z - 1; dz <= z + 1; ++dz) for (int dy = y - 1; dy <= y + 1; ++dy) for (int dx = x - 1; dx <= x + 1; ++dx)
			{
				int neigh = dx + dy * WIDTH + dz * WIDTH * HEIGHT;
				if (neigh >= 0 && neigh < VOLSIZE)
				{
					int value = buffer[neigh];
					if (value > 0)
					{
						sum += value;
						if (maxi < value) maxi = value;
						if (mini > value) mini = value;
						++db;
					}
				}
			}
			if (BITS > 8)
			{
				BigFootBuffer[6 * OBSERVED_CHANNELS + ch][pix] = (sum + db / 2) / db;
				BigFootBuffer[7 * OBSERVED_CHANNELS + ch][pix] = maxi;
				BigFootBuffer[8 * OBSERVED_CHANNELS + ch][pix] = mini;
			}
			else
			{
				FeatBuffer[6 * OBSERVED_CHANNELS + ch][pix] = (sum + db / 2) / db;
				FeatBuffer[7 * OBSERVED_CHANNELS + ch][pix] = maxi;
				FeatBuffer[8 * OBSERVED_CHANNELS + ch][pix] = mini;
			}
		}
		free(buffer);
	}

	if (VOLTYPE == 1)
		for (int patID = 0; patID < MAX_PATID; ++patID)
		{
			int minX = WIDTH / 2, maxX = WIDTH / 2, minY = HEIGHT / 2, maxY = HEIGHT / 2, minZ = MAX_SLICES - 1, maxZ = 0;
			for (int i = patLimits[patID].firstIndex; i <= patLimits[patID].lastIndex; ++i)
			{
				if (PosBuffer[1][i] > maxZ) maxZ = PosBuffer[1][i];
				if (PosBuffer[1][i] < minZ) minZ = PosBuffer[1][i];
				if (PosBuffer[2][i] > maxY) maxY = PosBuffer[2][i];
				if (PosBuffer[2][i] < minY) minY = PosBuffer[2][i];
				if (PosBuffer[3][i] > maxX) maxX = PosBuffer[3][i];
				if (PosBuffer[3][i] < minX) minX = PosBuffer[3][i];
			}
			for (int i = patLimits[patID].firstIndex; i <= patLimits[patID].lastIndex; ++i)
			{
				if (BITS > 8)
				{
					BigFootBuffer[18][i] = 1 + cvRound(Q * ((float)(PosBuffer[1][i] - minZ) / (float)(maxZ - minZ)));
					BigFootBuffer[19][i] = 1 + cvRound(Q * ((float)(PosBuffer[2][i] - minY) / (float)(maxY - minY)));
					BigFootBuffer[20][i] = 1 + cvRound(Q * ((float)(PosBuffer[3][i] - minX) / (float)(maxX - minX)));
				}
				else
				{
					FeatBuffer[18][i] = 1 + cvRound(Q * ((float)(PosBuffer[1][i] - minZ) / (float)(maxZ - minZ)));
					FeatBuffer[19][i] = 1 + cvRound(Q * ((float)(PosBuffer[2][i] - minY) / (float)(maxY - minY)));
					FeatBuffer[20][i] = 1 + cvRound(Q * ((float)(PosBuffer[3][i] - minX) / (float)(maxX - minX)));
				}
			}
		}

	printf("Feature generation finished. \n");
}


void Tester::CreateBuffers()
{
	if (BITS > 8)
	{
		BigFootBuffer = (unsigned short**)malloc(NR_FEATURES * sizeof(unsigned short*));
		for (int f = 0; f < NR_FEATURES; ++f)
			BigFootBuffer[f] = (unsigned short*)malloc(MAX_PIXELS * sizeof(unsigned short));
	}
	else
	{
		FeatBuffer = (uchar**)malloc(NR_FEATURES * sizeof(uchar*));
		for (int f = 0; f < NR_FEATURES; ++f)
			FeatBuffer[f] = (uchar*)malloc(MAX_PIXELS);
	}

	PosBuffer = (uchar**)malloc(NR_COORDS * sizeof(uchar*));
	for (int c = 0; c < NR_COORDS; ++c)
		PosBuffer[c] = (uchar*)malloc(MAX_PIXELS);

	GTBuffer = (uchar*)malloc(MAX_PIXELS);
}

void Tester::ReleaseBuffers()
{
	if (BITS > 8)
	{
		for (int f = 0; f < NR_FEATURES; ++f)
			free(BigFootBuffer[f]);
		free(BigFootBuffer);
	}
	else
	{
		for (int f = 0; f < NR_FEATURES; ++f)
			free(FeatBuffer[f]);
		free(FeatBuffer);
	}

	for (int c = 0; c < NR_COORDS; ++c)
		free(PosBuffer[c]);
	free(PosBuffer);

	free(GTBuffer);
}

void Tester::MultiSVM(int _testItem, int _nrTrainPixPerPatient, int _param, int RES_IMAGE)
{
	IplImage* imRe;
	const int PIXBATCH = 10000;

	int svmCount = _param;
	if (svmCount < 1) svmCount = 1;
	if (svmCount > 50) svmCount = 50;

	printf("\nTraining begins...\n");
	// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID - MAX_PATID / BATCH) * _nrTrainPixPerPatient;
	int* selForTrain = (int*)malloc(sizeof(int) * nrTrainData);

	Ptr<cv::ml::SVM> svm[50];

	Mat_<float> trainFeatures(nrTrainData, NR_FEATURES);
	Mat_<int> trainLabels(nrTrainData, 1);

	int trainLapse, testLapse;
	startTime();
	for (int svmID = 0; svmID < svmCount; ++svmID)
	{
		int trainCount = 0;

		for (int patID = 0; patID < MAX_PATID; ++patID)
			if (_testItem != (patID % BATCH))
			{
				int count = 0;
				int index = 0;
				while (count < _nrTrainPixPerPatient)
				{
					index = (index + rand() * rand() + rand()) % patLimits[patID].pixelCount;
					selForTrain[trainCount] = patLimits[patID].firstIndex + index;
					++count;
					++trainCount;
				}
			}


		for (int o = 0; o < nrTrainData; ++o)
		{
			for (int f = 0; f < NR_FEATURES; ++f)
			{
				float value = (float)(BITS > 8 ? BigFootBuffer[f][selForTrain[o]] : FeatBuffer[f][selForTrain[o]]);
				trainFeatures(o, f) = value;
			}
			int label = (int)(GTBuffer[selForTrain[o]]);
			trainLabels(o, 0) = (VOLTYPE == 1 ? label : (label > 1));
		}

		svm[svmID] = cv::ml::SVM::create();

		svm[svmID]->setType(cv::ml::SVM::C_SVC);
		svm[svmID]->setC(1);
		svm[svmID]->setKernel(cv::ml::SVM::INTER);
		svm[svmID]->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 200, 1e-6));
		svm[svmID]->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
	}
	trainLapse = endTime();

	printf("Training duration: %.3f seconds.\n", (float)trainLapse * 0.001);
	// confusion matrix declaration

	ConfusionMatrix overall(NR_TISSUES);
	ConfusionMatrix kungfu(NR_TISSUES);
		// testing the pixels of testBaby
	printf("\nTesting begins...\n");
	if (VOLTYPE == 2)
		imRe = cvCreateImage(cvSize(WIDTH * SLICES_PER_ROW, HEIGHT * SLICES_PER_COL), IPL_DEPTH_8U, 3);
	else
		imRe = cvCreateImage(cvSize(WIDTH * SLICES_PER_ROW, HEIGHT * SLICES_PER_COL), IPL_DEPTH_8U, 1);

	for (int testPaci = _testItem; testPaci < MAX_PATID; testPaci += BATCH)
	{
		int** votes = (int**)malloc(NR_TISSUES * sizeof(int*));
		for (int i = 0; i < NR_TISSUES; ++i)
			votes[i] = (int*)calloc((patLimits[testPaci].pixelCount), sizeof(int));

		cvSet(imRe, cvScalar(0,0,0,0));
		kungfu.reset();
		startTime();

		int pixelsToTest = patLimits[testPaci].pixelCount;
		int nextTestPixel = patLimits[testPaci].firstIndex;
		int firstTestPixel = patLimits[testPaci].firstIndex;

		while (pixelsToTest > 0)
		{
			int thisBatch = pixelsToTest;
			if (thisBatch > PIXBATCH) thisBatch = PIXBATCH;
			Mat_<float> testFeatures(thisBatch, NR_FEATURES);
			for (int pix = nextTestPixel; pix < nextTestPixel + thisBatch; ++pix)
			{
				for (int f = 0; f < NR_FEATURES; ++f)
				{
					float value = (float)(BITS > 8 ? BigFootBuffer[f][pix] : FeatBuffer[f][pix]);
					//					float value = (float)(FeatBuffer[f][pix]);
					testFeatures(pix - nextTestPixel, f) = value;
				}
			}

			for (int svmID = 0; svmID < svmCount; ++svmID)
			{
				Mat response;
				svm[svmID]->predict(testFeatures, response);
				for (int pix = nextTestPixel; pix < nextTestPixel + thisBatch; ++pix)
				{
					int deci = response.at<float>(pix - nextTestPixel, 0);
					votes[deci][pix - firstTestPixel]++;
				}
			}
			printf(".");
			pixelsToTest -= thisBatch;
			nextTestPixel += thisBatch;
		}

		pixelsToTest = patLimits[testPaci].pixelCount;

		for (int pix = firstTestPixel; pix < firstTestPixel + pixelsToTest; ++pix)
		{
			int deci = 1;
			int addr = pix - firstTestPixel;
			if (NR_TISSUES > 2)
			{
				if (votes[2][addr] > votes[deci][addr]) deci = 2;
			}
			if (votes[0][addr] > votes[deci][addr]) deci = 0;

			int label = (int)(GTBuffer[pix]);
			int gt = (VOLTYPE == 1 ? label : (label > 1));
			int z = PosBuffer[1][pix];
			int y = PosBuffer[2][pix];
			int x = PosBuffer[3][pix];
			int X = x + (z % SLICES_PER_ROW) * WIDTH;
			int Y = y + (z / SLICES_PER_ROW) * HEIGHT;

			if ((VOLTYPE == 1))
			{
				setGray(imRe, X, Y, 100 * deci + 50);
			}
			if ((VOLTYPE == 2))
			{
				if (deci > 0)
					if (gt > 0)
						setColor(imRe, X, Y, 0, 255, 0);
					else
						setColor(imRe, X, Y, 0, 0, 255);
				else
					if (gt > 0)
						setColor(imRe, X, Y, 255, 0, 0);
					else
						setColor(imRe, X, Y, 160, 160, 160);
			}
			kungfu.addItem(gt, deci);
		}

		testLapse = endTime();
		printf("\nTesting duration: %.3f seconds.\n", (float)testLapse * 0.001);
		char* strRes = kungfu.compute();

		if ((VOLTYPE == 1))
		{
			char fname[100];
			char stamp[20];
			sprintf(stamp, "%s", timestamp());
			sprintf(fname, "segm1/%s.png", stamp);
			cvSaveImage(fname, imRe);
			FILE* G = fopen("segm1/eval.csv", "at");
			fprintf(G, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n", stamp, ALG_SVM, testPaci, BITS, _nrTrainPixPerPatient, _param, 0, NR_FEATURES, trainLapse, testLapse, strRes);
			fclose(G);
		}
		if ((VOLTYPE == 2))
		{
			char fname[100];
			char stamp[20];
			sprintf(stamp, "%s", timestamp());
			sprintf(fname, "segm2/%s.png", stamp);
			cvSaveImage(fname, imRe);
			FILE* G = fopen("segm2/eval.csv", "at");
			fprintf(G, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n", stamp, ALG_SVM, testPaci, BITS, _nrTrainPixPerPatient, _param, 0, NR_FEATURES, trainLapse, testLapse, strRes);
			fclose(G);
		}

		for (int i = 0; i < NR_TISSUES; ++i)
			free(votes[i]);
		free(votes);
	}
}


void Tester::TrainAndTest(int _classifier, int _testItem, int _nrTrainPixPerPatient, int _param, int _param2)
{
	IplImage* imRe;
	const int PIXBATCH = 10000;

	int depilimit = _param;
	if (depilimit < 5) depilimit = 5;
	if (depilimit > 50) depilimit = 50;
	int nrTrees = _param2;
	if (nrTrees < 5) nrTrees = 5;
	if (nrTrees > 255) nrTrees = 255;

	int kknn = _param;
	if (kknn < 1) kknn = 1;
	if (kknn > 25) kknn = 25;

	printf("\nTraining begins...\n");
	// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID - MAX_PATID/BATCH) * _nrTrainPixPerPatient;
	int* selForTrain = (int*)malloc(sizeof(int) * nrTrainData);
	int trainCount = 0;

	for (int patID = 0; patID < MAX_PATID; ++patID)
		if (_testItem != (patID%BATCH))
		{
			int count = 0;
			int index = 0;
			while (count < _nrTrainPixPerPatient)
			{
				index = (index + rand() * rand() + rand()) % patLimits[patID].pixelCount;
				selForTrain[trainCount] = patLimits[patID].firstIndex + index;
				++count;
				++trainCount;
			}
		}

	Mat_<float> trainFeatures(nrTrainData, NR_FEATURES);
	Mat_<int> trainLabels(nrTrainData, 1);

	for (int o = 0; o < nrTrainData; ++o)
	{
		for (int f = 0; f < NR_FEATURES; ++f)
		{
			float value = (float)(BITS>8 ? BigFootBuffer[f][selForTrain[o]] : FeatBuffer[f][selForTrain[o]]);
			trainFeatures(o, f) = value;
		}
		int label = (int)(GTBuffer[selForTrain[o]]);
		trainLabels(o, 0) = (VOLTYPE==1 ? label : (label>1));
	}

	int trainLapse, testLapse;
	Ptr<ml::RTrees> rand_trees = ml::RTrees::create();
	Ptr<KNearest> knn(KNearest::create());
	Ptr<cv::ml::SVM>  svm = cv::ml::SVM::create();

	switch (_classifier)
	{
	case ALG_KNN:
		startTime();
		knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		trainLapse = endTime();
		break;
	case ALG_RF:
		//params
		rand_trees->setMaxDepth(depilimit);
		rand_trees->setMinSampleCount(0);
		rand_trees->setRegressionAccuracy(0.0f);
		rand_trees->setUseSurrogates(false);
		rand_trees->setMaxCategories(NR_TISSUES);
		rand_trees->setCVFolds(1);
		rand_trees->setUse1SERule(false);
		rand_trees->setTruncatePrunedTree(false);
		//	rand_trees->setPriors(Mat());
		rand_trees->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, nrTrees, 0.01f));
		startTime();
		rand_trees->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		trainLapse = endTime();
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

	ConfusionMatrix kungfu(NR_TISSUES), overall(NR_TISSUES);
	if (VOLTYPE==2)
		imRe = cvCreateImage(cvSize(WIDTH * SLICES_PER_ROW, HEIGHT * SLICES_PER_COL), IPL_DEPTH_8U, 3);
	else
		imRe = cvCreateImage(cvSize(WIDTH * SLICES_PER_ROW, HEIGHT * SLICES_PER_COL), IPL_DEPTH_8U, 1);

	printf("\nTesting begins...\n");
	for (int testPaci = _testItem; testPaci < MAX_PATID; testPaci += BATCH)
	{
		cvSet(imRe, cvScalar(0, 0, 0, 0));
		kungfu.reset();
		startTime();
		int pixelsToTest = patLimits[testPaci].pixelCount;
		int nextTestPixel = patLimits[testPaci].firstIndex;

		while (pixelsToTest > 0)
		{
			int thisBatch = pixelsToTest;
			if (thisBatch > PIXBATCH) thisBatch = PIXBATCH;
			Mat_<float> testFeatures(thisBatch, NR_FEATURES);
			for (int pix = nextTestPixel; pix < nextTestPixel + thisBatch; ++pix)
			{
				for (int f = 0; f < NR_FEATURES; ++f)
				{
					float value = (float)(BITS > 8 ? BigFootBuffer[f][pix] : FeatBuffer[f][pix]);
//					float value = (float)(FeatBuffer[f][pix]);
					testFeatures(pix - nextTestPixel, f) = value;
				}
			}
			Mat response;

			if (_classifier == ALG_RF)
				rand_trees->predict(testFeatures, response);
			else if (_classifier == ALG_KNN)
				knn->findNearest(testFeatures, kknn, response);
			else if (_classifier == ALG_SVM)
				svm->predict(testFeatures, response);

			for (int pix = nextTestPixel; pix < nextTestPixel + thisBatch; ++pix)
			{
				int deci = response.at<float>(pix - nextTestPixel, 0);
				int label = (int)(GTBuffer[pix]);
				int gt = (VOLTYPE == 1 ? label : (label > 1));
				int z = PosBuffer[1][pix];
				int y = PosBuffer[2][pix];
				int x = PosBuffer[3][pix];
				int X = x + (z % SLICES_PER_ROW) * WIDTH;
				int Y = y + (z / SLICES_PER_ROW) * HEIGHT;

				if ((VOLTYPE == 1))
				{
					setGray(imRe, X, Y, 100 * deci + 50);
				}
				if ((VOLTYPE == 2))
				{
					if (deci > 0)
						if (gt > 0)
							setColor(imRe, X, Y, 0, 255, 0);
						else
							setColor(imRe, X, Y, 0, 0, 255);
					else
						if (gt > 0)
							setColor(imRe, X, Y, 255, 0, 0);
						else
							setColor(imRe, X, Y, 160, 160, 160);
				}
				kungfu.addItem(gt, deci);
			}
			printf(".");
			pixelsToTest -= thisBatch;
			nextTestPixel += thisBatch;
		}
		testLapse = endTime();
		printf("\nTesting duration: %.3f seconds.\n", (float)testLapse * 0.001);
		char* strRes = kungfu.compute();

		if ((VOLTYPE == 1))
		{
			char fname[100];
			char stamp[20];
			sprintf(stamp, "%s", timestamp());
			sprintf(fname, "segm1/%s.png", stamp);
			cvSaveImage(fname, imRe);
			FILE* G = fopen("segm1/eval.csv", "at");
			fprintf(G, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n", stamp, _classifier, testPaci, BITS, _nrTrainPixPerPatient, _param, _param2, NR_FEATURES, trainLapse, testLapse, strRes);
			fclose(G);
		}
		if ((VOLTYPE == 2))
		{
			char fname[100];
			char stamp[20];
			sprintf(stamp, "%s", timestamp());
			sprintf(fname, "segm2/%s.png", stamp);
			cvSaveImage(fname, imRe);
			FILE* G = fopen("segm2/eval.csv", "at");
			fprintf(G, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n", stamp, _classifier, testPaci, BITS, _nrTrainPixPerPatient, _param, _param2, NR_FEATURES, trainLapse, testLapse, strRes);
			fclose(G);
		}

		free(strRes);
	}
	cvReleaseImage(&imRe);
}

char* Tester::timestamp() {
	time_t ltime = time(NULL);
	struct tm* mytime = localtime(&ltime);
	char timestring[20];
	sprintf(timestring, "%04d%02d%02d%02d%02d%02d%03d", mytime->tm_year + 1900, mytime->tm_mon + 1,
		mytime->tm_mday, mytime->tm_hour, mytime->tm_min, mytime->tm_sec, rand() % 1000);
	printf("%s\n", timestring);
	return timestring;
}
