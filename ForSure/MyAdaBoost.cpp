#include "MyAdaBoost.h"

MyAdaBoost::MyAdaBoost(char* dataSource)
{
	CreateBuffers();
	LoadFeatures(dataSource);
	nextTestID = 1000000;
	FILE* F = fopen("arnold-results/next.idx", "rt");
	if (F)
	{
		char text[100];
		fgets(text, 99, F);
		int value;
		sscanf(text, "%d", &value);
		fclose(F);
		if (value >= 1000000) nextTestID = value;
	}
	F = fopen("arnold-results/next.idx", "wt");
	fprintf(F,"%d",nextTestID+1);
	fclose(F);
	evalFileName = strdup("arnold-results/evaluation.csv");
}

MyAdaBoost::~MyAdaBoost()
{
	ReleaseBuffers();
}

void MyAdaBoost::startTime()
{
	QueryPerformanceFrequency(&procFreq);
	QueryPerformanceCounter(&timeStart);
}

int MyAdaBoost::endTime()
{
	QueryPerformanceCounter(&timeEnd);
	return (int)(1000.0 * (double(timeEnd.QuadPart - timeStart.QuadPart) / double(procFreq.QuadPart)));
}

void MyAdaBoost::setGray(IplImage* im, int x, int y, uchar v)
{
	im->imageData[im->widthStep * y + x] = v;
}

void MyAdaBoost::setColor(IplImage* im, int x, int y, uchar r, uchar g, uchar b)
{
	im->imageData[im->widthStep * y + 3*x] = b;
	im->imageData[im->widthStep * y + 3 * x + 1] = g;
	im->imageData[im->widthStep * y + 3 * x + 2] = r;
}
void MyAdaBoost::getColor(IplImage* im, int x, int y, int& r, int& g, int& b)
{
	b = (uchar)(im->imageData[im->widthStep * y + 3*x]);
	g = (uchar)(im->imageData[im->widthStep * y + 3 * x + 1]);
	r = (uchar)(im->imageData[im->widthStep * y + 3 * x + 2]);
}


void MyAdaBoost::paintPixel(int x, int y, int z, uchar v)
{
	int X = (z % 16) * WIDTH + x;
	int Y = (z / 16) * HEIGHT + y;
	setGray(imRe, X, Y, v);
}

uchar MyAdaBoost::getGray(IplImage* im, int x, int y)
{
	return (uchar)(im->imageData[im->widthStep * y + x]);
}

void MyAdaBoost::plane2space(int X, int Y, int& x, int& y, int& z)
{
	x = X % WIDTH;
	y = Y % HEIGHT;
	z = (X / WIDTH) + (Y / HEIGHT) * SLICES_PER_ROW;
}

void MyAdaBoost::space2plane(int x, int y, int z, int& X, int& Y)
{
	X = x + (z % SLICES_PER_ROW) * WIDTH;
	Y = y + (z / SLICES_PER_ROW) * HEIGHT;
}

void MyAdaBoost::CreateBuffers()
{
	FeatBuffer = (uchar**)malloc(NR_FEATURES * sizeof(uchar*));
	for (int f = 0; f < NR_FEATURES; ++f)
		FeatBuffer[f] = (uchar*)malloc(MAX_PIXELS);

	PosBuffer = (uchar**)malloc(NR_COORDS * sizeof(uchar*));
	for (int c = 0; c < NR_COORDS; ++c)
		PosBuffer[c] = (uchar*)malloc(MAX_PIXELS);

	GTBuffer = (uchar*)malloc(MAX_PIXELS);
}

void MyAdaBoost::ReleaseBuffers()
{
	for (int f = 0; f < NR_FEATURES; ++f)
		free(FeatBuffer[f]);
	free(FeatBuffer);

	for (int c = 0; c < NR_COORDS; ++c)
		free(PosBuffer[c]);
	free(PosBuffer);

	free(GTBuffer);
}


void MyAdaBoost::LoadFeatures(char* fname)
{
	int head[3];
	FILE* F = fopen(fname, "rb");
	fread(head, sizeof(int), 3, F);
	fread(patLimits, sizeof(PatientBasicData), MAX_PATID, F);
	fread(GTBuffer, sizeof(uchar), head[0], F);
	for (int c = 0; c < NR_COORDS; ++c)
		fread(PosBuffer[c], sizeof(uchar), head[0], F);
	for (int f = 0; f < NR_FEATURES; ++f)
		fread(FeatBuffer[f], sizeof(uchar), head[0], F);
	fclose(F);
	printf("Successfully loaded %d pixel data.\n", head[0]);
	FILE* G = fopen("kingkong.csv", "at");
	fprintf(G, "%s\n", fname);
	fclose(G);
}


void MyAdaBoost::AdaBoost(int stage, uchar* parti)
{
	char fname[100];
	int nrTrainData = trainSize * (MAX_PATID - 1);
	// innen kezdodo resz fog elkoltozni
	int* selForTrain = (int*)malloc(sizeof(int) * nrTrainData);
	int trainCount = 0;

	int lapseTrain;
	Ptr<ml::Boost> ada = ml::Boost::create();
	if (trainOnlyIfNecessary)
	{ 
		sprintf(fname, "ada%d-p%d-%dk-%d.xml", stage, patID, trainSize / 1000, depthLimit);
		FILE* F = fopen(fname, "rb");
		if (F)
		{
			fclose(F);
			startTime();
			ada = StatModel::load<ml::Boost>(fname);
			lapseTrain = endTime();

			printf("loaded ADA: %d\n", ada->isTrained());
		}
		else
			printf("ADA file (%s) does not exist...\n", fname);
		//(fname);
	}
	if (ada->isTrained() == 0)
	{
		for (int paci = 0; paci < MAX_PATID; ++paci)
			if (patID != paci)
			{
				int count = 0;
				int index = 0;
				while (count < trainSize)
				{
					index = (index + rand() * rand() + rand()) % patLimits[paci].pixelCount;
					if (GTBuffer[patLimits[paci].firstIndex + index] > 0 || stage == 1)
					{
						selForTrain[trainCount] = patLimits[paci].firstIndex + index;
						++count;
						++trainCount;
					}
				}
			}

		Mat_<float> trainFeatures(nrTrainData, NR_FEATURES);
		Mat_<int> trainLabels(nrTrainData, 1);

		for (int o = 0; o < nrTrainData; ++o)
		{
			for (int f = 0; f < NR_FEATURES; ++f)
			{
				float value = (float)(FeatBuffer[f][selForTrain[o]]);
				trainFeatures(o, f) = value;
			}
			//		trainLabels(o,0) = (int)(GTBuffer[selForTrain[o]]);
			trainLabels(o, 0) = (GTBuffer[selForTrain[o]] > (stage - 1) ? 1 : 0);
		}

		ada->setMaxDepth(depthLimit);
		ada->setBoostType(ml::Boost::DISCRETE);
		ada->setWeakCount(100);
		startTime();
		ada->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
		lapseTrain = endTime();
		sprintf(fname, "ada%d-p%d-%dk-%d.xml", stage, patID, trainSize /1000, depthLimit);
		ada->save(fname);
		printf("Training duration so far: %.3f seconds.\n", (float)lapseTrain * 0.001);
	}

	cm_reset();

	printf("\nTesting begins...\n");
	int pixelsToTest = patLimits[patID].pixelCount;
	int firstTestPixel = patLimits[patID].firstIndex;
	int nextTestPixel = patLimits[patID].firstIndex;

	startTime();
	while (pixelsToTest > 0)
	{
		int thisBatch = pixelsToTest;
		if (thisBatch > BATCH) thisBatch = BATCH;
		Mat_<float> testFeatures(thisBatch, NR_FEATURES);
		for (int pix = nextTestPixel; pix < nextTestPixel + thisBatch; ++pix)
		{
			for (int f = 0; f < NR_FEATURES; ++f)
			{
				float value = (float)(FeatBuffer[f][pix]);
				testFeatures(pix - nextTestPixel, f) = value;
			}
		}
		Mat response;
		ada->predict(testFeatures, response);

		int deci, gt;
		for (int pix = nextTestPixel; pix < nextTestPixel + thisBatch; ++pix)
		{
			gt = GTBuffer[pix];
			if (stage == 1 && gt > 0) gt = 1;
			if (stage == 2) deci = 0;
			if (parti[pix - firstTestPixel] >= (stage - 1))
			{
				deci = response.at<float>(pix - nextTestPixel, 0);
				if (deci > 0) deci = stage; else deci = stage - 1;
				parti[pix - firstTestPixel] = deci;
				//if (gt > 0) gt = 1;
				if ((stage == 1 && deci == 0) || (stage == 2 && deci > 0))
					paintPixel(PosBuffer[3][pix], PosBuffer[2][pix], PosBuffer[1][pix], 50 + 100 * deci);
			}
			cm_addItem(gt, deci);
		}
		printf(".");
		pixelsToTest -= thisBatch;
		nextTestPixel += thisBatch;
	}
	int lapseTest = endTime();
	printf("Testing duration in stage %d: %.3f seconds.\n", stage, (float)lapseTest * 0.001);

	cm_compute(evalFile);
	// itt van vege az elkoltoztetendo resznek
}

int MyAdaBoost::getNeighborCount(int X, int Y, int th)
{
	int whites = 0;
	for (int nX = X - 1; nX <= X + 1; ++nX) for (int nY = Y - 1; nY <= Y + 1; ++nY)
		if (nX != X || nY != Y)
			if (getGray(imRe, nX, nY) == th)
				++whites;
	return whites;
}

void MyAdaBoost::neighborCheck(bool whichCase, int color)
{
	char fname[100];
	cm_reset();
	IplImage* imRe2 = cvCloneImage(imRe);
	for (int index = patLimits[patID].firstIndex; index <= patLimits[patID].lastIndex; ++index)
	{
		int X, Y;
		space2plane(PosBuffer[3][index], PosBuffer[2][index], PosBuffer[1][index], X, Y);
		if (whichCase)
		{
			if (getGray(imRe, X, Y) == color)
				if (getNeighborCount(X, Y, color) < 5)
					setGray(imRe2, X, Y, GREY_GM);
		}
		else
		{
			if (getGray(imRe, X, Y) == GREY_GM)
				if (getNeighborCount(X, Y, color) >= 5)
					setGray(imRe2, X, Y, color);
		}

		cm_addItem(GTBuffer[index], getGray(imRe2, X, Y) / 100);
	}
	cm_compute(evalFile);
	sprintf(fname, "arnold-results/%d-2.png", nextTestID);
	cvSaveImage(fname, imRe2);
	cvCopy(imRe2, imRe);
	cvReleaseImage(&imRe2);
}

void MyAdaBoost::PostProcessing()
{
	int initHist[NR_TISSUES] = { 0 };
	for (int pix = 0; pix <= patLimits[MAX_PATID - 1].lastIndex; ++pix)
	{
		if (PosBuffer[0][pix] == patID)
			initHist[GTBuffer[pix]]++;
	}
	int deciHist[NR_TISSUES] = { 0 };
	for (int X = 0; X < imRe->width; X++) for (int Y = 0; Y < imRe->height; Y++)
	{
		int v = getGray(imRe, X, Y);
		if (v % 100 == 50)
			deciHist[v / 100]++;
	}
	bool whichCase = ((float)deciHist[2] / (float)deciHist[1] > (float)initHist[2] / (float)initHist[1]);
	bool whichCase2 = ((float)deciHist[0] / (float)deciHist[1] > (float)initHist[0] / (float)initHist[1]);

	neighborCheck(whichCase, GREY_WM);
	neighborCheck(whichCase2, GREY_CSF);
}


void MyAdaBoost::run(int testBaby, int nrTrainPixPerTrainBaby, int maxDepth, int forceNewTraining)
{
	patID = 7;
	if (testBaby >= 0 && testBaby < MAX_PATID)
		patID = testBaby;

	depthLimit = maxDepth;
	if (depthLimit < 5) depthLimit = 5;
	if (depthLimit > 50) depthLimit = 50;

	trainSize = nrTrainPixPerTrainBaby;
	trainOnlyIfNecessary = 0;// !forceNewTraining;

	printf("\nTraining begins...\n");
	// select train pixels, store their indexes in selForTrain
	int nrTrainData = (MAX_PATID - 1) * trainSize;

	evalFile = fopen(evalFileName, "at");
	fprintf(evalFile, "%d,%d,%d,%d,", nextTestID, patID, trainSize, depthLimit);

	uchar* parti = (uchar*)calloc(patLimits[patID].pixelCount, 1);
	imRe = cvCreateImage(cvSize(SLICES_PER_ROW * WIDTH, SLICES_PER_COL * HEIGHT), IPL_DEPTH_8U, 1);
	cvSet(imRe, cvScalar(0));
	// inicializalas
	AdaBoost(STAGE_ONE, parti);

	AdaBoost(STAGE_TWO, parti);

	PostProcessing();

	char fname[100];
	sprintf(fname, "arnold-results/%d-1.png", nextTestID);
	cvSaveImage(fname, imRe);
	cvReleaseImage(&imRe);

	imRe = cvLoadImage(fname, 1);
	for (int i = patLimits[patID].firstIndex; i <= patLimits[patID].lastIndex; ++i)
	{
		int X, Y, r, g, b;
		space2plane(PosBuffer[3][i], PosBuffer[2][i], PosBuffer[1][i], X, Y);
		getColor(imRe, X, Y, r, g, b);
		int deci = r / 100;
		int gt = GTBuffer[i];
		if (deci == gt) setColor(imRe,X,Y,0,150+50*deci,0);
		else setColor(imRe, X, Y, 255, 0, 0);
	}

	sprintf(fname, "arnold-results/%d-c.png", nextTestID);
	cvSaveImage(fname, imRe);
	cvReleaseImage(&imRe);
	free(parti);
	fprintf(evalFile, "\n");
	fclose(evalFile);
}


