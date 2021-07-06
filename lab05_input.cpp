void opencvDecTree(int testBaby, int nrTrainData, int maxDepth)
{
	const int neigh = 1;
	char fname[100];
	IplImage* imT1;
	IplImage* imT2;
	IplImage* imGT;

	int trainDataStored = 0;
	uchar featureData[FEATURES];

	Mat_<float> trainFeatures(nrTrainData, FEATURES - 1);
	Mat_<int> trainLabels(nrTrainData, 1);
	Mat_<float> testFeatures(1, FEATURES - 1);

	for (int trainBaby = 1; trainBaby <= PATIENTS; ++trainBaby)
		if (trainBaby != testBaby)
		{
			sprintf(fname, "baba/baba_p%02d_ch0.png", trainBaby);
			imT1 = cvLoadImage(fname, 0);
			sprintf(fname, "baba/baba_p%02d_ch1.png", trainBaby);
			imT2 = cvLoadImage(fname, 0);
			sprintf(fname, "baba/baba_p%02d_ch2.png", trainBaby);
			imGT = cvLoadImage(fname, 0);

			int count = 0;
			while (count < nrTrainData / (PATIENTS - 1))
			{
				int x = rand() % imGT->width;
				int y = rand() % imGT->height;
				uchar g = getGray(imGT, x, y);
				//if (g == 50 && rand()<0x4000 || g == 150 && rand()<0x4000 || g == 250)
				if (g == 50 || g == 150 || g == 250)
				{
					computeFeatures(imT1, imT2, imGT, x, y, featureData);
					trainLabels(trainDataStored, 0) = (featureData[0] / 100);
					for (int o = 1; o < FEATURES; ++o)
					{
						trainFeatures(trainDataStored, o - 1) = (float)(featureData[o]);
					}

					++count;
					++trainDataStored;
				}
			}
			cvReleaseImage(&imGT);
			cvReleaseImage(&imT1);
			cvReleaseImage(&imT2);
		}

	

	Ptr<ml::DTrees> dec_trees = ml::DTrees::create();
	//params
	dec_trees->setMaxDepth(maxDepth);
	dec_trees->setMinSampleCount(0);
	dec_trees->setRegressionAccuracy(0.0f);
	dec_trees->setUseSurrogates(false);
	dec_trees->setMaxCategories(3);
	dec_trees->setCVFolds(1);
	dec_trees->setUse1SERule(false);
	dec_trees->setTruncatePrunedTree(false);
	dec_trees->setPriors(Mat());


	dec_trees->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);


	// decision trees
	int pixelCount = 0;
	sprintf(fname, "baba/baba_p%02d_ch0.png", testBaby);
	imT1 = cvLoadImage(fname, 0);
	sprintf(fname, "baba/baba_p%02d_ch1.png", testBaby);
	imT2 = cvLoadImage(fname, 0);
	sprintf(fname, "baba/baba_p%02d_ch2.png", testBaby);
	imGT = cvLoadImage(fname, 0);
	IplImage* imRE = cvCloneImage(imGT);
	cvSet(imRE, cvScalar(0));

	int CM[TISSUES][TISSUES] = { 0 };

	for (int x = 0; x < imGT->width; ++x)
	{
		printf(".");
		for (int y = 0; y < imGT->height; ++y)
		{
			uchar gt = getGray(imGT, x, y);
			if (gt == 50 || gt == 150 || gt == 250)
			{
				++pixelCount;

				computeFeatures(imT1, imT2, imGT, x, y, featureData);
				for (int o = 1; o < FEATURES; ++o)
				{
					testFeatures(0, o - 1) = (float)(featureData[o]);
				}

				Mat response;
				dec_trees->predict(testFeatures, response);
				int res = response.at<float>(0, 0);
				setGray(imRE, x, y, 100 * res + 50);
				CM[gt / 100][res]++;

			}
		}
	}

	evaluation(((int*)CM));
	sprintf(fname, "output_dtrees.png");
	cvSaveImage(fname, imRE);
}
