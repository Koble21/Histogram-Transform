#include "Bratsform.h"

HistNormLinear::HistNormLinear(int _qBitDepth, float _l25)
{
	lambda25 = _l25;
	if (lambda25 < 0.25f) lambda25 = 0.25;
	if (lambda25 > 0.45f) lambda25 = 0.45;
	lambda75 = 1.0f - lambda25;

	qBitDepth = _qBitDepth;
	if (qBitDepth < 4) qBitDepth = 4;
	if (qBitDepth > 10) qBitDepth = 10;

	p_lo = 0.01f;
	nrMilestones = 5;
	ALG = 1;
}

HistNormLinear::HistNormLinear()
{
	lambda25 = 0.4f;
	lambda75 = 1.0f - lambda25;
	qBitDepth = 8;
	p_lo = 0.01f;
	nrMilestones = 5;
	ALG = 1;
}


void HistNormLinear::run()
{

	char fname[100];
	unsigned short* dataBuffers[OBSERVED_CHANNELS];
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		dataBuffers[ch] = (unsigned short*)malloc(VOLSIZE * sizeof(unsigned short));
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);
	unsigned short* flat = (unsigned short*)malloc(MILLIO * sizeof(unsigned short));

	for (int patID = 0; patID < MAX_PATID; ++patID)
	{
		sprintf(fname, "babainput/baba-%d.bab", patID);
		FILE* F = fopen(fname, "rb");
		int nrBytes = 0;
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
			nrBytes += fread(dataBuffers[ch], sizeof(unsigned short), VOLSIZE, F);
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

		int index, p25, p75;
		float a, b;
		float coltrans[OBSERVED_CHANNELS][MAX_INTENSITY] = { 0 };

		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		{
			index = 0;
			for (int value = 0; value < MAX_INTENSITY; ++value)
			{
				for (int i = 0; i < hist[ch][value]; ++i)
					flat[index++] = value + 1;
			}

			p25 = flat[pixCount / 4];
			p75 = flat[pixCount * 3 / 4];

			a = (lambda75 - lambda25) / ((float)(p75 - p25));
			b = lambda25 - a * (float)p25;

			for (int value = 0; value < MAX_INTENSITY; ++value)
			{
				float newValue = a * (value + 1) + b;
				if (newValue < 0.0f) newValue = 0.0f;
				if (newValue > 1.0f) newValue = 1.0f;
				coltrans[ch][value] = newValue;
			}
		}

		float Q = 1.0f;
		for (int i = 0; i < qBitDepth; ++i) Q = 2.0f * Q;
		for (int pix = 0; pix < VOLSIZE; ++pix)
			if (bufferGT[pix] > 0)
				for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
				{
					int value = dataBuffers[ch][pix];
					int newValue = 1 + cvRound((Q - 2.0) * coltrans[ch][value]);
					dataBuffers[ch][pix] = newValue;
				}

		sprintf(fname, "babainput/baba-A1-%d-%d-%d.bab", patID, qBitDepth, cvRound(100 * lambda25));
		F = fopen(fname, "wb");
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
			fwrite(dataBuffers[ch], 2, VOLSIZE, F);
		fwrite(bufferGT, 1, VOLSIZE, F);
		fclose(F);
	}
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);
	free(flat);

	GenerateFeatures();
}

void HistNormLinear::GenerateFeatures()
{
	FILE* F;
	char fname[100];
	printf("Starting feature generation. \n");


	unsigned short* dataBuffers[OBSERVED_CHANNELS];
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		dataBuffers[ch] = (unsigned short*)malloc(VOLSIZE * sizeof(unsigned short));
	unsigned char* bufferGT = (unsigned char*)malloc(VOLSIZE);

	CreateBuffers();
	int totalPixels = 0;

	// adatok beolvasása és jellemzők számítása
	for (int patID = 0; patID < MAX_PATID; ++patID)
	{
		if (ALG == 1)
			sprintf(fname, "babainput/baba-A1-%d-%d-%d.bab", patID, qBitDepth, cvRound(100 * lambda25));
		else
			sprintf(fname, "babainput/baba-A%d-%d-%d-%d-%d.bab", ALG, patID, qBitDepth, nrMilestones, cvRound(1000 * p_lo));

		F = fopen(fname, "rb");
		for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
			fread(dataBuffers[ch], sizeof(unsigned short), VOLSIZE, F);
		fread(bufferGT, 1, VOLSIZE, F);
		fclose(F);
		int count = 0;
		for (int i = 0; i < VOLSIZE; ++i)
		{
			if (bufferGT[i] > 0)
			{
				int index = totalPixels + count;
				PosBuffer[0][index] = patID;
				PosBuffer[1][index] = i / (WIDTH * HEIGHT) - FIRSTSLICES[patID];
				PosBuffer[2][index] = (i % (WIDTH * HEIGHT)) / WIDTH;
				PosBuffer[3][index] = i % WIDTH;
				GTBuffer[index] = bufferGT[i] / 100; //(bufferGT[i]==10 ? 50 : bufferGT[i]);
				if (qBitDepth > 8)
				{
					for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
					{
						BigFootBuffer[ch][index] = dataBuffers[ch][i];
						int sum = dataBuffers[ch][i];
						int db = 1;
						for (int s = 1; s <= 5; ++s)
						{
							for (int j = -s; j < s; ++j)
							{
								if (dataBuffers[ch][i - s * WIDTH + j] > 0)
								{
									sum += dataBuffers[ch][i - s * WIDTH + j];
									++db;
								}
								if (dataBuffers[ch][i + s + j * WIDTH] > 0)
								{
									sum += dataBuffers[ch][i + s + j * WIDTH];
									++db;
								}
								if (dataBuffers[ch][i + s * WIDTH - j] > 0)
								{
									sum += dataBuffers[ch][i + s * WIDTH - j];
									++db;
								}
								if (dataBuffers[ch][i - s - j * WIDTH] > 0)
								{
									sum += dataBuffers[ch][i - s - j * WIDTH];
									++db;
								}
							}
							BigFootBuffer[2 * s + ch][index] = (sum + db / 2) / db;
						}
						db = 0;
						sum = 0;
						int mini = dataBuffers[ch][i];
						int maxi = dataBuffers[ch][i];
						for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
						{
							int value = dataBuffers[ch][i + (dz * HEIGHT + dy) * WIDTH + dx];
							if (value > 0)
							{
								sum += value;
								if (maxi < value) maxi = value;
								if (mini > value) mini = value;
								++db;
							}
						}
						BigFootBuffer[12 + ch][index] = (sum + db / 2) / db;
						BigFootBuffer[14 + ch][index] = maxi;
						BigFootBuffer[16 + ch][index] = mini;
					}
				}
				else
				{
					for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
					{
						FeatBuffer[ch][index] = dataBuffers[ch][i];
						int sum = dataBuffers[ch][i];
						int db = 1;
						for (int s = 1; s <= 5; ++s)
						{
							for (int j = -s; j < s; ++j)
							{
								if (dataBuffers[ch][i - s * WIDTH + j] > 0)
								{
									sum += dataBuffers[ch][i - s * WIDTH + j];
									++db;
								}
								if (dataBuffers[ch][i + s + j * WIDTH] > 0)
								{
									sum += dataBuffers[ch][i + s + j * WIDTH];
									++db;
								}
								if (dataBuffers[ch][i + s * WIDTH - j] > 0)
								{
									sum += dataBuffers[ch][i + s * WIDTH - j];
									++db;
								}
								if (dataBuffers[ch][i - s - j * WIDTH] > 0)
								{
									sum += dataBuffers[ch][i - s - j * WIDTH];
									++db;
								}
							}
							FeatBuffer[2 * s + ch][index] = (sum + db / 2) / db;
						}
						db = 0;
						sum = 0;
						int mini = dataBuffers[ch][i];
						int maxi = dataBuffers[ch][i];
						for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
						{
							int value = dataBuffers[ch][i + (dz * HEIGHT + dy) * WIDTH + dx];
							if (value > 0)
							{
								sum += value;
								if (maxi < value) maxi = value;
								if (mini > value) mini = value;
								++db;
							}
						}
						FeatBuffer[12 + ch][index] = (sum + db / 2) / db;
						FeatBuffer[14 + ch][index] = maxi;
						FeatBuffer[16 + ch][index] = mini;
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
	float Q = 1.0;
	for (int i = 0; i < qBitDepth; ++i) Q = 2.0f * Q;
	Q = Q - 2.0f;
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
			if (qBitDepth > 8)
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
	// Mentés
	if (ALG == 1)
		sprintf(fname, "bigbaba-A1-%d-%d.dat", qBitDepth, cvRound(100 * lambda25));
	else
		sprintf(fname, "bigbaba-A%d-%d-%d-%d.dat", ALG, qBitDepth, nrMilestones, cvRound(1000 * p_lo));
	F = fopen(fname, "wb");
	fwrite(&totalPixels, sizeof(int), 1, F);
	fwrite(&NR_FEATURES, sizeof(int), 1, F);
	fwrite(&qBitDepth, sizeof(int), 1, F);
	fwrite(patLimits, sizeof(PatientBasicData), MAX_PATID, F);
	fwrite(GTBuffer, sizeof(uchar), totalPixels, F);
	for (int c = 0; c < NR_COORDS; ++c)
	{
		fwrite(PosBuffer[c], sizeof(uchar), totalPixels, F);
	}
	for (int f = 0; f < NR_FEATURES; ++f)
		if (qBitDepth > 8)
		{
			fwrite(BigFootBuffer[f], sizeof(unsigned short), totalPixels, F);
		}
		else
		{
			fwrite(FeatBuffer[f], sizeof(uchar), totalPixels, F);
		}
	fclose(F);
	printf("Done. \n");

	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);

	ReleaseBuffers();
}


void HistNormLinear::CreateBuffers()
{
	if (qBitDepth > 8)
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

void HistNormLinear::ReleaseBuffers()
{
	if (qBitDepth > 8)
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


/*  NYUL  */


HistNormNyul::HistNormNyul() 
{
	p_lo = 0.01f;
	p_hi = 1.0f - p_lo;
	nrMilestones = 5;
	qBitDepth = 8;
	lambda25 = 0.4f;
	ALG = 2;
}

HistNormNyul::HistNormNyul(int _qBitDepth, float _pLO, int _milestones)
{
	p_lo = _pLO;
	if (p_lo < 0.01f) p_lo = 0.01f;
	if (p_lo > 0.05f) p_lo = 0.05f;
	p_hi = 1.0f - p_lo;

	nrMilestones = _milestones;
	if (nrMilestones < 3) nrMilestones = 3;
	if (nrMilestones > 11) nrMilestones = 11;

	qBitDepth = _qBitDepth;
	if (qBitDepth < 4) qBitDepth = 4;
	if (qBitDepth > 10) qBitDepth = 10;

	lambda25 = 0.4f;
	ALG = 2;
}


void HistNormNyul::run()
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
	GenerateFeatures();
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);
}



/*  ROKA  */

HistNormRoka::HistNormRoka()
//	:HistNormNyul()
{
	p_lo = 0.01f;
	p_hi = 1.0f - p_lo;
	nrMilestones = 5;
	qBitDepth = 8;
	lambda25 = 0.4f;
	ALG = 3;
}

HistNormRoka::HistNormRoka(int _qBitDepth, float _pLO, int _milestones)
//	: HistNormNyul(_qBitDepth, _pLO, _milestones)
{
	p_lo = _pLO;
	if (p_lo < 0.01f) p_lo = 0.01f;
	if (p_lo > 0.05f) p_lo = 0.05f;
	p_hi = 1.0f - p_lo;

	nrMilestones = _milestones;
	if (nrMilestones < 3) nrMilestones = 3;
	if (nrMilestones > 11) nrMilestones = 11;

	qBitDepth = _qBitDepth;
	if (qBitDepth < 4) qBitDepth = 4;
	if (qBitDepth > 10) qBitDepth = 10;

	lambda25 = 0.4f;
	ALG = 3;
}

void HistNormRoka::HistFuzzyCMeans(int* hist, int lo, int hi)
{
	int c = nrMilestones - 2;

	float v[MAX_MILESTONES];
	float u[MAX_MILESTONES];
	float sumUp[MAX_MILESTONES];
	float sumDn[MAX_MILESTONES];

	if (c == 1)
	{
		sumUp[0] = 0;
		sumDn[0] = 0;
		for (int value = lo; value <= hi; ++value)
		{
			sumUp[0] += (hist[value] * value);
			sumDn[0] += hist[value];
		}
		fuzzyClusterPrototypes[0] = sumUp[0] / sumDn[0];
	}
	else
	{
		for (int i = 0; i < c; ++i)
			v[i] = (float)lo + (float)(hi - lo) * (float)(i + 1) / (float)(c + 1);
		// fuzzy c-means: m=2.0f; -2.0f/(m-1.0f)=-2.0f
		for (int cycle = 0; cycle < 20; ++cycle)
		{
			for (int i = 0; i < c; ++i)
			{
				sumUp[i] = 0.0f;
				sumDn[i] = 0.0f;
			}
			for (int value = lo; value <= hi; ++value)
			{
				int match = -1;
				for (int i = 0; i < c; ++i)
				{
					u[i] = (v[i] - value) * (v[i] - value);
					if (u[i] < 0.00001f) match = i;
				}
				if (match >= 0) // ritka eset
				{
					for (int i = 0; i < c; ++i) u[i] = 0.0f;
					u[match] = 1.0f;
				}
				else
				{
					float sum = 0.0f;
					for (int i = 0; i < c; ++i)
					{
						u[i] = 1.0f / u[i];
						sum += u[i];
					}
					for (int i = 0; i < c; ++i)
						u[i] /= sum;
				}

				for (int i = 0; i < c; ++i)
				{
					sumUp[i] += hist[value] * u[i] * u[i] * value;
					sumDn[i] += hist[value] * u[i] * u[i];
				}
			}

			for (int i = 0; i < c; ++i)
				v[i] = sumUp[i] / sumDn[i];
		}

		for (int i = 0; i < c; ++i)
		{
			int miniAt = 0;
			for (int j = 1; j < c; ++j)
			{
				if (v[miniAt] > v[j]) miniAt = j;
			}

			fuzzyClusterPrototypes[i] = cvRound(v[miniAt]);
			v[miniAt] = MAX_INTENSITY;
		}
	}
}

void HistNormRoka::run()
{
	int fuzzyMileStones[OBSERVED_CHANNELS][MAX_PATID][MAX_MILESTONES];
	int M = nrMilestones;

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

			HistFuzzyCMeans(hist[ch], p1, p99);
			for (int i = 0; i < M - 2; ++i)
				fuzzyMileStones[ch][patID][i] = fuzzyClusterPrototypes[i];

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
	GenerateFeatures();
	for (int ch = 0; ch < OBSERVED_CHANNELS; ++ch)
		free(dataBuffers[ch]);
	free(bufferGT);
}