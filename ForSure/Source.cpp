#include "opencv2/opencv.hpp"
#include <windows.h>
#include <conio.h>

const int MAX_PATID = 10;
const int MAX_SLICES = 256;
const int WIDTH = 144;
const int HEIGHT = 192;
const int VOLSIZE = WIDTH*HEIGHT*MAX_SLICES;
const int FIRSTSLICES[10] = {85,91,91,97,92,89,89,94,98,98};

void setGray(IplImage* im, int x, int y, uchar v)
{
	im->imageData[im->widthStep * y + x] = v;
}

uchar getGray(IplImage* im, int x, int y)
{
	return (uchar)(im->imageData[im->widthStep * y + x]);
}


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

void brain()
{
//	x01();
//	x06();
//	x03();
//	x04();
	x09();
}





void main()
{
//	x01();
//	x06();
//	x03();
//	x04();
	x09();
}