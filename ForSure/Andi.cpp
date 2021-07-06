
#include "Transform.h"



void main(int argc, char* argv[])
{
	if (argc == 7)
	{
		int dataID = atoi(argv[1]);
		int voltype = atoi(argv[2]);
		int classifier = atoi(argv[3]);
		int trainDataSize = atoi(argv[4]);
		int param1 = atoi(argv[5]);
		int param2 = atoi(argv[6]);


		char fname[100];
		sprintf(fname, "normhistres/%d.dat", dataID);
		Tester tst(fname, 555, voltype, 1);
		for (int paci = 0; paci < 10; ++paci)
		{
			if (classifier==1 || classifier == 2)
				tst.TrainAndTest(classifier, paci, 1000*trainDataSize, param1, param2);
			else if (classifier == 3)
				tst.MultiSVM(paci, 30, param1, param2);
		}
	}
	else
		printf("Invalid command.\n\n");
	getch();
}
