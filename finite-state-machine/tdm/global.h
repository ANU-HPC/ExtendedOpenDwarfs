#ifndef GLOBAL_H_
#define GLOBAL_H_

extern unsigned int MaxRecords;
extern unsigned int maxCandidates;
//const unsigned int maxLevel = 20;
extern unsigned int maxIntervals;
const unsigned int MaxListSize = 10;

static float support;

enum
{
	EVENT_26,
	EVENT_64,
};

// Device Dependent Values
static int MaxSharedMemory, MinThreads, MaxThreads, MaxSections;

#endif /* GLOBAL_H_ */
