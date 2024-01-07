/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include "cuda.h"

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256
#define STREAMCOUNT 4

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;


struct timeval t_start, t_end;

void cputimer_start(){
  gettimeofday(&t_start, 0);
}
void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d, int *g);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
	//int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	int globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x; // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}

//added by Yitong Li

//swap two float numbers
__device__ void swap(float* a, float* b) {
    float t = *a;
    *a = *b;
    *b = t;
}

//swap two int numbers
__device__ void swap_int(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

/*
partition the array, and return the pivot index 
for quickselect
*/
__device__ int partition(float* arr, int low, int high) {
    float pivot = arr[high];
    int i = low;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            swap(&arr[i], &arr[j]);
            i++;
        }
    }
    swap(&arr[i], &arr[high]);
    return i;
}

/*
partition the array, and return the pivot index
for quicksort
*/
__device__ int partition_result(float* arr, int* index, int low, int high) {
    float pivot = arr[high];
    int i = low;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            swap(&arr[i], &arr[j]);
            swap_int(&index[i], &index[j]);

            i++;
        }
    }
    swap(&arr[i], &arr[high]);
    swap_int(&index[i], &index[high]);
    return i;
}


/*
quicksort function
*/
__device__ void quickSort(float* arr, int* index, int low, int high, int* stack) {
    int top = -1;
    stack[++top] = low;
    stack[++top] = high;
    while (top >= 0) {
        high = stack[top--];
        low = stack[top--];
        int pivot = partition_result(arr, index, low, high);
        if (pivot - 1 > low) {
            stack[++top] = low;
            stack[++top] = pivot - 1;
        }
        if (pivot + 1 < high) {
            stack[++top] = pivot + 1;
            stack[++top] = high;
        }
    }
}


/*
find the min "numMin" elements in the array from "offset_mul" to "offset_mul + offset"
the result is stored in "d_minLoc" from "offset_min" to "offset_min + numMin"
time complexity is O(numRecords/sqrt(numRecords/numMin))
find_min and find_min_final uses the same algorithm, Quickselect
*/

__global__ void find_min( float *d_distances, int numRecords, int *d_minLoc, int offset, int numMin, float *tmp, int streamOffset)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int low = 0, high = (globalId + 2)* offset > numRecords ? numRecords - globalId* offset - 1 : offset - 1;
    int max = high, offset_min = numMin * globalId, k = 0, offset_mul = offset * globalId, pivotIndex;
    float min_k;
    if(offset_mul + offset < numRecords){
        float* now = tmp + offset_mul;
        for(int i = 0; i <= max; i++){
            now[i] = d_distances[i + offset_mul];
        }
        //quickselect
        while (low <= high) {
            // Partition the array and get the pivot index
            pivotIndex = partition(now, low, high);
           if (pivotIndex == numMin){
                min_k = now[pivotIndex];
                break;
            }   
            // If numMin is less, continue to the left part
            else if (pivotIndex > numMin) high = pivotIndex - 1;

            // If numMin is more, continue to the right part
            else low = pivotIndex + 1;
        }
        //get the min "numMin" elements
        for(int i = 0; i < max; i++){
            if(d_distances[i+offset_mul] < min_k){
                d_minLoc[offset_min + k] = i + offset_mul + streamOffset;
                k++;
            }
        }
        //if there are elements equal in min_k, get them 
        if(k < numMin){
            for(int i = 0; i < max; i++){
                if((d_distances[i+offset_mul] == min_k) && k < numMin){
                    d_minLoc[offset_min + k] = i + offset_mul + streamOffset;
                    k++;
                }
            }
        }      
    }
  //printf("\n");
}
/*
find the min "numMin" element on the base of the result of "find_min"
the result is stored in "d_minLoc" from "0" to "numMin"
time complexity is O(sqrt(numRecords/numMin) * numMin)
*/
__global__ void find_min_final(float *d_distances, int num, int *d_minLoc, int numMin, float *min_dis, float *dis_min, int *d_minmem, int *stack)
{


    float min_k = 0;
    int pivotIndex;
    int low = 0, high = num - 1, k = 0;
    for(int i = 0; i < num; i++)
    {
        dis_min[i] = d_distances[d_minLoc[i]];
    }
    //quickselect
    while (low <= high) {
        // Partition the array and get the pivot index
        pivotIndex = partition(dis_min, low, high);
         // If pivot itself is the kth smallest element
        if (pivotIndex == numMin){
            min_k = dis_min[pivotIndex];
            break;
        }   
         // If k is less, continue to the left part
        else if (pivotIndex > numMin) high = pivotIndex - 1;

         // If k is more, continue to the right part
        else low = pivotIndex + 1;
    }
    //get the min "numMin" elements
    for(int i = 0; i < num; i++)
    {
        if(d_distances[d_minLoc[i]] < min_k)
        {
          d_minmem[k] = d_minLoc[i];
          k++;
        }
    }
    //if there are elements equal in min_k, get them
    if(k < numMin){
        for(int i = 0; i < num; i++){
          if((d_distances[d_minLoc[i]] == min_k )&& (k < numMin)){
            d_minmem[k] = d_minLoc[i];
            k++;
          }
        }
      }     

    for(int i = 0; i< numMin; i++){
      min_dis[i] = d_distances[d_minmem[i]];
    }

    //to make sure I and the original code have the same result, I sort the result
    //but for KNN, it is not necessary
    quickSort(min_dis, d_minmem, 0, numMin-1,stack);

}
//timer function
double cpuSecond() {
   cudaDeviceSynchronize();
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

//print the result
void printLowest(std::vector<Record> &records, int *min_record, int topN, float *dis_min){
  for(int i = 0; i < topN; i++)
    printf("%s --> Distance=%f\n",records[min_record[i]].recString,dis_min[i]);
}




/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{

  
  //cputimer_start();
	float lat, lng;
	int quiet=0,timing=0,platform=0,device=0;

  std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount=10;
  int numStreams = STREAMCOUNT; //Number of Streams
  int gridMin = 1;

    // parse command line
    if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device, &gridMin)) {
      printUsage();
      return 0;
    }

    int numRecords = loadData(filename,records,locations);
    int blockSize = numRecords / numStreams; //Number of Records for each Streams
    if (resultsCount > numRecords) resultsCount = numRecords;


  //Pointers to host memory
  float *dis_min;
  int *min_record;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances, *min_record_dis, *tmp, *tmp_final_float;
  int *d_minLoc, *d_minmem, *stack;


	// Scaling calculations - added by Sam Kauffman
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties( &deviceProp, 0 );
	cudaDeviceSynchronize();
	unsigned long maxGridX = deviceProp.maxGridSize[0];
	unsigned long threadsPerBlock = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK );
	size_t totalDeviceMemory;
	size_t freeDeviceMemory;
	cudaMemGetInfo(  &freeDeviceMemory, &totalDeviceMemory );
	cudaDeviceSynchronize();
	unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
	unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
	if ( numRecords > maxThreads )
	{
		fprintf( stderr, "Error: Input too large.\n" );
		exit( 1 );
	}
	unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing
	unsigned long gridY = ceilDiv( blocks, maxGridX );
	unsigned long gridX = ceilDiv( blocks, gridY );
	// There will be no more than (gridY - 1) extra blocks
	dim3 gridDim( gridX, gridY );
  dim3 grid_min(2 * (((int)(sqrt((float)blockSize/(float)resultsCount)+1) + threadsPerBlock - 1)/threadsPerBlock));

	if ( DEBUG )
	{
		print( totalDeviceMemory ); // 804454400
		print( freeDeviceMemory );
		print( usableDeviceMemory );
		print( maxGridX ); // 65535
		print( deviceProp.maxThreadsPerBlock ); // 1024
		print( threadsPerBlock );
		print( maxThreads );
		print( blocks ); // 130933
		print( gridY );
		print( gridX );
		print( numRecords );
		print( numStreams );
		print( blockSize );
	}


  // creat the Cuda Stream

  cudaStream_t streams[numStreams];
  for (int i = 0; i < numStreams; ++i) {
      cudaStreamCreate(&streams[i]);
  }


	/**
	* Allocate memory on host and device
	*/
  
    
    min_record = (int *)malloc(sizeof(int) * resultsCount * numStreams);
    dis_min = (float *)malloc(sizeof(float) * resultsCount * numStreams);


    cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords);
    cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);
    cudaMalloc((void **) &d_minLoc,sizeof(int) * (int)(sqrt((float)numRecords/(float)resultsCount)+1)* resultsCount * numStreams);
    cudaMalloc((void **) &min_record_dis,sizeof(float) * resultsCount);
    cudaMalloc((void **) &tmp,sizeof(float) * numRecords * numStreams);
    cudaMalloc((void **) &tmp_final_float,sizeof(float) * (int)(sqrt((float)numRecords/(float)resultsCount)+1)* resultsCount * numStreams);
    cudaMalloc((void **) &d_minmem,sizeof(float) * (int)(sqrt((float)numRecords/(float)resultsCount)+1)* resultsCount * numStreams);
    cudaMalloc((void **) &stack,sizeof(int) * resultsCount);

    //added by Guoqing Liang


    // start the main computation
    // copy locations to constant memory first, then use streams to overlap computation and memory copy
    // Then calculate the distance and find the min "numMin" elements in each stream in each partition
    for (int i = 0; i < numStreams; ++i) {


      int numRecordsForThisBlock = blockSize;
      if (i == numStreams - 1) {
        numRecordsForThisBlock = numRecords - i * blockSize; 
      }

      cudaMemcpyAsync(d_locations + i * blockSize, &locations[i * blockSize], 
                      sizeof(LatLong) * numRecordsForThisBlock, cudaMemcpyHostToDevice, streams[i]);


      euclid<<<gridDim, threadsPerBlock, 0, streams[i]>>>(d_locations + i * blockSize, 
                                                          d_distances + i * blockSize, 
                                                          numRecordsForThisBlock, lat, lng);

      
      cudaStreamSynchronize(streams[i]);

      
      int offset =(float)numRecordsForThisBlock / (int)(sqrt((float)numRecordsForThisBlock/(float)resultsCount)+1) + 1;
      find_min<<< grid_min , threadsPerBlock, 0, streams[i] >>>(d_distances + i * blockSize, numRecordsForThisBlock, d_minLoc + i * (int)(sqrt((float)blockSize/(float)resultsCount)+1) * resultsCount , offset, resultsCount, tmp + i * blockSize, i * blockSize);

    }
  
    // Wait for synchronization and destroy streams
    for (int i = 0; i < numStreams; i++) {
      cudaStreamSynchronize(streams[i]);
      cudaStreamDestroy(streams[i]);
    }

    //find the min "numMin" elements in the result of all the streams
    find_min_final<<< 1, 1 >>>(d_distances, (int)(sqrt((float)blockSize/(float)resultsCount)+1) * resultsCount * numStreams , d_minLoc, resultsCount, min_record_dis, tmp_final_float, d_minmem,stack);
    cudaDeviceSynchronize();

    //copy the result from device to host
    cudaMemcpy( min_record, d_minmem, sizeof(int)*resultsCount * numStreams, cudaMemcpyDeviceToHost);
    cudaMemcpy( dis_min, min_record_dis, sizeof(float)*resultsCount, cudaMemcpyDeviceToHost);

    //Free Device memory
    cudaFree(d_locations);
    cudaFree(d_distances);
    cudaFree(d_minLoc);
    cudaFree(min_record_dis);
    cudaFree(tmp);
    cudaFree(tmp_final_float);
    cudaFree(d_minmem);
    cudaFree(stack);


    // print out results
    if (!quiet)
    printLowest(records, min_record, resultsCount, dis_min);
    //cputimer_stop("Execuation Time");

    //Free Host memory
    free(min_record);
    free(dis_min);
  return 0;

}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d, int*g){
    int i;
    if (argc < 2) return 1; // error
    strncpy(filename,argv[1],100);
    char flag;

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
              break;
            case 'h': // help
              return 1;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
            case 'g': // number of grid
              i++;
              *g = atoi(argv[i]);
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
