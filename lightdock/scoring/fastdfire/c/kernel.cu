#include <stdio.h> 
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
/**
 *
 * DFIRE distances
 *
 **/
static unsigned int dist_to_bins[50] = {
         1,  1,  1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 14, 15, 15, 16,
        16, 17, 17, 18, 18, 19, 19, 20, 20, 21,
        21, 22, 22, 23, 23, 24, 24, 25, 25, 26,
        26, 27, 27, 28, 28, 29, 29, 30, 30, 31};

// 16x16=64 threads per block
#define BLOCKDIMx 16   
#define BLOCKDIMy 16   


// error checking macro
#define cudaCheck(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// Calculate the partial energy by each block
__device__ double calculate_partial_energy( const double * dfire_en_array, unsigned int N, unsigned int * array){

    extern __shared__ double senergy[];
    unsigned int k;
    unsigned int x = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x +x;
    unsigned int index = i; //array[i];
    senergy[index] = dfire_en_array[index];
   __syncthreads();

   for ( k = 1; k < blockDim.x; k*2 ){
        if (x%(2*k) == 0) senergy[x] += senergy[x+k];
        __syncthreads();
   }
   if (x ==0 ) return senergy[0];
}



// Calculate the total energy by the last block
__device__ double calculate_total_energy(volatile double * energy)
{
    double tot_energy = 0.0;
    unsigned int i, nblocks = gridDim.x;

    for (i = 0; i < nblocks; i++)
        tot_energy += energy[i];

    return tot_energy;

}



__shared__ double receptor[BLOCKDIMx][3];
__shared__ double ligand[BLOCKDIMy][3];
__shared__ bool is_lastblock_done;
__device__ unsigned int block_count = 0;
__device__ unsigned int nindex = 0;
//__device__ int ninterface = -1;
//__device__ double *penergy; 

void __global__ computegpu(double ** rec_array, double ** lig_array, unsigned int rec_len, unsigned int lig_len,
                    unsigned long * rec_obj, unsigned long * lig_obj, unsigned int * interface_receptor,
                    unsigned int * interface_ligand, unsigned int * array, double interface_cutoff, unsigned int *interface_len,
                    double * dfire_en_array, double *energy){


   unsigned int i, j, x, y;
   x = threadIdx.x;
   y = threadIdx.y;

   i = blockIdx.y *blockDim.y +y;
   j = blockIdx.x *blockDim.x +x;
   
   if( (i<rec_len) && (j<lig_len))
   {
   	if(x==0) {
       		receptor[y][0] = rec_array[i][0];
		receptor[y][1] = rec_array[i][1];
		receptor[y][2] = rec_array[i][2];
   	}
	__syncthreads();

   	if(y==0){
		ligand[x][0] = lig_array[j][0];
		ligand[x][1] = lig_array[j][1];
		ligand[x][2] = lig_array[j][2];
   	}
	__syncthreads();

   	double sub1, sub2, sub3, dist;
	unsigned int sqrt_dist;
 	unsigned int index = i*lig_len +j; 
	unsigned long atoma, atomb, dfire_bin;

	sub1 = receptor[y][0] - ligand[x][0];
        sub2 = receptor[y][1] - ligand[x][1];
        sub3 = receptor[y][2] - ligand[x][2];
        dist = sub1*sub1 + sub2*sub2 + sub3*sub3;
  
 	if (dist <= 225.) {
		sqrt_dist = (sqrt(dist)*2.0 - 1.0);
		if (sqrt_dist < interface_cutoff){
                	atomicInc(&(*interface_len), rec_len);
                	interface_receptor[(*interface_len)] = i;
                	interface_ligand[(*interface_len)] = j;
		}
		atoma = rec_obj[i];
 		atomb = lig_obj[j];
		dfire_bin = 1; //dist_to_bind[sqrt_dist] -1;
		//atomicInc(&indexes_len, rec_len*lig_len);
		//array[indexes_len] = atoma*3360 + atomb*20 + dfire_bin;                
		atomicInc(&nindex, rec_len*lig_len);
		index = atoma*3360 + atomb*20 + dfire_bin;                
		array[nindex] = atoma*3360 + atomb*20 + dfire_bin;                
        }


    }
    __syncthreads();
   

  // extern  double penergy[];
 
   double partial_energy = calculate_partial_energy( dfire_en_array, nindex, array);
   if (x==0){
	energy[blockIdx.x] = partial_energy;
	__threadfence();
       	unsigned int prev_block_count = atomicInc(&block_count, gridDim.x);
	is_lastblock_done = (prev_block_count==(gridDim.x -1));
   }
   __syncthreads();
   if (is_lastblock_done){
	double total_energy = calculate_total_energy(energy);
        if (x==0){
		//(*energy) = total_energy;
		energy[0] = total_energy;
     		block_count = 0;
	}

   }

}

/*
 //Optimize ...
 //Calculate the partial energy by each block
__device__ double calculate_partial_energy( const double * dfire_en_array){
    
    extern __shared__ double senergy[];

    unsigned int k;
    unsigned int x = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2 + x;
    senergy[x] = dfire_en_array[i] + dfire_en_array[i + blockDim.x]
    __syncthreads();
 
    unsigned int k;
    for ( k = blockDim.x/2; k > 32; k >>=1){
        if(x<k)
        senergy[x] +=senergy[x +k];
        __syncthreads();
    }
    if (x < 32) calculate_warp(senergy, x) 
    if ( x==0 ) return senergy[0];
}

 //Clculate warp level energy by unrolling
__device__ void calculate_warp(volatile double * senergy){
  // blockDim should be >= 64 ...
   senergy[x] += senergy[x +32];
   senergy[x] += senergy[x +16];
   senergy[x] += senergy[x +8];
   senergy[x] += senergy[x +4];
   senergy[x] += senergy[x +2];
   senergy[x] += senergy[x +1];
}
*/


/*
 *  Acceleratable code using GPU or OpenACC etc.
 *
 */


void compute_acc(double ** rec_array, double ** lig_array, unsigned int rec_len, unsigned int lig_len,
                    unsigned long * rec_obj, unsigned long * lig_obj, unsigned int ** interface_receptor,
                    unsigned int ** interface_ligand, double interface_cutoff, unsigned int *interface_len,
		    double * dfire_en_array, double *energy){
  
   interface_len = new unsigned int; 
   //double * h_energy;
   energy = new double;
   double ** d_rec_array,  ** d_lig_array;
   unsigned long * d_rec_obj, * d_lig_obj;
   unsigned int  * d_interface_receptor, * d_interface_ligand;
   unsigned int  * d_array;
   double * d_energy, * d_dfire_en_array;
   unsigned int * d_interface_len;
    
   size_t rec_bytes = rec_len*sizeof(double);
   size_t lig_bytes = lig_len*sizeof(double);
   size_t rec_lbytes = rec_len*sizeof(unsigned long);
   size_t lig_lbytes = lig_len*sizeof(unsigned long);
   //size_t rec_ibytes = rec_len*sizeof(unsigned int);
   //size_t lig_ibytes = lig_len*sizeof(unsigned int);
   
   unsigned int rl_len = rec_len*lig_len;
   size_t rl_ibytes = rl_len*sizeof(unsigned int);
   size_t rl_bytes = rl_len*sizeof(double);
   
   unsigned int ** array = (unsigned int **) malloc(rl_ibytes);

   cudaMalloc((void**)&d_interface_len, sizeof(unsigned int));
   cudaCheck("Memory allocation for d_interface_len is failed ");
   cudaMalloc(&d_rec_array, rec_bytes);
   cudaCheck("Memory allocation for d_rec_array is failed ");
   cudaMalloc(&d_lig_array, lig_bytes);
   cudaCheck("Memory allocation for d_lig_array is failed ");
   cudaMalloc(&d_rec_obj, rec_lbytes);
   cudaCheck("Memory allocation for d_rec_obj is failed ");
   cudaMalloc(&d_lig_obj, lig_lbytes);
   cudaCheck("Memory allocation for d_lig_obj is failed ");
   cudaMalloc(&d_interface_receptor, rl_ibytes);
   cudaCheck("Memory allocation for d_interface_receptor is failed ");
   cudaMalloc(&d_interface_ligand, rl_ibytes);
   cudaCheck("Memory allocation for d_interface_ligand is failed ");
   cudaMalloc(&d_array, rl_ibytes);
   cudaCheck("Memory allocation for d_array is failed ");
   cudaMalloc(&d_dfire_en_array, rl_bytes);
   cudaCheck("Memory allocation for d_dfire_en_array is failed ");

   cudaMemcpyAsync(d_rec_array, rec_array, rec_bytes, cudaMemcpyHostToDevice, 0);
   cudaCheck("Data transfer from HtoD for d_rec_array is failed ");
   cudaMemcpyAsync(d_lig_array, lig_array, lig_bytes, cudaMemcpyHostToDevice, 0);
   cudaCheck("Data transfer from HtoD for d_rec_array is failed ");
   
   cudaMemcpyAsync(d_rec_obj, rec_obj, rec_lbytes, cudaMemcpyHostToDevice, 0);
   cudaCheck("Data transfer from HtoD for d_rec_array is failed ");
   cudaMemcpyAsync(d_lig_obj, lig_obj, lig_lbytes, cudaMemcpyHostToDevice, 0);
   cudaCheck("Data transfer from HtoD for d_lig_obj is failed ");
   
   cudaMemcpyAsync(d_dfire_en_array, dfire_en_array, rl_bytes, cudaMemcpyHostToDevice, 0);
   cudaCheck("Data transfer from HtoD for d_dfire_en_array is failed ");
// need to check the array - should be a local variable for the kernel

   const dim3 blockSize(BLOCKDIMx, BLOCKDIMy);
   int GRIDDIMx = (BLOCKDIMx +rec_len -1)/BLOCKDIMx;
   int GRIDDIMy = (BLOCKDIMy +rec_len -1)/BLOCKDIMy;
   const dim3 gridSize( GRIDDIMx, GRIDDIMy);

   int nblocks = GRIDDIMx*GRIDDIMy;
   cudaMalloc(&d_energy, nblocks*sizeof(double));
   cudaCheck("Memory allocation for d_energy is failed ");  

   cudaProfilerStart();
   computegpu<<< gridSize, blockSize, 0, 0>>>(d_rec_array, d_lig_array, rec_len, lig_len,
                   d_rec_obj, d_lig_obj, d_interface_receptor,
                   d_interface_ligand, d_array, interface_cutoff, d_interface_len,
                   d_dfire_en_array, d_energy);

   cudaCheck("Kernel computegpu is failed ");

   cudaMemcpyAsync(energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost, 0);
   cudaCheck("Data transfer from DtoH for energy is failed ");
   cudaMemcpyAsync(interface_len, d_interface_len, sizeof(int), cudaMemcpyDeviceToHost, 0);
   cudaCheck("Data transfer from DtoH for interface_len is failed ");

   cudaMemcpyAsync(interface_receptor, d_interface_receptor, rl_ibytes, cudaMemcpyDeviceToHost, 0);
   cudaCheck("Data transfer from DtoH for interface_receptor is failed ");
   cudaMemcpyAsync(interface_ligand, d_interface_ligand, rl_ibytes, cudaMemcpyDeviceToHost, 0);
   cudaCheck("Data transfer from DtoH for interface_ligand is failed ");

   cudaProfilerStop();
   printf("energy: %.f/n", (*energy));   
   //(*energy) = (*h_energy);
   //Free Device memory
   cudaFree(d_rec_array);
   cudaFree(d_lig_array);
   cudaFree(d_rec_obj);
   cudaFree(d_lig_obj);
   cudaFree(d_interface_receptor);
   cudaFree(d_interface_ligand);
   cudaFree(d_dfire_en_array);
   cudaFree(d_array);

  

   //---------------------------------------------------------
   /*
   cudaStream_t strem[nstreams];
   for( unsigned int i =0; i < nstreams; ++i)
	cudaStreamCreate(&stream[i]);

    unsigned int threadsPerBlock = 256;
    unsigned int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    const unsigned int streamSize = n / nstreams;
    const int streamBytes = streamSize * sizeof(double);

    for (unsigned int i = 0; i < nstreams; ++i)
    {
       unsigned int offset = i*streamSize;
       cudaMemcpyAsync(&d_rec_array[offset], &rec_array[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
       cudaMemcpyAsync(&d_lig_array[offset], &lig_array[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
       add_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_rec_array, d_lig_array, rec_lig, lig_len, offset);
    }
    for (unsigned int i = 0; i < nstreams; ++i)
       cudaStreamDestroy(stream[i]);

  */ 
   

}



