#include <stdio.h> 
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <math.h>

// 16x16=64 threads per block
#define BLOCKDIMx 16   
#define BLOCKDIMy 16   

/**
 *
 * DFIRE distances
 *
 **/
__device__ static unsigned int dist_to_bins[50] = {
         1,  1,  1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 14, 15, 15, 16,
        16, 17, 17, 18, 18, 19, 19, 20, 20, 21,
        21, 22, 22, 23, 23, 24, 24, 25, 25, 26,
        26, 27, 27, 28, 28, 29, 29, 30, 30, 31};


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




__shared__ double receptor[BLOCKDIMx][3];
__shared__ double ligand[BLOCKDIMy][3];
//__device__ unsigned int nindex = 0;
//__device__ int ninterface = -1;
//__device__ double *penergy; 

void __global__ compute_distance(double ** rec_array, double ** lig_array, unsigned int rec_len, unsigned int lig_len,
                                 double * dist){

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

   	double sub1, sub2, sub3;
 	unsigned int index = i*lig_len +j; 

	sub1 = receptor[y][0] - ligand[x][0];
        sub2 = receptor[y][1] - ligand[x][1];
        sub3 = receptor[y][2] - ligand[x][2];
        dist[index] = sub1*sub1 + sub2*sub2 + sub3*sub3;

   }
}

__shared__ unsigned long receptor_obj[BLOCKDIMx];
__shared__ unsigned long ligand_obj[BLOCKDIMy];
void __global__ compute_neighbours( unsigned long * rec_obj, unsigned long * lig_obj, unsigned int rec_len, unsigned int lig_len, double * dist,
                                    unsigned int * interface_receptor,  unsigned int * interface_ligand, int *interface_len, double interface_cutoff, 
				    unsigned int * array, int *index_len){

   unsigned int i, j, x, y;
   x = threadIdx.x;
   y = threadIdx.y;

   i = blockIdx.y *blockDim.y +y;
   j = blockIdx.x *blockDim.x +x;
   if( (i<rec_len) && (j<lig_len))
   {
        if(x==0) {
                receptor_obj[y] = rec_obj[i];
        }
        __syncthreads();

        if(y==0){
                ligand_obj[x] = lig_obj[j];
        }
        __syncthreads();

	unsigned int sqrt_dist;
        unsigned int index = i*lig_len +j;
        unsigned long dfire_bin;    
        unsigned int prev_index;  

 	if (dist[index] <= 225.) {
		sqrt_dist = (sqrt(dist[index])*2.0 - 1.0);
		if (sqrt_dist < interface_cutoff){
                        prev_index = atomicAdd(interface_len, 1);
                	//atomicInc(&(*interface_len), rec_len);
                	interface_receptor[prev_index] = i;
                	interface_ligand[prev_index] = j;
		}
		dfire_bin = dist_to_bins[sqrt_dist] -1;
		//atomicInc(indexes_len, rec_len*lig_len);
		array[atomicAdd(index_len, 1)] = receptor_obj[y]*3360 + ligand_obj[x]*20 + dfire_bin;                
        }
    }
}

  
__device__ double compute_partial_energy(double * dfire_en_array, unsigned int * array, int N , unsigned int en_len){

   __shared__ double senergy[BLOCKDIMx*BLOCKDIMy];
   unsigned int i, x;
   x  =  threadIdx.x;
   i = blockDim.x*blockIdx.x +x;

   senergy[x] = 0.0;
   unsigned int index;
   double energyi = 0.0;
   while (i < N){
        index = array[i];
        if ( index < en_len )  energyi = dfire_en_array[index];
        senergy[x] += energyi;
        i += gridDim.x*blockDim.x;
   }

   for( unsigned int k = blockDim.x/2; k>0; k>>=1){
        __syncthreads();
        if(x < k)
            senergy[x] += senergy[x+k];
   }
   return senergy[x];
}

__device__ double compute_total_energy(volatile double * energy){

   __shared__ double tot_energy;
   unsigned int i, x;
   x  =  threadIdx.x;
   i = blockDim.x*blockIdx.x +x;
   unsigned int lastblock = 64; //gridDim.x-1;
   //if(blockIdx.x == lastblock){
  	 if (x == 0){
                unsigned int k = 0;
                while(k < lastblock)
        	tot_energy = energy[k++];
               
   	}
   	__syncthreads();
 /*
   	if (i < lastblock){
        	double penergy = energy[i];
        //	atomicAdd(&tot_energy, penergy);
       // tot_energy += energy[i];
   	}
*/
   return tot_energy;  

}


__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__global__ void compute_energy(double * dfire_en_array, unsigned int * array, int N, unsigned int en_len, volatile double* energy){
 
    double partial_energy = compute_partial_energy( dfire_en_array, array, N, en_len);

    if (threadIdx.x == 0) {
        energy[blockIdx.x] = partial_energy;

        __threadfence();

        unsigned int value = atomicInc(&count, gridDim.x);
        isLastBlockDone = (value == (gridDim.x - 1));
    }
    __syncthreads();

    if (isLastBlockDone) {
        if(blockIdx.x == (gridDim.x -1)){
        double total_energy = compute_total_energy(energy);
        if (threadIdx.x == 0) {
            energy[0] = total_energy;
            count = 0;
        }
    }
    }

}



/*
 *  Acceleratable code using GPU or OpenACC etc.
 *
 */


void compute_acc(double ** rec_array, double ** lig_array, unsigned int rec_len, unsigned int lig_len,
                    unsigned long * rec_obj, unsigned long * lig_obj, unsigned int ** interface_receptor,
                    unsigned int ** interface_ligand, double interface_cutoff, unsigned int *interface_len,
		    double * dfire_en_array, double *energy){
 
   unsigned int en_len = 564480;
   int * h_index_len = new int; 
   int * h_interface_len = new int; 
   double * h_energy = new double;
   double ** d_rec_array,  ** d_lig_array,* d_dist;
   unsigned long * d_rec_obj, * d_lig_obj;
   unsigned int  * d_interface_receptor, * d_interface_ligand;
   unsigned int  * d_array;
   int * d_interface_len, *d_index_len;
   double * d_dfire_en_array;
   double * d_energy;
   unsigned int nblocks = 64;
    
   size_t rec_bytes = rec_len*sizeof(double);
   size_t lig_bytes = lig_len*sizeof(double);
   size_t rec_lbytes = rec_len*sizeof(unsigned long);
   size_t lig_lbytes = lig_len*sizeof(unsigned long);
   
   unsigned int rl_len = rec_len*lig_len;
   size_t rl_ibytes = rl_len*sizeof(unsigned int);
   size_t rl_bytes = rl_len*sizeof(double);
   printf("rl: %d \t rec: %d \t lig: %d \n", rl_len, rec_len, lig_len);  
   (*interface_receptor) = (unsigned int *) malloc(rl_ibytes);
   (*interface_ligand) = (unsigned int *)malloc(rl_ibytes);
 
   unsigned int * array = (unsigned int *) malloc(rl_ibytes);

   cudaMalloc(&d_interface_len, sizeof(int));
   cudaCheck("Memory allocation for d_interface_len is failed ");
   cudaMalloc(&d_index_len, sizeof(int));
   cudaCheck("Memory allocation for d_index_len is failed ");

   cudaMalloc(&d_rec_array, rec_bytes);
   cudaCheck("Memory allocation for d_rec_array is failed ");
   cudaMalloc(&d_lig_array, lig_bytes);
   cudaCheck("Memory allocation for d_lig_array is failed ");
   cudaMalloc(&d_dist, rl_bytes);
   cudaCheck("Memory allocation for distance is failed ");
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
   cudaMemset(d_array, 0, rl_ibytes);
   cudaCheck("Memory set for d_array is failed ");
   cudaMemset(d_index_len, 0, sizeof(int));
   cudaCheck("Memory set for d_index_len is failed ");
   cudaMemset(d_interface_len, 0, sizeof(int));
   cudaCheck("Memory set for d_interface_len is failed ");
   
   unsigned int en_bytes = en_len*sizeof(double);  
   cudaMalloc(&d_dfire_en_array, en_bytes);
   cudaCheck("Memory allocation for d_dfire_en_array is failed ");
   cudaMalloc(&d_energy, nblocks*sizeof(double));
   cudaCheck("Memory allocation for d_energy is failed ");
  
   cudaMemset(d_energy, 0, nblocks*sizeof(double));
   cudaCheck("Memory set for d_energy is failed ");
   
   unsigned int nstreams = 3;
   cudaStream_t stream[nstreams];
   for( unsigned int i =0; i < nstreams; ++i)
        cudaStreamCreate(&stream[i]);
   
   cudaMemcpyAsync(d_rec_array, rec_array, rec_bytes, cudaMemcpyHostToDevice, stream[0]);
   cudaCheck("Data transfer from H2D for d_rec_array is failed ");
   cudaMemcpyAsync(d_lig_array, lig_array, lig_bytes, cudaMemcpyHostToDevice, stream[0]);
   cudaCheck("Data transfer from H2D for d_rec_array is failed ");
   
   cudaMemcpyAsync(d_rec_obj, rec_obj, rec_lbytes, cudaMemcpyHostToDevice, stream[1]);
   cudaCheck("Data transfer from H2D for d_rec_array is failed ");
   cudaMemcpyAsync(d_lig_obj, lig_obj, lig_lbytes, cudaMemcpyHostToDevice, stream[1]);
   cudaCheck("Data transfer from H2D for d_lig_obj is failed ");
   
   cudaMemcpyAsync(d_dfire_en_array, dfire_en_array, en_bytes, cudaMemcpyHostToDevice, stream[2]);
   cudaCheck("Data transfer from H2D for d_dfire_en_array is failed ");

   const dim3 blockSize(BLOCKDIMx, BLOCKDIMy);
   int GRIDDIMx = (BLOCKDIMx +rec_len -1)/BLOCKDIMx;
   int GRIDDIMy = (BLOCKDIMy +lig_len -1)/BLOCKDIMy;
   const dim3 gridSize( GRIDDIMx, GRIDDIMy);

  //  int nblocks = GRIDDIMx*GRIDDIMy;
  // cudaProfilerStart();

   compute_distance<<< gridSize, blockSize, 0, stream[0] >>>(d_rec_array, d_lig_array, rec_len, lig_len, d_dist);
   cudaCheck(" compute_dist kernel launching is failed ");
 
   cudaStreamSynchronize(stream[0]);    //--- Make sure the completion of the distance computation
   
   compute_neighbours<<< gridSize, blockSize, 0, stream[1] >>>( d_rec_obj, d_lig_obj, rec_len, lig_len, d_dist,
                         d_interface_receptor, d_interface_ligand, d_interface_len, interface_cutoff, d_array, d_index_len);
   cudaCheck(" compute_neighbour kernel launching is failed ");
   
   cudaMemcpyAsync(h_index_len, d_index_len, sizeof(int), cudaMemcpyDeviceToHost, stream[1]);
   cudaCheck("Data transfer from D2H for index_len is failed ");

   int index_len = *h_index_len; 
   const unsigned int nthreads = BLOCKDIMx *BLOCKDIMy;
 //  unsigned int nblocks = ceil((index_len +nthreads -1)/nthreads);
   const dim3 blockSize1D(nthreads);
   const dim3 gridSize1D(nblocks);
   
   cudaMemcpyAsync(h_interface_len, d_interface_len, sizeof(int), cudaMemcpyDeviceToHost, stream[1]);
   cudaCheck("Data transfer from D2H for interface_len is failed ");
   printf("h_interfac::  %d \n", (*h_interface_len));

   cudaMemcpyAsync((*interface_receptor), d_interface_receptor, rl_ibytes, cudaMemcpyDeviceToHost, stream[1]);
   cudaCheck("Data transfer from D2H for interface_receptor is failed ");
   cudaMemcpyAsync((*interface_ligand), d_interface_ligand, rl_ibytes, cudaMemcpyDeviceToHost, stream[1]);
   cudaCheck("Data transfer from D2H for interface_ligand is failed ");
  
   cudaStreamSynchronize(stream[1]);    //--- Make sure the completion of the neighbor list computation
   compute_energy<<< gridSize1D, blockSize1D, 0, stream[2]>>>(d_dfire_en_array, d_array, index_len, en_len, d_energy);
   cudaCheck(" calculate_energy kernel launching is failed ");

   cudaMemcpyAsync(h_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost, stream[2]);
   cudaCheck("Data transfer from D2H for energy is failed ");
   printf("energy: %f \n", (*h_energy));   

  // cudaProfilerStop();
   
   (*energy) = (*h_energy);
   (*interface_len) = (*h_interface_len);
   
   //Free Device memory
   cudaFree(d_rec_array);
   cudaFree(d_lig_array);
   cudaFree(d_dist);
   cudaFree(d_rec_obj);
   cudaFree(d_lig_obj);
   cudaFree(d_interface_receptor);
   cudaFree(d_interface_ligand);
   cudaFree(d_interface_len);
   cudaFree(d_index_len);
   cudaFree(d_array);
   cudaFree(d_dfire_en_array);
   cudaFree(d_energy);

   //---------------------------------------------------------

}



