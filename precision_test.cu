#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <float.h>


__global__ void axpy(int n, double a, double* x, double* y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n){y[i] = a*x[i] + y[i];}
}

__global__ void axpy(int n, float a, float* x, float* y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n){y[i] = a*x[i] + y[i];}
}

__global__ void axpy(int n, half a, half* x, half* y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	half b = __hmul(a,x[i]);
	if (i < n){y[i] = __hadd(b,y[i]);}
}

__global__ void axpy(int n, nv_bfloat16 a, nv_bfloat16* x, nv_bfloat16* y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	half b = __hmul(a,x[i]);
	if (i < n){y[i] = __hadd(b,y[i]);}
}


int main() {
	int N = 1<<25;
	int threadsPerBlock = 1024;
	double *dx, *dy, *d_dx, *d_dy;
	float *sx, *sy, *d_sx, *d_sy;
	half *hx, *hy, *d_hx, *d_hy;
	nv_bfloat16 *bx, *by, *d_bx, *d_by;

	cudaEvent_t dstart, dstop, sstart, sstop, hstart, hstop, bstart, bstop;
	cudaEvent_t dMemCpyH2DStart, dMemCpyH2DStop, sMemCpyH2DStart, sMemCpyH2DStop, hMemCpyH2DStart, hMemCpyH2DStop, bMemCpyH2DStart, bMemCpyH2DStop;
	cudaEvent_t dMemCpyD2HStart, dMemCpyD2HStop, sMemCpyD2HStart, sMemCpyD2HStop, hMemCpyD2HStart, hMemCpyD2HStop, bMemCpyD2HStart, bMemCpyD2HStop;;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	cudaEventCreate(&dstart);
	cudaEventCreate(&dstop);
	cudaEventCreate(&sstart);
	cudaEventCreate(&sstop);
	cudaEventCreate(&hstart);
	cudaEventCreate(&hstop);
	cudaEventCreate(&bstart);
	cudaEventCreate(&bstop);

	cudaEventCreate(&dMemCpyH2DStart);
	cudaEventCreate(&dMemCpyH2DStop);
	cudaEventCreate(&sMemCpyH2DStart);
	cudaEventCreate(&sMemCpyH2DStop);
	cudaEventCreate(&hMemCpyH2DStart);
	cudaEventCreate(&hMemCpyH2DStop);
	cudaEventCreate(&bMemCpyH2DStart);
	cudaEventCreate(&bMemCpyH2DStop);

	cudaEventCreate(&dMemCpyD2HStart);
	cudaEventCreate(&dMemCpyD2HStop);
	cudaEventCreate(&sMemCpyD2HStart);
	cudaEventCreate(&sMemCpyD2HStop);
	cudaEventCreate(&hMemCpyD2HStart);
	cudaEventCreate(&hMemCpyD2HStop);
	cudaEventCreate(&bMemCpyD2HStart);
	cudaEventCreate(&bMemCpyD2HStop);


	dx = (double*)malloc(N*sizeof(double));
	dy = (double*)malloc(N*sizeof(double));
	sx = (float*)malloc(N*sizeof(float));
	sy = (float*)malloc(N*sizeof(float));
	hx = (half*)malloc(N*sizeof(half));
	hy = (half*)malloc(N*sizeof(half));
	bx = (nv_bfloat16*)malloc(N*sizeof(nv_bfloat16));
	by = (nv_bfloat16*)malloc(N*sizeof(nv_bfloat16));
	
	cudaMalloc(&d_dx, N*sizeof(double));
	cudaMalloc(&d_dy, N*sizeof(double));
	cudaMalloc(&d_sx, N*sizeof(float));
	cudaMalloc(&d_sy, N*sizeof(float));
	cudaMalloc(&d_hx, N*sizeof(half));
	cudaMalloc(&d_hy, N*sizeof(half));
	cudaMalloc(&d_bx, N*sizeof(nv_bfloat16));
	cudaMalloc(&d_by, N*sizeof(nv_bfloat16));

	
	srand (static_cast <unsigned> (time(0)));
	for (int i = 0; i< N; i++){
		dx[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
		dy[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
		sx[i] = static_cast <float> (dx[i]);
		sy[i] = static_cast <float> (dy[i]);
		hx[i] = __double2half(dx[i]);
		hy[i] = __double2half(dy[i]);
		bx[i] = __double2bfloat16(dx[i]);
		by[i] = __double2bfloat16(dy[i]);
		//printf ("%f, %f, %f, %f, %f, %f\n", dx[i], dy[i], sx[i], sy[i], __half2float(hx[i]), __half2float(hy[i]));
	}


	cudaEventRecord(dMemCpyH2DStart);
	cudaMemcpy(d_dx, dx, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dy, dy, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaEventRecord(dMemCpyH2DStop);

	cudaEventRecord(sMemCpyH2DStart);
	cudaMemcpy(d_sx, sx, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sy, sy, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(sMemCpyH2DStop);

	cudaEventRecord(hMemCpyH2DStart);
	cudaMemcpy(d_hx, hx, N*sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(d_hy, hy, N*sizeof(half), cudaMemcpyHostToDevice);
	cudaEventRecord(hMemCpyH2DStop);

	cudaEventRecord(bMemCpyH2DStart);
	cudaMemcpy(d_bx, bx, N*sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
	cudaMemcpy(d_by, by, N*sizeof(nv_bfloat16), cudaMemcpyHostToDevice);
	cudaEventRecord(hMemCpyH2DStop);

	int numThreadBlocks = (N+threadsPerBlock-1)/threadsPerBlock;
	cudaEventRecord(dstart);
	axpy<<<numThreadBlocks, threadsPerBlock>>>(N, static_cast <double> (2.0f), d_dx, d_dy);
	cudaEventRecord(dstop);

	cudaEventRecord(sstart);
	axpy<<<numThreadBlocks, threadsPerBlock>>>(N, 2.0f, d_sx, d_sy);
	cudaEventRecord(sstop);

	cudaEventRecord(hstart);
	axpy<<<numThreadBlocks, threadsPerBlock>>>(N, __float2half(2.0f), d_hx, d_hy);
	cudaEventRecord(hstop);

	cudaEventRecord(bstart);
	axpy<<<numThreadBlocks, threadsPerBlock>>>(N, __float2bfloat16(2.0f), d_bx, d_by);
	cudaEventRecord(bstop);


	cudaEventRecord(dMemCpyD2HStart);
	cudaMemcpy(dy, d_dy, N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(dMemCpyD2HStop);

	cudaEventRecord(sMemCpyD2HStart);
	cudaMemcpy(sy, d_sy, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(sMemCpyD2HStop);

	cudaEventRecord(hMemCpyD2HStart);
	cudaMemcpy(hy, d_hy, N*sizeof(half), cudaMemcpyDeviceToHost);
	cudaEventRecord(hMemCpyD2HStop);

	cudaEventRecord(bMemCpyD2HStart);
	cudaMemcpy(by, d_by, N*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
	cudaEventRecord(bMemCpyD2HStop);

	double dmaxError = 0;
	double smaxError = 0;
	double hmaxError = 0;
	double bmaxError = 0;

	double dTotalError = 0;
	double sTotalError = 0;
	double hTotalError = 0;
	double bTotalError = 0;

	for (int i = 0; i < N; i++){
		double dError = abs(dy[i]-dy[i]);
		double sError = abs((static_cast <double> (sy[i]))-dy[i]);
		double hError = abs(static_cast <double> (__half2float(hy[i])) - dy[i]);
		double bError = abs(static_cast <double> (__bfloat162float(by[i])) - dy[i]);
		dmaxError = max(dmaxError, dError);
		smaxError = max(smaxError, sError);
		hmaxError = max(hmaxError, hError);
		bmaxError = max(bmaxError, bError);
		dTotalError += dError;
		sTotalError += sError;
		hTotalError += hError;
		bTotalError += bError;
		//printf("%lg, %f, %f\n", dy[i], sy[i], __half2float(hy[i]));
	}

	cudaEventSynchronize(dstop);

	double dAvgError = dTotalError/N;
	double sAvgError = sTotalError/N;
	double hAvgError = hTotalError/N;
	double bAvgError = bTotalError/N;

	float d_H2D_milliseconds = 0;
	float s_H2D_milliseconds = 0;
	float h_H2D_milliseconds = 0;
	float b_H2D_milliseconds = 0;

	float d_compute_milliseconds = 0;
	float s_compute_milliseconds = 0;
	float h_compute_milliseconds = 0;
	float b_compute_milliseconds = 0;

	float d_D2H_milliseconds = 0;
	float s_D2H_milliseconds = 0;
	float h_D2H_milliseconds = 0;
	float b_D2H_milliseconds = 0;

	cudaEventElapsedTime(&d_H2D_milliseconds, dMemCpyH2DStart, dMemCpyH2DStop);
	cudaEventElapsedTime(&s_H2D_milliseconds, sMemCpyH2DStart, sMemCpyH2DStop);
	cudaEventElapsedTime(&h_H2D_milliseconds, hMemCpyH2DStart, hMemCpyH2DStop);
	cudaEventElapsedTime(&b_H2D_milliseconds, bMemCpyH2DStart, bMemCpyH2DStop);

	cudaEventElapsedTime(&d_compute_milliseconds, dstart, dstop);
	cudaEventElapsedTime(&s_compute_milliseconds, sstart, sstop);
	cudaEventElapsedTime(&h_compute_milliseconds, hstart, hstop);
	cudaEventElapsedTime(&b_compute_milliseconds, bstart, bstop);

	cudaEventElapsedTime(&d_D2H_milliseconds, dMemCpyD2HStart, dMemCpyD2HStop);
	cudaEventElapsedTime(&s_D2H_milliseconds, sMemCpyD2HStart, sMemCpyD2HStop);
	cudaEventElapsedTime(&h_D2H_milliseconds, hMemCpyD2HStart, hMemCpyD2HStop);
	cudaEventElapsedTime(&b_D2H_milliseconds, bMemCpyD2HStart, bMemCpyD2HStop);

	printf("-------------------------------------------------------------------------\n");
	printf("  Random Number A*X+Y on %d elements\n\n", N);
	printf("  %s\n",prop.name);
	printf("-------------------------------------------------------------------------\n");
	printf("|        | max error | avg error |  compute  | memcpy time | memcpy time |\n");
	printf("|  type  | to double | to double |    /ms    |  H2D /ms    |   D2H /ms   |\n");
	printf("-------------------------------------------------------------------------\n");
	printf("|        |           |           |           |             |             |\n");
	printf("|  fp64  |    N/A    |    N/A    | %6.2f    |   %6.2f    |   %6.2f    |\n", d_compute_milliseconds, d_H2D_milliseconds, d_D2H_milliseconds);
	printf("|        |           |           |           |             |             |\n");
	printf("-------------------------------------------------------------------------\n");
	printf("|        |           |           |           |             |             |\n");
	printf("|  fp32  | %1.3e | %1.3e | %6.2f    |   %6.2f    |   %6.2f    |\n", smaxError, sAvgError, s_compute_milliseconds, s_H2D_milliseconds, s_D2H_milliseconds);
	printf("|        |           |           |           |             |             |\n");
	printf("-------------------------------------------------------------------------\n");
	printf("|        |           |           |           |             |             |\n");
	printf("|  fp16  | %1.3e | %1.3e | %6.2f    |   %6.2f    |   %6.2f    |\n", hmaxError, hAvgError, h_compute_milliseconds, h_H2D_milliseconds, h_D2H_milliseconds);
	printf("|        |           |           |           |             |             |\n");
	printf("-------------------------------------------------------------------------\n");
	printf("|        |           |           |           |             |             |\n");
	printf("|  bf16  | %1.3e | %1.3e | %6.2f    |   %6.2f    |   %6.2f    |\n", bmaxError, bAvgError, b_compute_milliseconds, b_H2D_milliseconds, b_D2H_milliseconds);
	printf("|        |           |           |           |             |             |\n");
	printf("-------------------------------------------------------------------------\n");
	//printf("Max error between double, double: %1.5e, avg:%1.5e, took: %fms\n", dmaxError, dAvgError, d_compute_milliseconds);
	//printf("Max error between single, double: %1.5e, avg:%1.5e, took: %fms\n", smaxError, sAvgError, s_compute_milliseconds);
	//printf("Max error between half,   double: %1.5e, avg:%1.5e, took: %fms\n", hmaxError, hAvgError, h_compute_milliseconds);

	cudaFree(d_dx);
	cudaFree(d_dy);
	cudaFree(d_sx);
	cudaFree(d_sy);
	cudaFree(d_hx);
	cudaFree(d_hy);
	cudaFree(d_bx);
	cudaFree(d_by);
	free(dx);
	free(dy);
	free(sx);
	free(sy);
	free(hx);
	free(hy);
	free(bx);
	free(by);

}