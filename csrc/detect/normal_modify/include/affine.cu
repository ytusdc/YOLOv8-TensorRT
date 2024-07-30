#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cmath>
#include<stdio.h>
#include"utils.hpp"

typedef unsigned char uint8_t;

__device__ void affine_project(float* matrix,int x,int y,float* proj_x,float* proj_y) {
	// m0, m1, m2
	// m3, m4, m5
	*proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
	*proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}



__global__ void warp_affine_bilinear_kernel(
	uint8_t* src, int src_line_size, int src_width, int src_height,
	float* dst,int dst_width, int dst_height,uint8_t fill_value, AffineMatrix matrix)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx >= dst_width || dy >= dst_height) return;

	float c0 = fill_value, c1 = fill_value, c2 = fill_value;
	float src_x = 0; float src_y = 0;
	affine_project(matrix.d2i, dx, dy, &src_x, &src_y);
	
	if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height) {
		//src_x<-1时，其高位high_x<0，超出范围
		//src_x>=-1时，其高位high_x>=0，存在取值
	}
	else {
		int x_low = std::floor(src_x);
		int y_low = std::floor(src_y);
		int x_high = x_low + 1;
		int y_high = y_low + 1;

		uint8_t const_values[] = {fill_value,fill_value,fill_value};
		float lx = src_x - x_low;
		float ly = src_y - y_low;
		float hx = 1 - lx;
		float hy = 1 - ly;
		float w1 = hx * hy, w2 = lx * hy, w3 = hx * ly, w4 = lx * ly;
		uint8_t* v1 = const_values;
		uint8_t* v2 = const_values;
		uint8_t* v3 = const_values;
		uint8_t* v4 = const_values;
		if (y_low >= 0) {
			if (x_low >= 0)
				v1 = src + y_low * src_line_size + x_low * 3;

			if (x_high < src_width)
				v2 = src + y_low * src_line_size + x_high * 3;
		}

		if (y_high < src_height) {
			if (x_low >= 0)
				v3 = src + y_high * src_line_size + x_low * 3;

			if (x_high < src_width)
				v4 = src + y_high * src_line_size + x_high * 3;
		}

		c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
		c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
		c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);

	}

	//uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
	//pdst[0] = c0; pdst[1] = c1; pdst[2] = c2;

	float* pdst = dst + dy * dst_width+dx;
	pdst[0] = c2/255.0f; 
	pdst[dst_width*dst_height] = c1/255.0f;
	pdst[2*dst_width*dst_height] = c0/255.0f;
	
}


void warp_affine_bilinear(uint8_t* src, int src_line_size, int src_width, int src_height,
	float* dst,int dst_width, int dst_height, uint8_t fill_value, AffineMatrix matrix, cudaStream_t stream)


{
	dim3 block_size(32, 32);//blocksize最大1024
	dim3 grid_size((dst_width+31)/32,(dst_height+31)/32);

	warp_affine_bilinear_kernel <<<grid_size, block_size,0,stream>> > (
		src, src_line_size, src_width, src_height,
		dst, dst_width, dst_height,
		fill_value,matrix);

	//printf("核函数完成\n");
}

