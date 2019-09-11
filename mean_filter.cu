// Compile file => nvcc <input_filename.cu> -o image3 <output_filename>
// Run file => ./<filename> <bmp image> <window size>

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

unsigned char* mean_filter_h(unsigned char* matrix, int width, int height, int window_size) {
	int ignored_length = window_size/2;
	for(int i = ignored_length; i < (height-ignored_length); i++) {
		for(int j = ignored_length; j < (width-ignored_length); j++) {			
			// filter window handling
			int val_sum = 0;
			for(int k=i-ignored_length; k<=i+ignored_length;k++){
				for(int l=j-ignored_length;l<=j+ignored_length;l++){
					val_sum += matrix[(k)*width + (l)];
					// printf("%d ",matrix[(k)*width + (l)]);
				}
				// printf("\n");
			}
			// printf("===============================================\n");
			matrix[i*width + j] = val_sum/(window_size*window_size);
		}
	}
	return matrix;
}

__global__ void mean_filter_d(unsigned char* matrix, int length, int window_size) {
	int ignored = window_size/2;
	int i = blockIdx.x + ignored;
	int j = threadIdx.x + ignored;
	
	int val_sum = 0;
	for(int k = i-ignored; k < i+ignored+1; k++){
		for(int l = j-ignored; l < j+ignored+1; l++){
			val_sum += matrix[k*length + l];
		}
	}
	matrix[i*length + j] = val_sum/(window_size*window_size);
}

void saveBmp(unsigned char* info,unsigned char** rows,int row_padded,int height,char* filename){
    // write to output_file
    char outputfile[15];
    sprintf(outputfile,filename,height);

	FILE* fw = fopen(outputfile, "wb");
	fwrite(info, 54*sizeof(unsigned char), 1, fw);
	for(int i = 0; i<height; i++) {
		fwrite(rows[i], row_padded*sizeof(unsigned char), 1, fw);
	}
	fclose(fw);
}

int main(int argc, const char * argv[]) {
	
	const char* input_file = argv[1];
	int window_size = strtol(argv[2], NULL, 10);
	
	FILE* fr = fopen(input_file, "rb");
	if(fr == NULL) {
		return 2;
	}

	// read image header (54 bytes) and capture height and wigth
    unsigned char* info = (unsigned char*)malloc(54*sizeof(unsigned char));
    fread(info, sizeof(unsigned char), 54, fr); // read the 54-byte header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

	printf("Input file: %s \n",input_file);
	printf("Window size: %d \n",window_size);
	printf("Dimensions(%d,%d): \n", width, height);

	// row padding fix
    int row_padded = (width*3 + 3) & (~3);
	
	// pixel array (row major)
	unsigned char* matrix = (unsigned char*)malloc(height*width*sizeof(unsigned char));
	
	// binary data row matrix
	unsigned char** rows = (unsigned char**) malloc(height*sizeof(unsigned char*));
	unsigned char tmp;

	// fill pixel matrix and binary data matrix
    for(int i = 0; i < height; i++) {
		unsigned char* data = (unsigned char*)malloc(row_padded*sizeof(unsigned char));
        fread(data, sizeof(unsigned char), row_padded, fr);
        for(int j = 0; j < width*3; j += 3) {
            // Convert (B, G, R) to (R, G, B)
            tmp = data[j];
            data[j] = data[j+2];
            data[j+2] = tmp;
			//printf("i - %d, j - %d >> R-%d, G-%d, B-%d\n", i, j, (int)data[j], (int)data[j+1], (int)data[j+2]);
			int im = height-i-1; 
			int jm = j/3;
			matrix[im*width + jm] = data[j];
        }
		rows[i] = data;
    }
    
    // get cpu output to matrix
	unsigned char* matrix_out_h = (unsigned char*)malloc(height*width*sizeof(unsigned char));

	//CPU pixel array operation
   	printf("Running CPU Mean filter... ");
    clock_t start_h = clock();
	matrix_out_h = mean_filter_h(matrix,width,height,window_size);
    clock_t end_h = clock();
	printf(" => Done \n");

    printf("Saving CPU output... ");
	// save changes in binary data matrix from CPU
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width*3; j += 3) {
			int im = height-i-1; 
			int jm = j/3;
            rows[i][j] = matrix_out_h[im*width + jm];
			rows[i][j+1] = matrix_out_h[im*width + jm];
			rows[i][j+2] = matrix_out_h[im*width + jm];
        }
    }
	printf(" => Done \n");

	// Write CPU output
    saveBmp(info,rows,row_padded,height,"CPU_out_%d.bmp");

    // GPU pixel array operation
	int ignored = window_size/2;
	int grid_size = height - 2*ignored;
	int block_size = width - 2*ignored;
	
	unsigned char* matrix_d;
    // get GPU output to matrix
	unsigned char* matrix_out_d = (unsigned char*)malloc(height*width*sizeof(unsigned char));
	
	cudaMalloc((void **)&matrix_d, height*width*sizeof(unsigned char));
	cudaMemcpy(matrix_d, matrix, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice);
	printf("Running GPU Mean filter... ");
    clock_t start_d=clock();
	mean_filter_d<<<grid_size, block_size>>>(matrix_d, width, window_size);
    cudaThreadSynchronize();
    clock_t end_d=clock();
	printf(" => Done \n");
	cudaMemcpy(matrix_out_d, matrix_d, height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(matrix_d);

    printf("Saving GPU output... ");
    // save changes in binary data matrix from GPU
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width*3; j += 3) {
			int im = height-i-1; 
			int jm = j/3;
            rows[i][j] = matrix_out_d[im*width + jm];
			rows[i][j+1] = matrix_out_d[im*width + jm];
			rows[i][j+2] = matrix_out_d[im*width + jm];
        }
    }
	// Write GPU output
    saveBmp(info,rows,row_padded,height,"GPU_out_%d.bmp");
    printf(" => Done \n");
    fclose(fr);

    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
	double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;
    printf("\n******************************** FINAL OUT PUT ********************************\n");
	printf("Image dimensions: (%d,%d) \nGPU Time: %f \nCPU Time: %f\n",width,height,time_d,time_h);
    printf("*********************************************************************************\n");
    return 0;
}

