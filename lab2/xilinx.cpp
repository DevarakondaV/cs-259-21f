#include "ap_int.h"
//#include <iostream>
//using namespace std;


const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

const int INPUT_X = kNum + kNum * kNum + (1 + kNum) / 2 * kNum;
const int INPUT_y = kInImSize*kInImSize;
const int INPUT_Z = kNum;

const int WEIGHT_X = kNum;
const int WEIGHT_Y = kKernel * kKernel;
const int WEIGHT_Z = kNum;

#define BLOCK_SIZE 512
#define FLOAT_SIZE 32
#define SYSTOLIC_INPUT_BLOCK_COUNT ((INPUT_X * INPUT_Y) * FLOAT_SIZE) / BLOCK_SIZE
#define SYSTOLIC_WEIGHT_BLOCK_COUNT ((WEIGHT_X * WEIGHT_Y) * FLOAT_SIZE) / BLOCK_SIZE // ((WEIGHT_X+WEIGHT_Y+WEIGHT_Z)*32) / 512

// typedef ap_int<BLOCK_SIZE> block;
// typedef systolicInput block[SYSTOLIC_INPUT_BLOCK_COUNT];
// typedef systolicWeight block[SYSTOLIC_WEIGHT_BLOCK_COUNT];

typedef systolicInput float[INPUT_X * INPUT_Y];
typedef systolicWeight float[WEIGHT_X * WEIGHT_Y];
typedef systolicOutput float[INPUT_X * INPUT_y]


#define weight(i, j, p, q) \
    weight[(i) * kNum * kKernel * kKernel + (j) * kKernel * kKernel + \
    (p) * kKernel + (q)]
#define input(j, h, w) \
    input[(j) * kInImSize * kInImSize + (h) * kInImSize + (w)]
#define output(i, h, w) \
    output[(i) * kOutImSize * kOutImSize + (h) * kOutImSize + (w)]

#define max(a, b) ((a) > (b) ? (a) : (b))

void loadWeight(hsl::stream<systolicWeight> & weightBuffer, const float* weight){
#pragma HLS inline off
  for (int i = 0; i < kNum ; i++){
    #pragma HLS pipeline
    systolicWeight weightsX;
    for(int p = 0; p < kKernel; p++){
      for(int q = 0; q < kKernel; q++){
        for(int j = 0; j < kNum; j+=25){
          weightX[j] = weight(i, j, p, q)
        }
      }
    }
    weightBuffer.write(weightsX);   
  }
}

void loadInput(hls::stream<systolicInput> & InputBuffer, const float * input){
#pragma HLS inline off
  for(int i = 0 ; i < kNum; i++){
    #pragma HLS pipeline
    systolicInput inputImg;
    for(int h = 0; h < kInImSize; h++){
      for(int w = 0; w < kInImSize; w++){
        for(int p = 0; p < kKernel; p++){
          for(int q = 0; q < kKernel; q++){
            inputImg[i][h+p][w+q] = input(j, h + p, w + q)
          }
        }
      }
    }
  }
}


void computeUnit(
  float ** systolic_array,
  float weight,
  float input,
  int loc_x, loc_y)
#pragma HLS inline off
  float prod = weight * input;
  systolic_array[loc_x][loc_y] = prod
  float weight_val = systolic_array[read_x][read_y];
  float input_val = input[input_x][input_y];
  float prod = weight_val * input_val;
  systolic_array[weight_x][weight_y+1] = weight_val;
  systolic_array[weight_x]
}

void compute(
  hls::stream<systolicInput> & inputBuffer,
  hls::stream<systolicWeight> & weightBuffer,
  hls::stream<systolicOutput> & outputBuffer){
#pragma HLS inline off
  for(int i = 0; i < kNum; i++){
    systolicInput input = inputBuffer.read();
    systolicWeight weight = weightBuffer.read();
    systolicOutput output;
    // #pragma HLS array_partition variable=input type=complete
    // #pragma HLS array_partition variable=weight type=complete
    // #pragma HLS array_partition variable=output type=complete
    for(int h = 0; h < kInImSize; h++){
      for(int w = 0; w < kInImSize; w++) {
        output[h] += weight[i][]
      }
    }

  }
}


extern "C"{
void CnnKernel(const float* input, const float* weight,
               const float* bias, float* output) {
// void CnnKernel(const block* input, const block* weight,
//                const block* bias, float* output){
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem1 depth=13307904
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem2 depth=1638400
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem3 depth=256
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem4 depth=3211264

#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control               
    
    hls::stream<image> imageBuffer;
    hls::stream<image> result;

    #pragma HLS STREAM variable=imageBuffer
    #pragma HLS STREAM variable=result

    #pragma HLS RESOURCE variable=imageBuffer core=FIFO_SRL
    #pragma HLS RESOURCE variable=result core=FIFO_SRL

    #pragma HLS dataflow
    load(imageBuffer, input);
    compute(imageBuffer, result);
    store(result, output);

    int IShape[3] = {
      kNum, 
      kInImSize*kInImSize + kInImSize * kNum + (kNum * kNum) / 4,
      kNum + kNum * kNum + ((1 + kNum) / 2 * kNum)};
    int KShape[3] = {kKernel*kKernel, kNum, kNum};


    static float C[kNum][kImSize][kImSize];

    for (int i = 0; i < kNum; ++i) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          C[i][h][w] = bias[i];
        }
      }
    }

  // const int kNum = 256;
  // const int kKernel = 5;
  // const int kImSize = 224;
  // const int kInImSize = 228;
  // const int kOutImSize = 112;
  // const int blockTotal = kInImSize * kInImSize;

  // weight[(i) * kNum * kKernel * kKernel + (j) * kKernel * kKernel + \
  //   (p) * kKernel + (q)]

  // #define input(j, h, w) \
  //   input[(j) * kInImSize * kInImSize + (h) * kInImSize + (w)]

  // Image -> flattened by row -> row0,row1,row2 ... 
  // 256 * w(0,0,0,0) -> weight[0 * 256 * 5 * 5 + 0 * 5 * 5 + 0 * 5 + 0] -> weight[0+0] = 0
  // 256 * w(1,1,0,0) -> weight[1 * 256 * 5 * 5 + 1 * 5 * 5 + 0 * 5 + 0] -> weight[]
    // Convolution
    for (int i = 0; i < kNum; ++i) {
      for (int j = 0; j < kNum; ++j) {
        for (int h = 0; h < kImSize; ++h) {
          for (int w = 0; w < kImSize; ++w) {
            for (int p = 0; p < kKernel; ++p) {
              for (int q = 0; q < kKernel; ++q)
                C[i][h][w] += weight(i, j, p, q) * input(j, h + p, w + q);
              }
            }
        }
      }
    }

	
	// ReLU
	for (int i = 0; i < kNum; ++i) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          C[i][h][w] = max(0.f, C[i][h][w]);
        }
      }
    }
	
	// Max pooling
    for (int i = 0; i < kNum; ++i) {
      for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w) {
          output(i, h, w) = max(
            max(C[i][h * 2][w * 2    ], C[i][h * 2 + 1][w * 2    ]),
            max(C[i][h * 2][w * 2 + 1], C[i][h * 2 + 1][w * 2 + 1]));
        }
      }
    }
  }
}



/*
Single Image: 228*228*32 = < 102 blocks
Total Images: 256
Total Blocks: 256 * 102 -> Exactly 25992 blocks

Task Level parallelization:
1) Each systolic array can be parallelized

Each Clock cycle: Move 1 value right


kernelMatrix -> 
Y = 25 + (25 * 25) + (1 + 25)/2 * 25 = 975
X = 975

KernelMatrix -> 975 X 975

ImageMatrix ->
Y = 256 * 256
X = 975

ImageMatrix -> 975 X 65536


float * 
j * kInImSize * kInImSize + (h+p) * kInImSize + (w+q)
j * kInImSize * kInImSize -> Image(j)
(h+p) * kInImSize -> row
(w+q) -> col 


abc
def
ghi

abcdefghi

(3,3) -> (h+p * kInImSize, w+q)
(3,2)
(3,1)
(2,3)
(2,2)
(2,1)
(1,3)
(1,2)
(1,1)

*/


void loadWeight(systolicWeight & weightBuffer, const block* weight){
#pragma HLS inline off
  for (int i = 0; i <  ; i++){
    #pragma HLS pipeline
    
  }
  for (int i = 0; i < kNum ; i++){
  }
}

void loadInput(hls::stream<systolicInput> & imageBuffer, const  block* input){

}

void load(hls::stream<image> & imageBuffer, const block* input){
#pragma HLS inline off
  for(int i=0;i<blockTotal;i++){
    #pragma HLS pipeline
    image curImage;
    imageBuffer.write(curImage);
  }
}

void compute(hls::stream<image> & imageBuffer, hls::stream<image> & result){
#pragma HLS inline off
  for(int i=0;i < imageCount; i++){
    #pragma HLS pipeline
    image curImage = imageBuffer.read();
    image newImage;
    // Relu
    // Pool
    result.write(newImage);
  }
}

void store(hls::stream<image> & result, float * output){
#pragma HLS inline off
  for(int i=0; i < kNum; i++){
    image resImage = result.read();
  }
}