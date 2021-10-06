#include <assert.h>

extern "C" {
  /*
  void fillCab(float * C, const float * a, const float * b, const int num_elems){
    #pragma HLS dataflow
    for(int i=0; i< num_elems; i++){
      #pragma HLS pipeline II=1
      C[i][0] = a[i];
      C[i][1] = b[i];
      C[i][2] = 0;
    }
  }

void fillEmpty(float * C, const int num_elems){
  #pragma HLS dataflow
  for(int j=num_elems; j < 4098; j++){
    #pragma HLS pipeline II=1
    C[j][0] = 0;
    C[j][1] = 0;
    C[j][1] = 0;
    }
  */
 
 
void dot_prod_kernel(const float *a, const float *b, float *c, const int num_elems) {
  /********  you can change AXI bus width  **********/
#pragma HLS interface m_axi port = a offset = slave bundle = gmem
#pragma HLS interface m_axi port = b offset = slave bundle = gmem
#pragma HLS interface m_axi port = c offset = slave bundle = gmem
#pragma HLS interface s_axilite port = a bundle = control
#pragma HLS interface s_axilite port = b bundle = control
#pragma HLS interface s_axilite port = c bundle = control
#pragma HLS interface s_axilite port = num_elems bundle = control
#pragma HLS interface s_axilite port = return bundle = control
  assert(num_elems <= 4096); // this helps HLS estimate the loop trip count
  /***************************
   * your code goes here ... *
   ***************************/
  //#pragma HLS array_partition variable=a type=block factor=2 dim=0
  //#pragma HLS array_partition variable=b type=block factor=2 dim=0
 
  /* Approach 2. Place values inside C. Partition C completely. Compute*/
  float C[4098][3];
  //#pragma HLS dataflow
  for(int i=0;i<num_elems;i++){
    #pragma HLS pipeline II=1 rewind 
    C[i][0] = a[i];
    C[i][1] = b[i];
    C[i][2] = 0;
   }

 for(int j = num_elems; j < 4098; j++){
    #pragma HLS pipeline II=1 rewind 
    C[j][0] = 0;
    C[j][1] = 0;
    C[j][2] = 0;
    }

  //  fillCab(C, a, b, num_elems);
  //fillEmpty(C, num_elems);

  // 282 optimal block size but doesn't compile
  #pragma HLS array_partition variable=C dim=0 block=2
  for(int i=0; i <= 4096; i++){
    #pragma HLS pipeline II=1 rewind
    #pragma HLS unroll factor=2
    C[i+1][2] = C[i][2] + C[i][0] * C[i][1];
  }

  *c = C[4097][2];
}


}  // extern "C"

/*
Read a,
Read b
ops * a b
write c
 */

/*
Load, compute, store pattern
 */

/*
18k
18000 / 2^5*2
 */

/*
current: 

float: 2^5
a = 2^12
b = 2^12

c = 2^5

totMem = 2^5 * 2^12 * 2 = 2^18 + 2^5 = 262176 

latency -> [1, 344207]

without unroll -> 24k
*/
