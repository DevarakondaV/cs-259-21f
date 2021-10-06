#include <assert.h>

extern "C" {

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
  for(int i=0; i < num_elems; i++){
    #pragma HLS pipeline II=1 style=stp
    #pragma HLS unroll factor=2
    a[i] = a[i] * b[i];
  }

  // sum
  for(int i=0; i < num_elems; i++){
    *c = *c +  a[i];
  }



  /* Approach 2. Place values inside C. Partition C completely. Compute*/
  float C[4098][2];

  for(int i=0;i<num_elems;i++){
  #pragma HLS pipeline II=1 rewind 
    C[i][0] = a[i];
    C[i][1] = b[i];
  }

  for(int j = num_elems; i < 4098; i++){
  #pragma HLS pipeline II=1 rewind 
    C[i][0] = 0;
    C[i][1] = 0;
  }

  #pragma HLS array_partition variable=C dim=0
  for(int i=0; i <= 4096; i++){
    #pragma HLS unroll
    C[4097][0] = C[4097][0] + C[i][0] * C[i][1];
    // C[i][0] = C[i][0] * C[i][1];
  }

  /*
  for(int i = 0; i <= 4096; i++){
    #pragma HLS dataflow
    #pragma HLS dependence variable=C[4097] class=pointer type=inter direction=RAW dependent=true 
    C[4097][0] = C[4097][0] + C[i][0] * C[i][1]
  }*/

  *c = C[4097][0];
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
current: 

float: 2^5
a = 2^12
b = 2^12

c = 2^5

totMem = 2^5 * 2^12 * 2 = 2^18 + 2^5 = 262176 

latency -> [1, 344207]

without unroll -> 24k
*/
