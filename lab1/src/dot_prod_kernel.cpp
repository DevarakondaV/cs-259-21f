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
  float C[4097];
  //#pragma HLS array_partition variable=C type=block factor=2 dim=0
  for(int i=0; i < num_elems; i++){
    #pragma HLS pipeline II=1 style=stp
    #pragma HLS unroll factor=2
    //#pragma HLS dependence variable=c pointer type=intra dependent=true
    C[i] = a[i] * b[i];
      //*c = *c + (a[i]*b[i]);
  }

  // sum
  for(int i=0; i < num_elems; i++){
    *c = *c +  C[i];
  }
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
