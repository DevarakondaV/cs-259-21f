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
  #pragma HLS array_partition variable=a factor=num_elems dim=0
  #pragma HLS array_partition variable=b factor=num_elems dim=0

  for(int i=0; i < num_elems; i++){
    #pragma HLS dataflow
    #pragma HLS stable variable=c
    #pragma HLS unroll
    *c = *c + (a[i]*b[i])
  }
}

}  // extern "C"
