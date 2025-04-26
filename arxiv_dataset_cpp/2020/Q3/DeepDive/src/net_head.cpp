#include <hw/net_head.hpp>
// #include <hw/DataCopyWeights.hpp>
//for 'copy_bias_zp' function
//#include <hw/DataCopy_Body.hpp>

//---------------------------- TOP LEVEL FUNCTION -------------------------------------------//
void compute_head(dType_8u *input_feature,
                  dType_8u *weights,
                  dType_8u *biases,
                  dType_8u *out_0,

                  dType_8u *weight_zp,
                  dType_8u *iMult_bias_acc,
                  dType_8t *nShift_bias_acc,
                  dType_8u *iMult_output,
                  dType_8u *nShift_output,

                  //opzp
                  dType_8u opzp1,
                  dType_8u opzp2,
                  dType_8u opzp3,
                  dType_8u opzp4,
                  dType_8u bzp_nc,
                  dType_8u bzp_dw,
                  dType_8u bzp_pw)
{

  dType_8u bias_zeros[__TOTAL_BIAS_ZERO_POINTS_LENGTH__];
#pragma HLS ARRAY_PARTITION variable = bias_zeros complete dim = 0

  dType_8u weights_nc[__FEATURES_0_0__OUTPUT_CHAN__][__FEATURES_0_0__INPUT_CHAN__ * __FEATURES_0_0__KERNEL_SIZE__ * __FEATURES_0_0__KERNEL_SIZE__];
#pragma HLS ARRAY_PARTITION variable = weights_nc complete dim = 2

  //Layer 1
  dType_8u biases_local_1[__FEATURES_0_0_BASE_B_Q_LENGTH__];
  dType_8u weights_zp_1[__FEATURES_0_0_W_ZERO_POINT_LENGTH__];
  dType_8u downscaling_iMult_1[__FEATURES_0_0_W_ZERO_POINT_LENGTH__];
  dType_8u downscaling_nShift_1[__FEATURES_0_0_W_ZERO_POINT_LENGTH__];
  dType_8t nShif_biases_acc_1[__FEATURES_0_0_W_ZERO_POINT_LENGTH__];
  dType_8u iMult_biases_acc_1[__FEATURES_0_0_W_ZERO_POINT_LENGTH__];

  //Layer 2

  dType_4u weights_dw[__FEATURES_1_CONV_0__OUTPUT_CHAN__][ALIGNED_KERNEL_SIZE(3)];
#pragma HLS ARRAY_PARTITION variable = weights_dw complete dim = 2
  dType_4u biases_local_2[__FEATURES_1_CONV_0_BASE_B_Q_LENGTH__];
  dType_4u weights_zp_2[__FEATURES_1_CONV_0_W_ZERO_POINT_LENGTH__];
  dType_8u downscaling_iMult_2[__FEATURES_1_CONV_0_W_ZERO_POINT_LENGTH__];
  dType_8u downscaling_nShift_2[__FEATURES_1_CONV_0_W_ZERO_POINT_LENGTH__];
  dType_8t nShif_biases_acc_2[__FEATURES_1_CONV_0_W_ZERO_POINT_LENGTH__];
  dType_8u iMult_biases_acc_2[__FEATURES_1_CONV_0_W_ZERO_POINT_LENGTH__];

  //Layer 3
  // dType_4u weights_pw[__FEATURES_1_CONV_2__OUTPUT_CHAN__ * __FEATURES_1_CONV_2__INPUT_CHAN__];
  dType_4u weights_pw[__FEATURES_1_CONV_2__OUTPUT_CHAN__][__FEATURES_1_CONV_2__INPUT_CHAN__];
#pragma HLS ARRAY_PARTITION variable = weights_pw complete dim = 2
  dType_4u biases_local_3[__FEATURES_1_CONV_2_BASE_B_Q_LENGTH__];
  dType_4u weights_zp_3[__FEATURES_1_CONV_2_W_ZERO_POINT_LENGTH__];
  dType_8u downscaling_iMult_3[__FEATURES_1_CONV_2_W_ZERO_POINT_LENGTH__];
  dType_8u downscaling_nShift_3[__FEATURES_1_CONV_2_W_ZERO_POINT_LENGTH__];
  dType_8t nShif_biases_acc_3[__FEATURES_1_CONV_2_W_ZERO_POINT_LENGTH__];
  dType_8u iMult_biases_acc_3[__FEATURES_1_CONV_2_W_ZERO_POINT_LENGTH__];

  fused_pipes Pipe_0("HDDR");
  fused_pipes Pipe_1("HNC");
  fused_pipes Pipe_2("HDW");
  fused_pipes Pipe_3("HPW");

#pragma HLS DATAFLOW

  burst_read_weights_head<__FEATURES_0_0__OUTPUT_CHAN__, __FEATURES_0_0__INPUT_CHAN__, __FEATURES_0_0__KERNEL_SIZE__,
                          __FEATURES_1_CONV_0__OUTPUT_CHAN__,
                          ALIGNED_KERNEL_SIZE(3),
                          __FEATURES_1_CONV_2__OUTPUT_CHAN__,
                          __FEATURES_1_CONV_2__INPUT_CHAN__>(weights, weights_nc, weights_dw, weights_pw,
                                                             __FEATURES_0_0__WEIGHT_LENGTH__, __FEATURES_1_CONV_0__WEIGHT_LENGTH__,
                                                             __FEATURES_1_CONV_2__WEIGHT_LENGTH__);

  copy_biases_head<__FEATURES_0_0_B_SCALE_LENGTH__,
                   __FEATURES_0_0_B_SCALE_OFFSET__,
                   __FEATURES_1_CONV_0_B_SCALE_LENGTH__,
                   __FEATURES_1_CONV_0_B_SCALE_OFFSET__,
                   __FEATURES_1_CONV_2_B_SCALE_LENGTH__,
                   __FEATURES_1_CONV_2_B_SCALE_OFFSET__>(biases,
                                                         biases_local_1,
                                                         biases_local_2,
                                                         biases_local_3);

  copy_weight_zp<__FEATURES_0_0_W_ZERO_POINT_LENGTH__,
                 __FEATURES_0_0_W_ZERO_POINT_OFFSET__,
                 __FEATURES_1_CONV_0_W_ZERO_POINT_LENGTH__,
                 __FEATURES_1_CONV_0_W_ZERO_POINT_OFFSET__,
                 __FEATURES_1_CONV_2_W_ZERO_POINT_LENGTH__,
                 __FEATURES_1_CONV_2_W_ZERO_POINT_OFFSET__>(weight_zp,
                                                            weights_zp_1,
                                                            weights_zp_2,
                                                            weights_zp_3);

  //copy_bias_zp_head<__FEATURES_0_0__OUTPUT_CHAN__, __FEATURES_1_CONV_0__INPUT_CHAN__, __FEATURES_1_CONV_2__OUTPUT_CHAN__>(weight_zp, bias_zeros_l1, bias_zeros_l2, bias_zeros_l3);
  // copy_output_zp(output_zp, opzp);

  copy_iMult_downscaling<__FEATURES_0_0_ACCUM_SCALE_LENGTH__,
                         __FEATURES_0_0_ACCUM_SCALE_OFFSET__,
                         __FEATURES_1_CONV_0_ACCUM_SCALE_LENGTH__,
                         __FEATURES_1_CONV_0_ACCUM_SCALE_OFFSET__,
                         __FEATURES_1_CONV_2_ACCUM_SCALE_LENGTH__,
                         __FEATURES_1_CONV_2_ACCUM_SCALE_OFFSET__>(iMult_output,
                                                                   downscaling_iMult_1,
                                                                   downscaling_iMult_2,
                                                                   downscaling_iMult_3);

  copy_nShift_downscaling<__FEATURES_0_0_BASE_B_Q_LENGTH__,
                          __FEATURES_0_0_BASE_B_Q_OFFSET__,
                          __FEATURES_1_CONV_0_BASE_B_Q_LENGTH__,
                          __FEATURES_1_CONV_0_BASE_B_Q_OFFSET__,
                          __FEATURES_1_CONV_2_BASE_B_Q_LENGTH__,
                          __FEATURES_1_CONV_2_BASE_B_Q_OFFSET__>(nShift_output,
                                                                 downscaling_nShift_1,
                                                                 downscaling_nShift_2,
                                                                 downscaling_nShift_3);

  copy_iMult_upscaling<__FEATURES_0_0_B_SCALE_LENGTH__,
                       __FEATURES_0_0_B_SCALE_OFFSET__,
                       __FEATURES_1_CONV_0_B_SCALE_LENGTH__,
                       __FEATURES_1_CONV_0_B_SCALE_OFFSET__,
                       __FEATURES_1_CONV_2_B_SCALE_LENGTH__,
                       __FEATURES_1_CONV_2_B_SCALE_OFFSET__>(iMult_bias_acc,
                                                             iMult_biases_acc_1,
                                                             iMult_biases_acc_2,
                                                             iMult_biases_acc_3);

  copy_nShift_upscaling<__FEATURES_0_0_BASE_B_Q_LENGTH__,
                        __FEATURES_0_0_BASE_B_Q_OFFSET__,
                        __FEATURES_1_CONV_0_BASE_B_Q_LENGTH__,
                        __FEATURES_1_CONV_0_BASE_B_Q_OFFSET__,
                        __FEATURES_1_CONV_2_BASE_B_Q_LENGTH__,
                        __FEATURES_1_CONV_2_BASE_B_Q_OFFSET__>(nShift_bias_acc,
                                                               nShif_biases_acc_1,
                                                               nShif_biases_acc_2,
                                                               nShif_biases_acc_3);

  readImage<__FEATURES_0_0__INPUT_CHAN__, __FEATURES_0_0__INPUT_SIZE__>(input_feature, Pipe_0);
  //Layer 1 - Normal Convolution - Channelwise Output
  NormalConvolution<ap_accuracy_truc_clip, dType_Reg, __FEATURES_0_0__INPUT_CHAN__, __FEATURES_0_0__INPUT_SIZE__, __FEATURES_0_0__OUTPUT_CHAN__,
                    __FEATURES_0_0__KERNEL_SIZE__, __FEATURES_0_0__STRIDE_SIZE__>(Pipe_0, Pipe_1, weights_nc, iMult_biases_acc_1, nShif_biases_acc_1,
                                                                                  downscaling_iMult_1, downscaling_nShift_1,
                                                                                  weights_zp_1, bzp_nc, opzp1, opzp2, biases_local_1,
                                                                                  __FEATURES_0_0__OUTPUT_SIZE__, __FEATURES_0_0__INPUT_SIZE__,
                                                                                  __FEATURES_0_0__OUTPUT_CHAN__, __FEATURES_0_0__INPUT_CHAN__,
                                                                                  ap_accuracy_truc_clip());

  //Layer 2 - Depthwise Convolution - Channelwise Output
  DepthWiseConvolution<ap_accuracy_truc_clip, dType_Reg, __FEATURES_1_CONV_0__INPUT_CHAN__, __FEATURES_1_CONV_0__INPUT_SIZE__,
                       __FEATURES_1_CONV_0__OUTPUT_CHAN__, __FEATURES_1_CONV_0__KERNEL_SIZE__>(Pipe_1, Pipe_2, weights_dw, iMult_biases_acc_2,
                                                                                               nShif_biases_acc_2, downscaling_iMult_2, downscaling_nShift_2,
                                                                                               weights_zp_2, bzp_dw, opzp2, opzp3, biases_local_2,
                                                                                               __FEATURES_1_CONV_0__OUTPUT_SIZE__, __FEATURES_1_CONV_0__INPUT_SIZE__,
                                                                                               __FEATURES_1_CONV_0__OUTPUT_CHAN__, __FEATURES_1_CONV_0__INPUT_CHAN__,
                                                                                               __FEATURES_1_CONV_0__STRIDE_SIZE__, ap_accuracy_truc_clip());

  //Layer 3 - Pointwise Convolution - Channelwise Output
  PointWiseConvolution<ap_accuracy_truc_clip, dType_Reg, __FEATURES_1_CONV_2__OUTPUT_CHAN__, __FEATURES_1_CONV_2__INPUT_CHAN__,
                       __FEATURES_1_CONV_2__INPUT_CHAN__, fused_pipes, dType_8u, ap_resource_dflt>(Pipe_2, Pipe_3, weights_pw, iMult_biases_acc_3, nShif_biases_acc_3,
                                                                                                   downscaling_iMult_3, downscaling_nShift_3, weights_zp_3, bzp_pw, opzp3,
                                                                                                   opzp4, biases_local_3, __FEATURES_1_CONV_2__OUTPUT_SIZE__,
                                                                                                   __FEATURES_1_CONV_2__INPUT_SIZE__, __FEATURES_1_CONV_2__OUTPUT_CHAN__,
                                                                                                   __FEATURES_1_CONV_2__INPUT_CHAN__, ap_accuracy_truc_clip(), ap_resource_dflt());

  writeData<fused_pipes, dType_8u, __FEATURES_1_CONV_2__OUTPUT_CHAN__, __FEATURES_1_CONV_2__OUTPUT_SIZE__>(out_0, Pipe_3);
  // writeData<__FEATURES_0_0__OUTPUT_CHAN__, __FEATURES_0_0__OUTPUT_SIZE__>(out_0, Pipe_1);
}
