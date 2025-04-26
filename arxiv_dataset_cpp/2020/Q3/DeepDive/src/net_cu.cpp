
#include <hw/net_cu.hpp>

#define __ALIGNED_DEPTH_WISE_WEIGHT_LENGTH__ __MAX_DW_STR1_CONV_MAX_OUTPUT_CHANNEL__ *ALIGNED_KERNEL_SIZE(3)
#define __PRJ_CORE_SIZE__ 120


void big_compute_unit(dType_8u *input_feature,
                      dType_8u *weights,
                      dType_8u *biases,
                      dType_8u *output_feature,
                      dType_8u *weight_zp,

                      dType_8u *iMult_bias_acc,
                      dType_8t *nShift_bias_acc,
                      dType_8u *iMult_output,
                      dType_8u *nShift_output,

                      dType_Reg layer1_ip_chan, dType_Reg layer1_ip_size,
                      dType_Reg layer1_op_chan, dType_Reg layer1_op_size,
                      dType_Reg layer1_k_size, dType_Reg layer1_stride, dType_Reg layer1_padding,
                      dType_Reg layer1_weight_length, dType_Reg layer1_weight_offset,
                      dType_Reg layer1_op_zp_length, dType_Reg layer1_op_zp_offset,
                      dType_Reg layer1_b_scale_length, dType_Reg layer1_b_scale_offset,
                      dType_Reg layer1_w_zp_length, dType_Reg layer1_w_zp_offset,
                      dType_Reg layer1_acc_scale_length, dType_Reg layer1_acc_scale_offset,
                      dType_Reg layer1_base_b_q_length, dType_Reg layer1_base_b_q_offset,

                      dType_Reg layer2_ip_chan, dType_Reg layer2_ip_size,
                      dType_Reg layer2_op_chan, dType_Reg layer2_op_size,
                      dType_Reg layer2_k_size, dType_Reg layer2_stride, dType_Reg layer2_padding,
                      dType_Reg layer2_weight_length, dType_Reg layer2_weight_offset,
                      dType_Reg layer2_op_zp_length, dType_Reg layer2_op_zp_offset,
                      dType_Reg layer2_b_scale_length, dType_Reg layer2_b_scale_offset,
                      dType_Reg layer2_w_zp_length, dType_Reg layer2_w_zp_offset,
                      dType_Reg layer2_acc_scale_length, dType_Reg layer2_acc_scale_offset,
                      dType_Reg layer2_base_b_q_length, dType_Reg layer2_base_b_q_offset,

                      dType_Reg layer3_ip_chan, dType_Reg layer3_ip_size,
                      dType_Reg layer3_op_chan, dType_Reg layer3_op_size,
                      dType_Reg layer3_k_size, dType_Reg layer3_stride, dType_Reg layer3_padding,
                      dType_Reg layer3_weight_length, dType_Reg layer3_weight_offset,
                      dType_Reg layer3_op_zp_length, dType_Reg layer3_op_zp_offset,
                      dType_Reg layer3_b_scale_length, dType_Reg layer3_b_scale_offset,
                      dType_Reg layer3_w_zp_length, dType_Reg layer3_w_zp_offset,
                      dType_Reg layer3_acc_scale_length, dType_Reg layer3_acc_scale_offset,
                      dType_Reg layer3_base_b_q_length, dType_Reg layer3_base_b_q_offset,

                      dType_Reg layer_no,

                      //opzp
                      dType_8u opzp1,
                      dType_8u opzp2,
                      dType_8u opzp3,
                      dType_8u opzp4,
                      dType_8u bzp_pw_ex,
                      dType_8u bzp_dw,
                      dType_8u bzp_pw_pj)
{

  dType_4u weights_PW_EXPND[__EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__][__EXP_PW_STR1_RES_CMN_CONV_MAX_INPUT_CHAN__];
  // #pragma HLS ARRAY_PARTITION variable = weights_PW_EXPND complete dim = 2
  CORE_SIZE(weights_PW_EXPND, __EXP_PW_STR1_RES_CMN_CONV_MAX_INPUT_CHAN__, 2)

  dType_4u weights_DW[__IRB_DW_CONV_STD_1_MAX_OUTPUT_CHANNEL__][ALIGNED_KERNEL_SIZE(3)];
#pragma HLS ARRAY_PARTITION variable = weights_DW complete dim = 2

  dType_4u weights_PW_PRJ[__PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__][__PRJC_PW_STR1_RES_CMN_CONV_MAX_INPUT_CHAN__];
  // #pragma HLS ARRAY_PARTITION variable = weights_PW_PRJ complete dim = 2
  CORE_SIZE(weights_PW_PRJ, __PRJ_CORE_SIZE__, 2)

  //Layer 1
  dType_4u biases_local_PW_EXPND[__EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_4u weights_zp_PW_EXPND[__EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u downscaling_iMult_PW_EXPND[__EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u downscaling_nShift_PW_EXPND[__EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8t nShif_biases_acc_PW_EXPND[__EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u iMult_biases_acc_PW_EXPND[__EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];

  //Layer 2
  dType_4u biases_local_DW[__MAX_DW_STR1_CONV_MAX_OUTPUT_CHANNEL__];
  dType_4u weights_zp_DW[__MAX_DW_STR1_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u downscaling_iMult_DW[__MAX_DW_STR1_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u downscaling_nShift_DW[__MAX_DW_STR1_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8t nShif_biases_acc_DW[__MAX_DW_STR1_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u iMult_biases_acc_DW[__MAX_DW_STR1_CONV_MAX_OUTPUT_CHANNEL__];

  //Layer 3
  dType_4u biases_local_PW_PRJ[__PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_4u weights_zp_PW_PRJ[__PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u downscaling_iMult_PW_PRJ[__PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u downscaling_nShift_PW_PRJ[__PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8t nShif_biases_acc_PW_PRJ[__PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];
  dType_8u iMult_biases_acc_PW_PRJ[__PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__];

  fused_pipes Pipe_0("BDDR");
  fused_pipes Pipe_1("BPWE");
  fused_pipes Pipe_2("BDW");
  fused_pipes Pipe_3("BPWP");

#pragma HLS DATAFLOW

  burst_read_weights<__EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__, __EXP_PW_STR1_RES_CMN_CONV_MAX_INPUT_CHAN__,
                     __IRB_DW_CONV_STD_1_MAX_OUTPUT_CHANNEL__, ALIGNED_KERNEL_SIZE(3),
                     __PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__, __PRJC_PW_STR1_RES_CMN_CONV_MAX_INPUT_CHAN__>(weights, weights_PW_EXPND, weights_DW, weights_PW_PRJ,
                                                                                                                     layer1_ip_chan, layer1_op_chan, layer1_weight_length, ALIGNED_KERNEL_SIZE(3), layer2_op_chan, layer2_weight_length, layer3_ip_chan, layer3_op_chan);

  copy_biases_body(biases, biases_local_PW_EXPND, biases_local_DW, biases_local_PW_PRJ, //biases_local_DW_ST2,
                   layer1_b_scale_length, layer1_b_scale_offset,
                   layer2_b_scale_length, layer2_b_scale_offset,
                   layer3_b_scale_length, layer3_b_scale_offset);

  copy_weight_zp_body(weight_zp, weights_zp_PW_EXPND, weights_zp_DW, weights_zp_PW_PRJ, //weights_zp_DW_ST2,
                      layer1_w_zp_length, layer1_w_zp_offset,
                      layer2_w_zp_length, layer2_w_zp_offset,
                      layer3_w_zp_length, layer3_w_zp_offset);

  copy_iMult_downscaling_body(iMult_output, downscaling_iMult_PW_EXPND, downscaling_iMult_DW, downscaling_iMult_PW_PRJ, //downscaling_iMult_DW_ST2,
                              layer1_acc_scale_length, layer1_acc_scale_offset,
                              layer2_acc_scale_length, layer2_acc_scale_offset,
                              layer3_acc_scale_length, layer3_acc_scale_offset);

  copy_nShift_downscaling_body(nShift_output, downscaling_nShift_PW_EXPND, downscaling_nShift_DW, downscaling_nShift_PW_PRJ, //downscaling_nShift_DW_ST2,
                               layer1_base_b_q_length, layer1_base_b_q_offset,
                               layer2_base_b_q_length, layer2_base_b_q_offset,
                               layer3_base_b_q_length, layer3_base_b_q_offset);

  copy_iMult_upscaling_body(iMult_bias_acc, iMult_biases_acc_PW_EXPND, iMult_biases_acc_DW, iMult_biases_acc_PW_PRJ, //iMult_biases_acc_DW_ST2,
                            layer1_b_scale_length, layer1_b_scale_offset,
                            layer2_b_scale_length, layer2_b_scale_offset,
                            layer3_b_scale_length, layer3_b_scale_offset);

  copy_nShift_upscaling_body(nShift_bias_acc, nShif_biases_acc_PW_EXPND, nShif_biases_acc_DW, nShif_biases_acc_PW_PRJ, //nShif_biases_acc_DW_ST2,
                             layer1_base_b_q_length, layer1_base_b_q_offset,
                             layer2_base_b_q_length, layer2_base_b_q_offset,
                             layer3_base_b_q_length, layer3_base_b_q_offset);

  readFeatureMap(input_feature, Pipe_0, layer1_ip_chan, layer1_ip_size);

  //Expansion - Pointwise Convolution - Channelwise Output
  PointWiseConvolution<ap_accuracy_truc_clip, dType_Reg,
                       __EXP_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__,
                       __EXP_PW_STR1_RES_CMN_CONV_MAX_INPUT_CHAN__,
                       __EXP_PW_STR1_RES_CMN_CONV_MAX_INPUT_CHAN__,
                       fused_pipes, dType_8u, ap_resource_dflt>(Pipe_0, Pipe_1, weights_PW_EXPND, iMult_biases_acc_PW_EXPND,
                                                               nShif_biases_acc_PW_EXPND, downscaling_iMult_PW_EXPND, downscaling_nShift_PW_EXPND,
                                                               weights_zp_PW_EXPND, bzp_pw_ex, opzp1, opzp2,
                                                               biases_local_PW_EXPND, layer1_op_size, layer1_ip_size,
                                                               layer1_op_chan, layer1_ip_chan, ap_accuracy_truc_clip(), ap_resource_dflt());

  //Depthwise Convolution - Channelwise Output
  DepthWiseConvolution<ap_accuracy_truc_clip, dType_Reg,
                       __IRB_DW_CONV_STD_1_MAX_OUTPUT_CHANNEL__,
                       __IRB_DW_CONV_STD_2_MAX_INPUT_SIZE__,
                       __IRB_DW_CONV_STD_1_MAX_INPUT_CHAN__,
                       __FEATURES_1_CONV_0__KERNEL_SIZE__>(Pipe_1, Pipe_2, weights_DW, iMult_biases_acc_DW,
                                                           nShif_biases_acc_DW, downscaling_iMult_DW, downscaling_nShift_DW,
                                                           weights_zp_DW, bzp_dw, opzp2, opzp3, biases_local_DW, layer2_op_size,
                                                           layer2_ip_size, layer2_op_chan, layer2_ip_chan, layer2_stride,
                                                           ap_accuracy_truc_clip());

  //Expansion - Pointwise Convolution - Channelwise Output
  PointWiseConvolution<ap_accuracy_round_clip, dType_32f,
                       __PRJC_PW_STR1_RES_CMN_CONV_MAX_OUTPUT_CHANNEL__,
                       __PRJC_PW_STR1_RES_CMN_CONV_MAX_INPUT_CHAN__,
                       __PRJ_CORE_SIZE__,
                       fused_pipes, dType_8u, ap_resource_dflt>(Pipe_2, Pipe_3, weights_PW_PRJ, iMult_biases_acc_PW_PRJ,
                                                               nShif_biases_acc_PW_PRJ, downscaling_iMult_PW_PRJ, downscaling_nShift_PW_PRJ,
                                                               weights_zp_PW_PRJ, bzp_pw_pj, opzp3, opzp4,
                                                               biases_local_PW_PRJ, layer3_op_size, layer3_ip_size,
                                                               layer3_op_chan, layer3_ip_chan, ap_accuracy_round_clip(), ap_resource_dflt());

  writeFeatureMap<fused_pipes, dType_8u,
                  __SCRATCH_PAD_SIZE__>(output_feature, Pipe_3, layer3_op_chan, layer3_op_size);
  // writeFeatureMap(output_feature, Pipe_7, layer2_op_chan, layer2_op_size);
}
