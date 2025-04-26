/*
Copyright (c) 2019, University of North Carolina at Charlotte All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Reza Baharani    - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
Steven Furgurson - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
Kaustubh Mhatre (^_^) - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
Ushma Bharucha - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte

This file has been generated automatically. DO NOT MODIFY IT.

*/

#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <string>

#include <global/net_headers.hpp>
#include <utils/npy.hpp>
#include <utils/aux.hpp>
#include <utils/check_results.hpp>

#include <hw/net_head.hpp>
#include <hw/net_cu.hpp>
#include <hw/net_tail.hpp>
#include <hw/net_linear.hpp>
#include <hw/QVector_Add.hpp>
#include <hw/DataCopy_Compute_skipline.hpp>

#ifndef __FIXED_INPUT__
#if __SDSCC__
#undef __ARM_NEON__
#undef __ARM_NEON
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#define __ARM_NEON__
#define __ARM_NEON
#else
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/core/version.hpp>
// #include <opencv2/video/video.hpp>
#endif
#endif

#ifdef __HW__
#include "utils/sds_utils.h"
#define generic_alloc sds_alloc
#define generic_free sds_free
#else
#define generic_alloc malloc
#define generic_free free
#endif

//#define __HW__
int main(int argc, char **argv)
{

#ifndef __FIXED_INPUT__
	if (argc != 2)
	{
		std::cout << "x> Usage: MobileNetV2 ImageToClassify" << std::endl;
		return EXIT_FAILURE;
	}
	// std::cout << cv::getBuildInformation() << std::endl;

	std::cout << "-> OpenCV version: "
			  << CV_MAJOR_VERSION << "."
			  << CV_MINOR_VERSION << "."
			  << CV_SUBMINOR_VERSION
			  << std::endl;
#endif

	std::vector<std::string> classes;

	// Get the contents of file in a vector

	if (!(read_imagenet_classes("./data/imagenet1000_labels.txt", classes)))
	{
		return EXIT_FAILURE;
	}

	// Allocate Memory in Host (Image, Weights and Output)
	// size_t image_size = sizeof(dType_8u) * __FEATURES_0_0__INPUT_CHAN__ * __FEATURES_0_0__INPUT_SIZE__ * __FEATURES_0_0__INPUT_SIZE__;
	size_t image_size = sizeof(dType_8u) * __FEATURES_2_CONV_0__INPUT_CHAN__ * __FEATURES_2_CONV_0__INPUT_SIZE__ * __FEATURES_2_CONV_0__INPUT_SIZE__;

	size_t net_biases_size = sizeof(dType_8u) * __TOTAL_BIASES_LENGTH__;
	size_t net_biases_zps = sizeof(dType_8u) * __TOTAL_BIAS_ZERO_POINTS_LENGTH__;
	size_t net_output_zps = sizeof(dType_8u) * __TOTAL_OUTPUT_ZERO_POINTS_LENGTH__;
	size_t output_size = sizeof(dType_8u) * (__SCRATCH_PAD_SIZE__);
	size_t net_weight_size_head = sizeof(dType_8u) * __TOTAL_WIGHT_LENGTH_BIT_8__;
	size_t net_weight_size_expnd = sizeof(dType_8u) * (__TOTAL_WIGHT_LENGTH_BIT_8__ + __TOTAL_WIGHT_LENGTH_BIT_4__);
	size_t net_weight_size_dw = sizeof(dType_8u) * __TOTAL_WIGHT_LENGTH_BIT_4__;
	size_t net_weight_size_proj = sizeof(dType_8u) * __TOTAL_WIGHT_LENGTH_BIT_4__;
	size_t net_weight_classifier_size = sizeof(dType_8u) * __CLASSIFIER__WEIGHT_LENGTH__;
	size_t skiplineSize = sizeof(dType_8u) * __TOTAL_RES_PATH_LENGTH__;
	size_t skip_zp_size = sizeof(dType_8u) * __TOTAL_RES_PATH_ZERO_POINT_LENGTH__;
	std::cout << "-> Allocating memory..." << std::endl;

	//This is feature buffer.
	dType_8u *image_0 = (dType_8u *)generic_alloc(image_size);
	//This is weight unq val for all worker.
	dType_8u *net_biases_0 = (dType_8u *)generic_alloc(net_biases_size);
	dType_8u *weight_zeropoint = (dType_8u *)generic_alloc(net_biases_size);
	dType_8u *bias_zeropoint = (dType_8u *)generic_alloc(net_biases_zps);
	dType_8u *output_zeropoint = (dType_8u *)generic_alloc(net_output_zps);
	dType_8u *iMult_bias_acc = (dType_8u *)generic_alloc(net_biases_size);
	dType_8t *nShift_bias_acc = (dType_8t *)generic_alloc(net_biases_size);
	dType_8u *iMult_output = (dType_8u *)generic_alloc(net_biases_size);
	dType_8u *nShift_output = (dType_8u *)generic_alloc(net_biases_size);
	dType_8u *skipline_param = (dType_8u *)generic_alloc(skiplineSize);
	dType_8u *skipline_zp = (dType_8u *)generic_alloc(skip_zp_size);

	//This is output per worker.
	dType_8u *output_0 = (dType_8u *)generic_alloc(output_size);
	dType_8u *out_buffer = (dType_8u *)generic_alloc(output_size);
	dType_8u *skip_line = (dType_8u *)generic_alloc(output_size);
	//dType_16t *b16t_out_buffer = (dType_16t *)generic_alloc(output_size);
	//This is weight idx ptr per worker.
	dType_8u *net_weight = (dType_8u *)generic_alloc(net_weight_size_expnd);

	//Buffer to coppy data for skipline
	//dType_8u *data_skipline = (dType_8u *)generic_alloc(output_size);
	//	dType_8u *colWise = (dType_8u *)generic_alloc(output_size);
	dType_8u *net_weight_classifier = (dType_8u *)generic_alloc(net_weight_classifier_size);

	if (IS_MEM_NOT_ALLOCATED)
	{
		std::cout << "x> Failed to allocate memory" << std::endl;
		return EXIT_FAILURE;
	}

	std::vector<dType_8u> net_classifier_weight_vect;

	//net biases
	std::vector<dType_8u> net_biases_vect;

	//Input_0 scale
	std::vector<float> input_0_scale;
	std::vector<dType_8u> input_0_zp;

	//weight scale
	std::vector<dType_8u> weights_zp;
	//std::vector<dType_8u> net_weight_vect;
	std::vector<dType_8u> net_weight_vect_4bit;
	std::vector<dType_8u> net_weight_vect_8bit;

	//biases scale
	std::vector<dType_8u> biases_zp;

	//Accu scale
	std::vector<dType_8u> accum_iMult;
	std::vector<dType_8t> accum_nShift;

	//output scale
	std::vector<dType_8u> output_zp;
	std::vector<float> output_0_scale;

	//base_accu scale
	std::vector<dType_8u> iMult_biases_acc;
	std::vector<dType_8t> nShif_biases_acc;

	//Skipline param
	std::vector<dType_8u> skipline_para;
	std::vector<dType_8u> skipline_zeropoints;

	std::vector<unsigned long> biases_npy_size;
	std::vector<unsigned long> classifier_weight_size;
	std::vector<unsigned long> input_0_scale_size;
	std::vector<unsigned long> input_0_zp_size;
	std::vector<unsigned long> weights_zp_size;
	std::vector<unsigned long> biases_zp_size;
	std::vector<unsigned long> accum_iMult_size;
	std::vector<unsigned long> accum_nShift_size;
	std::vector<unsigned long> output_zp_size;
	std::vector<unsigned long> output_scale_size;
	std::vector<unsigned long> iMult_biases_acc_size;
	std::vector<unsigned long> nShif_biases_acc_size;
	//std::vector<unsigned long> weight_size;
	std::vector<unsigned long> weight_size_4bit;
	std::vector<unsigned long> weight_size_8bit;
	std::vector<unsigned long> skipline_para_size;
	std::vector<unsigned long> skipline_zps_size;

	std::cout << "-> Reading network parameters..." << std::endl;
	//npy::LoadArrayFromNumpy("./data/weights.npy", weight_size, net_weight_vect);
	npy::LoadArrayFromNumpy("./data/weights_bit_4.npy", weight_size_4bit, net_weight_vect_4bit);
	npy::LoadArrayFromNumpy("./data/weights_bit_8.npy", weight_size_8bit, net_weight_vect_8bit);
	npy::LoadArrayFromNumpy("./data/biases.npy", biases_npy_size, net_biases_vect);
	npy::LoadArrayFromNumpy("./data/inputs_0_scale.npy", input_0_scale_size, input_0_scale);
	npy::LoadArrayFromNumpy("./data/inputs_0_zp.npy", input_0_zp_size, input_0_zp);
	npy::LoadArrayFromNumpy("./data/weights_zp.npy", weights_zp_size, weights_zp);
	npy::LoadArrayFromNumpy("./data/biases_zp.npy", biases_zp_size, biases_zp);
	npy::LoadArrayFromNumpy("./data/downScaling_iMult.npy", accum_iMult_size, accum_iMult);
	npy::LoadArrayFromNumpy("./data/downScaling_nShift.npy", accum_nShift_size, accum_nShift);
	npy::LoadArrayFromNumpy("./data/output_zp.npy", output_zp_size, output_zp);
	npy::LoadArrayFromNumpy("./data/output_scale.npy", output_scale_size, output_0_scale);
	npy::LoadArrayFromNumpy("./data/nShif_biases_acc.npy", nShif_biases_acc_size, nShif_biases_acc);
	npy::LoadArrayFromNumpy("./data/iMult_biases_acc.npy", iMult_biases_acc_size, iMult_biases_acc);
	npy::LoadArrayFromNumpy("./data/weights_classifier.npy", classifier_weight_size, net_classifier_weight_vect);
	npy::LoadArrayFromNumpy("./data/skiplineparam.npy", skipline_para_size, skipline_para);
	npy::LoadArrayFromNumpy("./data/skiplineparamipzpopzp.npy", skipline_zps_size, skipline_zeropoints);

#ifdef __FIXED_INPUT__
	std::cout << "!> Reading from fixed input file." << std::endl;
	std::vector<unsigned long> norm_input_size;
	std::vector<float> input_norm_vect;
	// npy::LoadArrayFromNumpy("./data/norm_image.npy", norm_input_size, input_norm_vect);
	// npy::LoadArrayFromNumpy("./data/norm_img_channelWise.npy", norm_input_size, input_norm_vect);
	npy::LoadArrayFromNumpy("./data/image160_quant_flat_channel.npy", norm_input_size, input_norm_vect);
	// npy::LoadArrayFromNumpy("./data/features.16.res_path.npy", norm_input_size, input_norm_vect);
// #else
#endif

	std::cout << "-> Checking the data size consistency..." << std::endl;
	assert(weight_size_8bit[0] == __TOTAL_WIGHT_LENGTH_BIT_8__);
	assert(weight_size_4bit[0] == __TOTAL_WIGHT_LENGTH_BIT_4__);
	assert(classifier_weight_size[0] == __CLASSIFIER__WEIGHT_LENGTH__);
	assert(biases_npy_size[0] == __TOTAL_BIASES_LENGTH__);
	assert(input_0_scale_size[0] == __INPUT_SCALE_ZP_LENGTH__);
	assert(input_0_zp_size[0] == __INPUT_SCALE_ZP_LENGTH__);
	assert(weights_zp_size[0] == __TOTAL_WEIGHT_ZERO_POINTS_LENGTH__);
	assert(biases_zp_size[0] == __TOTAL_BIAS_ZERO_POINTS_LENGTH__);
	assert(accum_iMult_size[0] == __TOTAL_iMULT_UPSCALING_LENGTH__);
	assert(accum_nShift_size[0] == __TOTAL_nSHIFT_UPSCALING_LENGTH__);
	assert(output_zp_size[0] == __TOTAL_OUTPUT_ZERO_POINTS_LENGTH__);
	assert(output_scale_size[0] == __TOTAL_OUTPUT_SCALING_LENGTH__);
	assert(nShif_biases_acc_size[0] == __TOTAL_nSHIFT_DOWNSCALING_LENGTH__);
	assert(iMult_biases_acc_size[0] == __TOTAL_iMULT_DOWNSCALING_LENGTH__);
	assert(input_0_zp_size[0] == __INPUT_SCALE_ZP_LENGTH__);
	assert(skipline_para_size[0] == __TOTAL_RES_PATH_LENGTH__);
	assert(skipline_zps_size[0] == __TOTAL_RES_PATH_ZERO_POINT_LENGTH__);

	for (int idx = 0; idx < __TOTAL_BIASES_LENGTH__; idx++)
	{
		net_biases_0[idx] = net_biases_vect[idx];
	}

	for (int idx = 0; idx < __TOTAL_WEIGHT_ZERO_POINTS_LENGTH__; idx++)
	{
		weight_zeropoint[idx] = weights_zp[idx];
	}

	for (int idx = 0; idx < __TOTAL_BIAS_ZERO_POINTS_LENGTH__; idx++)
	{
		bias_zeropoint[idx] = biases_zp[idx];
	}

	for (int idx = 0; idx < __TOTAL_OUTPUT_ZERO_POINTS_LENGTH__; idx++)
	{
		output_zeropoint[idx] = output_zp[idx];
	}

	for (int idx = 0; idx < __TOTAL_iMULT_UPSCALING_LENGTH__; idx++)
	{
		iMult_bias_acc[idx] = iMult_biases_acc[idx];
	}

	for (int idx = 0; idx < __TOTAL_nSHIFT_UPSCALING_LENGTH__; idx++)
	{
		nShift_bias_acc[idx] = nShif_biases_acc[idx];
	}

	for (int idx = 0; idx < __TOTAL_iMULT_DOWNSCALING_LENGTH__; idx++)
	{
		iMult_output[idx] = accum_iMult[idx];
	}

	for (int idx = 0; idx < __TOTAL_nSHIFT_DOWNSCALING_LENGTH__; idx++)
	{
		nShift_output[idx] = accum_nShift[idx];
	}

	for (int idx = 0; idx < __TOTAL_RES_PATH_LENGTH__; idx++)
	{
		skipline_param[idx] = skipline_para[idx];
	}

	for (int idx = 0; idx < __TOTAL_RES_PATH_ZERO_POINT_LENGTH__; idx++)
	{
		skipline_zp[idx] = skipline_zeropoints[idx];
	}

	std::cout << "-> Transfering net. params. into FPGA..." << std::endl;
#ifdef __HW__
	sds_utils::perf_counter compute_Total;
	sds_utils::perf_counter compute_head_0;
	sds_utils::perf_counter compute_body_1;
	sds_utils::perf_counter compute_body_2;
	sds_utils::perf_counter compute_body_3;
	sds_utils::perf_counter compute_body_4;
	sds_utils::perf_counter compute_body_5;
	sds_utils::perf_counter compute_body_6;
	sds_utils::perf_counter compute_body_7;
	sds_utils::perf_counter compute_body_8;
	sds_utils::perf_counter compute_body_9;
	sds_utils::perf_counter compute_body_10;
	sds_utils::perf_counter compute_body_11;
	sds_utils::perf_counter compute_body_12;
	sds_utils::perf_counter compute_body_13;
	sds_utils::perf_counter compute_body_14;
	sds_utils::perf_counter compute_body_15;
	sds_utils::perf_counter compute_body_16;
	sds_utils::perf_counter compute_tail_cnt;
	sds_utils::perf_counter compute_linear_cnt;
	std::cout << "-> Starting computation..." << std::endl;
	// compute_head_0.start();
#endif

	//Read Camera input
	// cv::VideoCapture capture(0);

	//Static video input
	cv::VideoCapture capture(0);

	if (!capture.isOpened())
	{
		std::cout << "Error opening VideoCapture." << std::endl;
		return -1;
	}

	// std::cout << "Frame width: " << capture.get(cv::CV_CAP_PROP_FRAME_WIDTH) << std::endl;
	// std::cout << "     height: " << capture.get(cv::CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
	// std::cout << "Capturing FPS: " << capture.get(CV_CAP_PROP_FPS) << std::endl;

	// std::cout << "-> Openning image file..." << std::endl;
	cv::Mat src, dst_rs, dst_rgb, disp_rgb;

	// capture.read(frame_img);
	// cv::imshow("Video", frame_img);

	for (;;)
	{
		capture.read(src);
		if (src.empty())
		{
			std::cout << "ERROR: Can't grab camera frame." << std::endl;
			break;
			//return -1;
		}

		// src = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR); // Read the file

		// if (!src.data) // Check for invalid input
		// {
		// 	std::cout << "Could not open or find the image" << std::endl;
		// 	return EXIT_FAILURE;
		// }
		cv::Size size(__FEATURES_0_0__INPUT_SIZE__, __FEATURES_0_0__INPUT_SIZE__); //the dst image size,e.g.192x192
		std::cout << "-> Resizing image file..." << std::endl;
		std::cout << "-> Showing the input image." << std::endl;
		cv::namedWindow("TeCSAR-DeepDive-Input", cv::WINDOW_AUTOSIZE);
		cv::imshow("TeCSAR-DeepDive-Input", src);
		// cvtColor(src, disp_rgb, cv::COLOR_RGB2BGR);
		resize(src, dst_rs, size); //resize image
		cvtColor(dst_rs, dst_rgb, cv::COLOR_BGR2RGB);

		
		std::cout << "-> Normalizing the image..." << std::endl;
		float mean[3] = {0.485, 0.456, 0.406};
		float std[3] = {0.229, 0.224, 0.225};
		int counter = 0;

		for (int i = 0; i < __FEATURES_0_0__INPUT_SIZE__; i++)
		{
			for (int j = 0; j < __FEATURES_0_0__INPUT_SIZE__; j++)
			{
				for (int c = 0; c < __FEATURES_0_0__INPUT_CHAN__; c++)
				{
					float norm_input = ((dst_rgb.at<cv::Vec3b>(i, j)[c] / 255.0) - mean[c]) / std[c];
					norm_input = (norm_input * input_0_scale[__CONV_0__INPUT_SCALE_ZP_OFFSET__]) + input_0_zp[__CONV_0__INPUT_SCALE_ZP_OFFSET__];
					dType_8uf tmp = norm_input;
					image_0[counter++] = tmp;
					//image_0[(c * (__FEATURES_0_0__INPUT_SIZE__ * __FEATURES_0_0__INPUT_SIZE__) + (i * __FEATURES_0_0__INPUT_SIZE__) + j)] = 21;
				}
			}
		}

		// cv::namedWindow("TeCSAR-DeepDive", cv::WINDOW_AUTOSIZE);
		// cv::imshow("TeCSAR-DeepDive", dst_rs);

		// cv::waitKey(0);
	// }
	
		int count = 0;
		int total_weight_length = __FEATURES_1_CONV_0__WEIGHT_LENGTH__ + __FEATURES_1_CONV_2__WEIGHT_LENGTH__;
		for (int idx = 0; idx < __TOTAL_WIGHT_LENGTH_BIT_8__; idx++)
		{
			net_weight[count] = net_weight_vect_8bit[idx + __FEATURES_0_0__WEIGHT_OFFSET__];
			count++;
		}
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[count] = net_weight_vect_4bit[idx + __FEATURES_1_CONV_0__WEIGHT_OFFSET__];
			count++;
		}

		//Call to Compute Head
		int layerNo = 0;
		int useSkipLine;
		//opzp
		dType_8uf opzp1;
		dType_8uf opzp2;
		dType_8uf opzp3;
		dType_8uf opzp4;

		dType_8uf bzp_nc;
		dType_8uf bzp_pw_ex;
		dType_8uf bzp_dw;
		dType_8uf bzp_pw_pj;
		dType_Reg skipLineOffset = 0;

		//skipline zero points
		dType_Reg skipline_ipzp1;
		dType_Reg skipline_ipzp2;
		dType_Reg skipline_opzp1;

		//Skipline Data
		dType_Reg copy_data_length;
		dType_Reg output_length;

		//opzp
		opzp1 = 114;
		opzp2 = output_zeropoint[0]; //0;
		opzp3 = output_zeropoint[1]; //0;
		opzp4 = output_zeropoint[2]; //126;
		bzp_nc = bias_zeropoint[0];
		bzp_dw = bias_zeropoint[1];
		bzp_pw_pj = bias_zeropoint[2];
		// useSkipLine = false;
		skipLineOffset = 0;

#ifndef __RELEASE__
		std::cout << "-> Head call." << std::endl;
#endif

#ifdef __HW__
		compute_head_0.start();
		compute_Total.start();
#endif
		compute_head(image_0,
					 net_weight,
					 net_biases_0,
					 output_0,

					 weight_zeropoint,
					 iMult_bias_acc,
					 nShift_bias_acc,
					 iMult_output,
					 nShift_output,

					 opzp1,
					 opzp2,
					 opzp3,
					 opzp4,
					 bzp_nc,
					 bzp_dw,
					 bzp_pw_pj);

#ifdef __HW__
		compute_head_0.stop();
#endif

		//Body 1 - Weights
		count = 0;
		total_weight_length = __FEATURES_2_CONV_0__WEIGHT_LENGTH__ + __FEATURES_2_CONV_2__WEIGHT_LENGTH__ + __FEATURES_2_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[count] = net_weight_vect_4bit[idx + __FEATURES_2_CONV_0__WEIGHT_OFFSET__];
			count++;
		}

		//Call to Compute Body 1
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[2];
		opzp2 = output_zeropoint[3];
		opzp3 = output_zeropoint[4];
		opzp4 = output_zeropoint[5];

		bzp_pw_ex = bias_zeropoint[3];
		bzp_dw = bias_zeropoint[4];
		bzp_pw_pj = bias_zeropoint[5];
		// useSkipLine = false;
		skipLineOffset = 0;
#ifdef __HW__
		compute_body_1.start();
#endif

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_head.npy", __FEATURES_1_CONV_2__OUTPUT_CHAN__, __FEATURES_1_CONV_2__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_head.npy", "Head", __FEATURES_1_CONV_2__OUTPUT_CHAN__, __FEATURES_1_CONV_2__OUTPUT_SIZE__);

#endif
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 1 call." << std::endl;
#endif

		big_compute_unit(output_0,
						 net_weight,
						 net_biases_0,
						 out_buffer,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_2_CONV_0__INPUT_CHAN__, __FEATURES_2_CONV_0__INPUT_SIZE__,
						 __FEATURES_2_CONV_0__OUTPUT_CHAN__, __FEATURES_2_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_2_CONV_0__KERNEL_SIZE__, __FEATURES_2_CONV_0__STRIDE_SIZE__, __FEATURES_2_CONV_0__PADDING_SIZE__,
						 __FEATURES_2_CONV_0__WEIGHT_LENGTH__, __FEATURES_2_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_2_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_2_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_2_CONV_0_B_SCALE_LENGTH__, __FEATURES_2_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_2_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_2_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_2_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_2_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_2_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_2_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_2_CONV_2__INPUT_CHAN__, __FEATURES_2_CONV_2__INPUT_SIZE__,
						 __FEATURES_2_CONV_2__OUTPUT_CHAN__, __FEATURES_2_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_2_CONV_2__KERNEL_SIZE__, __FEATURES_2_CONV_2__STRIDE_SIZE__, __FEATURES_2_CONV_2__PADDING_SIZE__,
						 __FEATURES_2_CONV_2__WEIGHT_LENGTH__, __FEATURES_2_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_2_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_2_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_2_CONV_2_B_SCALE_LENGTH__, __FEATURES_2_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_2_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_2_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_2_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_2_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_2_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_2_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_2_CONV_4__INPUT_CHAN__, __FEATURES_2_CONV_4__INPUT_SIZE__,
						 __FEATURES_2_CONV_4__OUTPUT_CHAN__, __FEATURES_2_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_2_CONV_4__KERNEL_SIZE__, __FEATURES_2_CONV_4__STRIDE_SIZE__, __FEATURES_2_CONV_4__PADDING_SIZE__,
						 __FEATURES_2_CONV_4__WEIGHT_LENGTH__, __FEATURES_2_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_2_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_2_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_2_CONV_4_B_SCALE_LENGTH__, __FEATURES_2_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_2_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_2_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_2_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_2_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_2_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_2_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_body1.npy", __FEATURES_2_CONV_4__OUTPUT_CHAN__, __FEATURES_2_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(out_buffer, "./data/out_body1.npy", "Body1", __FEATURES_2_CONV_4__OUTPUT_CHAN__, __FEATURES_2_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_1.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 2 call." << std::endl;
#endif

		//Body 2 - Weights

		total_weight_length = __FEATURES_3_CONV_0__WEIGHT_LENGTH__ + __FEATURES_3_CONV_2__WEIGHT_LENGTH__ + __FEATURES_3_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_3_CONV_0__WEIGHT_OFFSET__];
		}

		//skipline zps
		skipline_ipzp1 = skipline_zp[0];
		skipline_ipzp2 = skipline_zp[1];
		skipline_opzp1 = output_zeropoint[6];

		//Call to Compute Body 2
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[5];
		opzp2 = output_zeropoint[7];
		opzp3 = output_zeropoint[8];
		opzp4 = output_zeropoint[9];
		useSkipLine = true;
		skipLineOffset = 0;
		bzp_pw_ex = bias_zeropoint[6];
		bzp_dw = bias_zeropoint[7];
		bzp_pw_pj = bias_zeropoint[8];

		//copy_data_length = __FEATURES_3_CONV_0__INPUT_CHAN__ * __FEATURES_3_CONV_0__INPUT_SIZE__ * __FEATURES_3_CONV_0__INPUT_SIZE__;
		//copyDataSkipline(out_buffer, data_skipline, copy_data_length);
		// memcpy(data_skipline,output_0,copy_data_length);

#ifdef __HW__
		compute_body_2.start();
#endif
		big_compute_unit(out_buffer,
						 net_weight,
						 net_biases_0,
						 output_0,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_3_CONV_0__INPUT_CHAN__, __FEATURES_3_CONV_0__INPUT_SIZE__,
						 __FEATURES_3_CONV_0__OUTPUT_CHAN__, __FEATURES_3_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_3_CONV_0__KERNEL_SIZE__, __FEATURES_3_CONV_0__STRIDE_SIZE__, __FEATURES_3_CONV_0__PADDING_SIZE__,
						 __FEATURES_3_CONV_0__WEIGHT_LENGTH__, __FEATURES_3_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_3_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_3_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_3_CONV_0_B_SCALE_LENGTH__, __FEATURES_3_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_3_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_3_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_3_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_3_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_3_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_3_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_3_CONV_2__INPUT_CHAN__, __FEATURES_3_CONV_2__INPUT_SIZE__,
						 __FEATURES_3_CONV_2__OUTPUT_CHAN__, __FEATURES_3_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_3_CONV_2__KERNEL_SIZE__, __FEATURES_3_CONV_2__STRIDE_SIZE__, __FEATURES_3_CONV_2__PADDING_SIZE__,
						 __FEATURES_3_CONV_2__WEIGHT_LENGTH__, __FEATURES_3_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_3_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_3_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_3_CONV_2_B_SCALE_LENGTH__, __FEATURES_3_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_3_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_3_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_3_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_3_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_3_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_3_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_3_CONV_4__INPUT_CHAN__, __FEATURES_3_CONV_4__INPUT_SIZE__,
						 __FEATURES_3_CONV_4__OUTPUT_CHAN__, __FEATURES_3_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_3_CONV_4__KERNEL_SIZE__, __FEATURES_3_CONV_4__STRIDE_SIZE__, __FEATURES_3_CONV_4__PADDING_SIZE__,
						 __FEATURES_3_CONV_4__WEIGHT_LENGTH__, __FEATURES_3_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_3_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_3_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_3_CONV_4_B_SCALE_LENGTH__, __FEATURES_3_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_3_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_3_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_3_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_3_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_3_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_3_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_body2.npy", __FEATURES_3_CONV_4__OUTPUT_CHAN__, __FEATURES_3_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_body2.npy", "Body2", __FEATURES_3_CONV_4__OUTPUT_CHAN__, __FEATURES_3_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_2.stop();
#endif

		output_length = __FEATURES_3_CONV_4__OUTPUT_CHAN__ * __FEATURES_3_CONV_4__OUTPUT_SIZE__ * __FEATURES_3_CONV_4__OUTPUT_SIZE__;

		dType_Reg skiplinelayerParamOffset = (skipLineOffset << 2);
		dType_8u xMultiplier = skipline_param[skiplinelayerParamOffset];
		dType_8u xShift = skipline_param[skiplinelayerParamOffset + 1];
		dType_8u yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		dType_8u yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(output_0,
					out_buffer,
					skip_line,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(output_0,
					out_buffer,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/

#ifndef __RELEASE__
		std::cout << "-> Body 3 call." << std::endl;
#endif

		//Body 3 - Weights

		total_weight_length = __FEATURES_4_CONV_0__WEIGHT_LENGTH__ + __FEATURES_4_CONV_2__WEIGHT_LENGTH__ + __FEATURES_4_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_4_CONV_0__WEIGHT_OFFSET__];
		}

		//Call to Compute Body 3
		layerNo = layerNo + 1;
		opzp1 = skipline_zp[0]; //output_zeropoint[9];
		opzp2 = output_zeropoint[10];
		opzp3 = output_zeropoint[11];
		opzp4 = output_zeropoint[12];
		bzp_pw_ex = bias_zeropoint[9];
		bzp_dw = bias_zeropoint[10];
		bzp_pw_pj = bias_zeropoint[11];
		useSkipLine = false;
#ifdef __HW__
		compute_body_3.start();
#endif
		big_compute_unit(skip_line,
						 net_weight,
						 net_biases_0,
						 out_buffer,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_4_CONV_0__INPUT_CHAN__, __FEATURES_4_CONV_0__INPUT_SIZE__,
						 __FEATURES_4_CONV_0__OUTPUT_CHAN__, __FEATURES_4_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_4_CONV_0__KERNEL_SIZE__, __FEATURES_4_CONV_0__STRIDE_SIZE__, __FEATURES_4_CONV_0__PADDING_SIZE__,
						 __FEATURES_4_CONV_0__WEIGHT_LENGTH__, __FEATURES_4_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_4_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_4_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_4_CONV_0_B_SCALE_LENGTH__, __FEATURES_4_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_4_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_4_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_4_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_4_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_4_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_4_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_4_CONV_2__INPUT_CHAN__, __FEATURES_4_CONV_2__INPUT_SIZE__,
						 __FEATURES_4_CONV_2__OUTPUT_CHAN__, __FEATURES_4_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_4_CONV_2__KERNEL_SIZE__, __FEATURES_4_CONV_2__STRIDE_SIZE__, __FEATURES_4_CONV_2__PADDING_SIZE__,
						 __FEATURES_4_CONV_2__WEIGHT_LENGTH__, __FEATURES_4_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_4_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_4_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_4_CONV_2_B_SCALE_LENGTH__, __FEATURES_4_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_4_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_4_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_4_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_4_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_4_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_4_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_4_CONV_4__INPUT_CHAN__, __FEATURES_4_CONV_4__INPUT_SIZE__,
						 __FEATURES_4_CONV_4__OUTPUT_CHAN__, __FEATURES_4_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_4_CONV_4__KERNEL_SIZE__, __FEATURES_4_CONV_4__STRIDE_SIZE__, __FEATURES_4_CONV_4__PADDING_SIZE__,
						 __FEATURES_4_CONV_4__WEIGHT_LENGTH__, __FEATURES_4_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_4_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_4_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_4_CONV_4_B_SCALE_LENGTH__, __FEATURES_4_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_4_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_4_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_4_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_4_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_4_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_4_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_body3.npy", __FEATURES_4_CONV_4__OUTPUT_CHAN__, __FEATURES_4_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(out_buffer, "./data/out_body3.npy", "Body3", __FEATURES_4_CONV_4__OUTPUT_CHAN__, __FEATURES_4_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_3.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 4 call." << std::endl;
#endif

		//Body 4 - Weights

		total_weight_length = __FEATURES_5_CONV_0__WEIGHT_LENGTH__ + __FEATURES_5_CONV_2__WEIGHT_LENGTH__ + __FEATURES_5_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_5_CONV_0__WEIGHT_OFFSET__];
		}

		// for (int idx = 0; idx < __FEATURES_5_CONV_2__WEIGHT_LENGTH__; idx++)
		// {
		// 	net_weight_dw[idx] = net_weight_vect_4bit[idx + __FEATURES_5_CONV_2__WEIGHT_OFFSET__];
		// }

		// for (int idx = 0; idx < __FEATURES_5_CONV_4__WEIGHT_LENGTH__; idx++)
		// {
		// 	net_weight_proj[idx] = net_weight_vect_4bit[idx + __FEATURES_5_CONV_4__WEIGHT_OFFSET__];
		// }

		//skipline zps
		skipline_ipzp1 = skipline_zp[2];
		skipline_ipzp2 = skipline_zp[3];
		skipline_opzp1 = output_zeropoint[13];

		//Call to Compute Body 4
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[12];
		opzp2 = output_zeropoint[14];
		opzp3 = output_zeropoint[15];
		opzp4 = output_zeropoint[16];
		useSkipLine = true;
		skipLineOffset = 1;
		bzp_pw_ex = bias_zeropoint[12];
		bzp_dw = bias_zeropoint[13];
		bzp_pw_pj = bias_zeropoint[14];

		// copy_data_length = __FEATURES_5_CONV_0__INPUT_CHAN__ * __FEATURES_5_CONV_0__INPUT_SIZE__ * __FEATURES_5_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_4.start();
#endif
		big_compute_unit(out_buffer,
						 net_weight,
						 net_biases_0,
						 output_0,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_5_CONV_0__INPUT_CHAN__, __FEATURES_5_CONV_0__INPUT_SIZE__,
						 __FEATURES_5_CONV_0__OUTPUT_CHAN__, __FEATURES_5_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_5_CONV_0__KERNEL_SIZE__, __FEATURES_5_CONV_0__STRIDE_SIZE__, __FEATURES_5_CONV_0__PADDING_SIZE__,
						 __FEATURES_5_CONV_0__WEIGHT_LENGTH__, __FEATURES_5_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_5_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_5_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_5_CONV_0_B_SCALE_LENGTH__, __FEATURES_5_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_5_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_5_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_5_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_5_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_5_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_5_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_5_CONV_2__INPUT_CHAN__, __FEATURES_5_CONV_2__INPUT_SIZE__,
						 __FEATURES_5_CONV_2__OUTPUT_CHAN__, __FEATURES_5_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_5_CONV_2__KERNEL_SIZE__, __FEATURES_5_CONV_2__STRIDE_SIZE__, __FEATURES_5_CONV_2__PADDING_SIZE__,
						 __FEATURES_5_CONV_2__WEIGHT_LENGTH__, __FEATURES_5_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_5_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_5_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_5_CONV_2_B_SCALE_LENGTH__, __FEATURES_5_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_5_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_5_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_5_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_5_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_5_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_5_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_5_CONV_4__INPUT_CHAN__, __FEATURES_5_CONV_4__INPUT_SIZE__,
						 __FEATURES_5_CONV_4__OUTPUT_CHAN__, __FEATURES_5_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_5_CONV_4__KERNEL_SIZE__, __FEATURES_5_CONV_4__STRIDE_SIZE__, __FEATURES_5_CONV_4__PADDING_SIZE__,
						 __FEATURES_5_CONV_4__WEIGHT_LENGTH__, __FEATURES_5_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_5_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_5_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_5_CONV_4_B_SCALE_LENGTH__, __FEATURES_5_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_5_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_5_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_5_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_5_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_5_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_5_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);
#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_body4.npy", __FEATURES_5_CONV_4__OUTPUT_CHAN__, __FEATURES_5_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_body4.npy", "Body4", __FEATURES_5_CONV_4__OUTPUT_CHAN__, __FEATURES_5_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_4.stop();
#endif

		output_length = __FEATURES_5_CONV_4__OUTPUT_CHAN__ * __FEATURES_5_CONV_4__OUTPUT_SIZE__ * __FEATURES_5_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(output_0,
					out_buffer,
					skip_line,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(output_0, out_buffer,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/

#ifndef __RELEASE__
		std::cout << "-> Body 5 call." << std::endl;
#endif

		//Body 5 - Weights

		total_weight_length = __FEATURES_6_CONV_0__WEIGHT_LENGTH__ + __FEATURES_6_CONV_2__WEIGHT_LENGTH__ + __FEATURES_6_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_6_CONV_0__WEIGHT_OFFSET__];
		}

		//skipline zps
		skipline_ipzp1 = skipline_zp[4];
		skipline_ipzp2 = skipline_zp[5];
		skipline_opzp1 = output_zeropoint[17];

		//Call to Compute Body 5
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[16];
		opzp2 = output_zeropoint[18];
		opzp3 = output_zeropoint[19];
		opzp4 = output_zeropoint[20];
		useSkipLine = true;
		skipLineOffset = 2;
		bzp_pw_ex = bias_zeropoint[15];
		bzp_dw = bias_zeropoint[16];
		bzp_pw_pj = bias_zeropoint[17];

		// copy_data_length = __FEATURES_6_CONV_0__INPUT_CHAN__ * __FEATURES_6_CONV_0__INPUT_SIZE__ * __FEATURES_6_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_5.start();
#endif
		big_compute_unit(skip_line,
						 net_weight,
						 net_biases_0,
						 out_buffer,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_6_CONV_0__INPUT_CHAN__, __FEATURES_6_CONV_0__INPUT_SIZE__,
						 __FEATURES_6_CONV_0__OUTPUT_CHAN__, __FEATURES_6_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_6_CONV_0__KERNEL_SIZE__, __FEATURES_6_CONV_0__STRIDE_SIZE__, __FEATURES_6_CONV_0__PADDING_SIZE__,
						 __FEATURES_6_CONV_0__WEIGHT_LENGTH__, __FEATURES_6_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_6_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_6_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_6_CONV_0_B_SCALE_LENGTH__, __FEATURES_6_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_6_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_6_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_6_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_6_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_6_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_6_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_6_CONV_2__INPUT_CHAN__, __FEATURES_6_CONV_2__INPUT_SIZE__,
						 __FEATURES_6_CONV_2__OUTPUT_CHAN__, __FEATURES_6_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_6_CONV_2__KERNEL_SIZE__, __FEATURES_6_CONV_2__STRIDE_SIZE__, __FEATURES_6_CONV_2__PADDING_SIZE__,
						 __FEATURES_6_CONV_2__WEIGHT_LENGTH__, __FEATURES_6_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_6_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_6_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_6_CONV_2_B_SCALE_LENGTH__, __FEATURES_6_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_6_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_6_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_6_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_6_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_6_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_6_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_6_CONV_4__INPUT_CHAN__, __FEATURES_6_CONV_4__INPUT_SIZE__,
						 __FEATURES_6_CONV_4__OUTPUT_CHAN__, __FEATURES_6_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_6_CONV_4__KERNEL_SIZE__, __FEATURES_6_CONV_4__STRIDE_SIZE__, __FEATURES_6_CONV_4__PADDING_SIZE__,
						 __FEATURES_6_CONV_4__WEIGHT_LENGTH__, __FEATURES_6_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_6_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_6_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_6_CONV_4_B_SCALE_LENGTH__, __FEATURES_6_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_6_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_6_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_6_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_6_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_6_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_6_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_body5.npy", __FEATURES_6_CONV_4__OUTPUT_CHAN__, __FEATURES_6_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(out_buffer, "./data/out_body5.npy", "Body5", __FEATURES_6_CONV_4__OUTPUT_CHAN__, __FEATURES_6_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_5.stop();
#endif

		output_length = __FEATURES_6_CONV_4__OUTPUT_CHAN__ * __FEATURES_6_CONV_4__OUTPUT_SIZE__ * __FEATURES_6_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(out_buffer,
					skip_line,
					output_0,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(out_buffer, output_0,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/

#ifndef __RELEASE__
		std::cout << "-> Body 6 call." << std::endl;
#endif

		//Body 6 - Weights

		total_weight_length = __FEATURES_7_CONV_0__WEIGHT_LENGTH__ + __FEATURES_7_CONV_2__WEIGHT_LENGTH__ + __FEATURES_7_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_7_CONV_0__WEIGHT_OFFSET__];
		}

		//Call to Compute Body 6
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[17];
		// printf("\n%d\n", (int)opzp1);
		opzp2 = output_zeropoint[21];
		// printf("\n%d\n", (int)opzp1);
		opzp3 = output_zeropoint[22];
		// printf("\n%d\n", (int)opzp1);
		opzp4 = output_zeropoint[23];
		// printf("\n%d\n", (int)opzp1);
		bzp_pw_ex = bias_zeropoint[18];
		bzp_dw = bias_zeropoint[19];
		bzp_pw_pj = bias_zeropoint[20];
		useSkipLine = false;
		skipLineOffset = 2;

#ifdef __HW__
		compute_body_6.start();
#endif
		big_compute_unit(output_0,
						 net_weight,
						 net_biases_0,
						 out_buffer,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_7_CONV_0__INPUT_CHAN__, __FEATURES_7_CONV_0__INPUT_SIZE__,
						 __FEATURES_7_CONV_0__OUTPUT_CHAN__, __FEATURES_7_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_7_CONV_0__KERNEL_SIZE__, __FEATURES_7_CONV_0__STRIDE_SIZE__, __FEATURES_7_CONV_0__PADDING_SIZE__,
						 __FEATURES_7_CONV_0__WEIGHT_LENGTH__, __FEATURES_7_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_7_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_7_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_7_CONV_0_B_SCALE_LENGTH__, __FEATURES_7_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_7_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_7_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_7_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_7_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_7_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_7_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_7_CONV_2__INPUT_CHAN__, __FEATURES_7_CONV_2__INPUT_SIZE__,
						 __FEATURES_7_CONV_2__OUTPUT_CHAN__, __FEATURES_7_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_7_CONV_2__KERNEL_SIZE__, __FEATURES_7_CONV_2__STRIDE_SIZE__, __FEATURES_7_CONV_2__PADDING_SIZE__,
						 __FEATURES_7_CONV_2__WEIGHT_LENGTH__, __FEATURES_7_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_7_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_7_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_7_CONV_2_B_SCALE_LENGTH__, __FEATURES_7_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_7_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_7_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_7_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_7_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_7_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_7_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_7_CONV_4__INPUT_CHAN__, __FEATURES_7_CONV_4__INPUT_SIZE__,
						 __FEATURES_7_CONV_4__OUTPUT_CHAN__, __FEATURES_7_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_7_CONV_4__KERNEL_SIZE__, __FEATURES_7_CONV_4__STRIDE_SIZE__, __FEATURES_7_CONV_4__PADDING_SIZE__,
						 __FEATURES_7_CONV_4__WEIGHT_LENGTH__, __FEATURES_7_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_7_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_7_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_7_CONV_4_B_SCALE_LENGTH__, __FEATURES_7_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_7_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_7_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_7_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_7_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_7_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_7_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_body6.npy", __FEATURES_7_CONV_4__OUTPUT_CHAN__, __FEATURES_7_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(out_buffer, "./data/out_body6.npy", "Body6", __FEATURES_7_CONV_4__OUTPUT_CHAN__, __FEATURES_7_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_6.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 7 call." << std::endl;
#endif

		//Body 7 - Weights

		total_weight_length = __FEATURES_8_CONV_0__WEIGHT_LENGTH__ + __FEATURES_8_CONV_2__WEIGHT_LENGTH__ + __FEATURES_8_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_8_CONV_0__WEIGHT_OFFSET__];
		}

		//skipline zps
		skipline_ipzp1 = skipline_zp[6];
		skipline_ipzp2 = skipline_zp[7];
		skipline_opzp1 = output_zeropoint[24];

		//Call to Compute Body 7
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[23];
		opzp2 = output_zeropoint[25];
		opzp3 = output_zeropoint[26];
		opzp4 = output_zeropoint[27];
		bzp_pw_ex = bias_zeropoint[21];
		bzp_dw = bias_zeropoint[22];
		bzp_pw_pj = bias_zeropoint[23];
		useSkipLine = true;
		skipLineOffset = 3;

		// copy_data_length = __FEATURES_8_CONV_0__INPUT_CHAN__ * __FEATURES_8_CONV_0__INPUT_SIZE__ * __FEATURES_8_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_7.start();
#endif
		big_compute_unit(out_buffer,
						 net_weight,
						 net_biases_0,
						 output_0,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_8_CONV_0__INPUT_CHAN__, __FEATURES_8_CONV_0__INPUT_SIZE__,
						 __FEATURES_8_CONV_0__OUTPUT_CHAN__, __FEATURES_8_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_8_CONV_0__KERNEL_SIZE__, __FEATURES_8_CONV_0__STRIDE_SIZE__, __FEATURES_8_CONV_0__PADDING_SIZE__,
						 __FEATURES_8_CONV_0__WEIGHT_LENGTH__, __FEATURES_8_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_8_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_8_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_8_CONV_0_B_SCALE_LENGTH__, __FEATURES_8_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_8_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_8_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_8_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_8_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_8_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_8_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_8_CONV_2__INPUT_CHAN__, __FEATURES_8_CONV_2__INPUT_SIZE__,
						 __FEATURES_8_CONV_2__OUTPUT_CHAN__, __FEATURES_8_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_8_CONV_2__KERNEL_SIZE__, __FEATURES_8_CONV_2__STRIDE_SIZE__, __FEATURES_8_CONV_2__PADDING_SIZE__,
						 __FEATURES_8_CONV_2__WEIGHT_LENGTH__, __FEATURES_8_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_8_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_8_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_8_CONV_2_B_SCALE_LENGTH__, __FEATURES_8_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_8_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_8_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_8_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_8_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_8_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_8_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_8_CONV_4__INPUT_CHAN__, __FEATURES_8_CONV_4__INPUT_SIZE__,
						 __FEATURES_8_CONV_4__OUTPUT_CHAN__, __FEATURES_8_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_8_CONV_4__KERNEL_SIZE__, __FEATURES_8_CONV_4__STRIDE_SIZE__, __FEATURES_8_CONV_4__PADDING_SIZE__,
						 __FEATURES_8_CONV_4__WEIGHT_LENGTH__, __FEATURES_8_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_8_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_8_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_8_CONV_4_B_SCALE_LENGTH__, __FEATURES_8_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_8_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_8_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_8_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_8_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_8_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_8_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_body7.npy", __FEATURES_8_CONV_4__OUTPUT_CHAN__, __FEATURES_8_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_body7.npy", "Body7", __FEATURES_8_CONV_4__OUTPUT_CHAN__, __FEATURES_8_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_7.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 8 call." << std::endl;
#endif

		output_length = __FEATURES_8_CONV_4__OUTPUT_CHAN__ * __FEATURES_8_CONV_4__OUTPUT_SIZE__ * __FEATURES_8_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(output_0,
					out_buffer,
					skip_line,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(out_buffer, output_0,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/
		//Body 8 - Weights

		total_weight_length = __FEATURES_9_CONV_0__WEIGHT_LENGTH__ + __FEATURES_9_CONV_2__WEIGHT_LENGTH__ + __FEATURES_9_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_9_CONV_0__WEIGHT_OFFSET__];
		}

		//skipline zps
		skipline_ipzp1 = skipline_zp[8];
		skipline_ipzp2 = skipline_zp[9];
		skipline_opzp1 = output_zeropoint[28];

		//Call to Compute Body 8
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[24];
		opzp2 = output_zeropoint[29];
		opzp3 = output_zeropoint[30];
		opzp4 = output_zeropoint[31];
		bzp_pw_ex = bias_zeropoint[24];
		bzp_dw = bias_zeropoint[25];
		bzp_pw_pj = bias_zeropoint[26];
		useSkipLine = true;
		skipLineOffset = 4;

		// copy_data_length = __FEATURES_9_CONV_0__INPUT_CHAN__ * __FEATURES_9_CONV_0__INPUT_SIZE__ * __FEATURES_9_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_8.start();
#endif
		big_compute_unit(skip_line,
						 net_weight,
						 net_biases_0,
						 output_0,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_9_CONV_0__INPUT_CHAN__, __FEATURES_9_CONV_0__INPUT_SIZE__,
						 __FEATURES_9_CONV_0__OUTPUT_CHAN__, __FEATURES_9_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_9_CONV_0__KERNEL_SIZE__, __FEATURES_9_CONV_0__STRIDE_SIZE__, __FEATURES_9_CONV_0__PADDING_SIZE__,
						 __FEATURES_9_CONV_0__WEIGHT_LENGTH__, __FEATURES_9_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_9_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_9_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_9_CONV_0_B_SCALE_LENGTH__, __FEATURES_9_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_9_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_9_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_9_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_9_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_9_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_9_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_9_CONV_2__INPUT_CHAN__, __FEATURES_9_CONV_2__INPUT_SIZE__,
						 __FEATURES_9_CONV_2__OUTPUT_CHAN__, __FEATURES_9_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_9_CONV_2__KERNEL_SIZE__, __FEATURES_9_CONV_2__STRIDE_SIZE__, __FEATURES_9_CONV_2__PADDING_SIZE__,
						 __FEATURES_9_CONV_2__WEIGHT_LENGTH__, __FEATURES_9_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_9_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_9_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_9_CONV_2_B_SCALE_LENGTH__, __FEATURES_9_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_9_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_9_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_9_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_9_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_9_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_9_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_9_CONV_4__INPUT_CHAN__, __FEATURES_9_CONV_4__INPUT_SIZE__,
						 __FEATURES_9_CONV_4__OUTPUT_CHAN__, __FEATURES_9_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_9_CONV_4__KERNEL_SIZE__, __FEATURES_9_CONV_4__STRIDE_SIZE__, __FEATURES_9_CONV_4__PADDING_SIZE__,
						 __FEATURES_9_CONV_4__WEIGHT_LENGTH__, __FEATURES_9_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_9_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_9_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_9_CONV_4_B_SCALE_LENGTH__, __FEATURES_9_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_9_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_9_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_9_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_9_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_9_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_9_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_body8.npy", __FEATURES_9_CONV_4__OUTPUT_CHAN__, __FEATURES_9_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_body8.npy", "Body8", __FEATURES_9_CONV_4__OUTPUT_CHAN__, __FEATURES_9_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_8.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 9 call." << std::endl;
#endif

		output_length = __FEATURES_9_CONV_4__OUTPUT_CHAN__ * __FEATURES_9_CONV_4__OUTPUT_SIZE__ * __FEATURES_9_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(output_0,
					skip_line,
					out_buffer,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);
		/*computeSkipline(output_0, out_buffer,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/

		//Body 9 - Weights

		total_weight_length = __FEATURES_10_CONV_0__WEIGHT_LENGTH__ + __FEATURES_10_CONV_2__WEIGHT_LENGTH__ + __FEATURES_10_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_10_CONV_0__WEIGHT_OFFSET__];
		}

		// for (int idx = 0; idx < __FEATURES_10_CONV_2__WEIGHT_LENGTH__; idx++)
		// {
		// 	net_weight_dw[idx] = net_weight_vect_4bit[idx + __FEATURES_10_CONV_2__WEIGHT_OFFSET__];
		// }

		// for (int idx = 0; idx < __FEATURES_10_CONV_4__WEIGHT_LENGTH__; idx++)
		// {
		// 	net_weight_proj[idx] = net_weight_vect_4bit[idx + __FEATURES_10_CONV_4__WEIGHT_OFFSET__];
		// }

		//skipline zps
		skipline_ipzp1 = skipline_zp[10];
		skipline_ipzp2 = skipline_zp[11];
		skipline_opzp1 = output_zeropoint[32];

		//Call to Compute Body 9
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[28];
		opzp2 = output_zeropoint[33];
		opzp3 = output_zeropoint[34];
		opzp4 = output_zeropoint[35];
		bzp_pw_ex = bias_zeropoint[27];
		bzp_dw = bias_zeropoint[28];
		bzp_pw_pj = bias_zeropoint[29];
		useSkipLine = true;
		skipLineOffset = 5;

		// copy_data_length = __FEATURES_10_CONV_0__INPUT_CHAN__ * __FEATURES_10_CONV_0__INPUT_SIZE__ * __FEATURES_10_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_9.start();
#endif
		big_compute_unit(out_buffer,
						 net_weight,
						 net_biases_0,
						 output_0,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_10_CONV_0__INPUT_CHAN__, __FEATURES_10_CONV_0__INPUT_SIZE__,
						 __FEATURES_10_CONV_0__OUTPUT_CHAN__, __FEATURES_10_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_10_CONV_0__KERNEL_SIZE__, __FEATURES_10_CONV_0__STRIDE_SIZE__, __FEATURES_10_CONV_0__PADDING_SIZE__,
						 __FEATURES_10_CONV_0__WEIGHT_LENGTH__, __FEATURES_10_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_10_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_10_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_10_CONV_0_B_SCALE_LENGTH__, __FEATURES_10_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_10_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_10_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_10_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_10_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_10_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_10_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_10_CONV_2__INPUT_CHAN__, __FEATURES_10_CONV_2__INPUT_SIZE__,
						 __FEATURES_10_CONV_2__OUTPUT_CHAN__, __FEATURES_10_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_10_CONV_2__KERNEL_SIZE__, __FEATURES_10_CONV_2__STRIDE_SIZE__, __FEATURES_10_CONV_2__PADDING_SIZE__,
						 __FEATURES_10_CONV_2__WEIGHT_LENGTH__, __FEATURES_10_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_10_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_10_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_10_CONV_2_B_SCALE_LENGTH__, __FEATURES_10_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_10_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_10_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_10_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_10_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_10_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_10_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_10_CONV_4__INPUT_CHAN__, __FEATURES_10_CONV_4__INPUT_SIZE__,
						 __FEATURES_10_CONV_4__OUTPUT_CHAN__, __FEATURES_10_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_10_CONV_4__KERNEL_SIZE__, __FEATURES_10_CONV_4__STRIDE_SIZE__, __FEATURES_10_CONV_4__PADDING_SIZE__,
						 __FEATURES_10_CONV_4__WEIGHT_LENGTH__, __FEATURES_10_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_10_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_10_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_10_CONV_4_B_SCALE_LENGTH__, __FEATURES_10_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_10_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_10_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_10_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_10_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_10_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_10_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_body9.npy", __FEATURES_10_CONV_4__OUTPUT_CHAN__, __FEATURES_10_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_body9.npy", "Body9", __FEATURES_10_CONV_4__OUTPUT_CHAN__, __FEATURES_10_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_9.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 10 call." << std::endl;
#endif

		output_length = __FEATURES_10_CONV_4__OUTPUT_CHAN__ * __FEATURES_10_CONV_4__OUTPUT_SIZE__ * __FEATURES_10_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(output_0,
					out_buffer,
					skip_line,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(out_buffer, output_0,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/
		//Body 10 - Weights

		total_weight_length = __FEATURES_11_CONV_0__WEIGHT_LENGTH__ + __FEATURES_11_CONV_2__WEIGHT_LENGTH__ + __FEATURES_11_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_11_CONV_0__WEIGHT_OFFSET__];
		}

		//Call to Compute Body 10
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[32];
		opzp2 = output_zeropoint[36];
		opzp3 = output_zeropoint[37];
		opzp4 = output_zeropoint[38];
		bzp_pw_ex = bias_zeropoint[30];
		bzp_dw = bias_zeropoint[31];
		bzp_pw_pj = bias_zeropoint[32];
		useSkipLine = false;
		skipLineOffset = 5;
#ifdef __HW__
		compute_body_10.start();
#endif
		big_compute_unit(skip_line,
						 net_weight,
						 net_biases_0,
						 output_0,

						 weight_zeropoint,

						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_11_CONV_0__INPUT_CHAN__, __FEATURES_11_CONV_0__INPUT_SIZE__,
						 __FEATURES_11_CONV_0__OUTPUT_CHAN__, __FEATURES_11_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_11_CONV_0__KERNEL_SIZE__, __FEATURES_11_CONV_0__STRIDE_SIZE__, __FEATURES_11_CONV_0__PADDING_SIZE__,
						 __FEATURES_11_CONV_0__WEIGHT_LENGTH__, __FEATURES_11_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_11_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_11_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_11_CONV_0_B_SCALE_LENGTH__, __FEATURES_11_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_11_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_11_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_11_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_11_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_11_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_11_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_11_CONV_2__INPUT_CHAN__, __FEATURES_11_CONV_2__INPUT_SIZE__,
						 __FEATURES_11_CONV_2__OUTPUT_CHAN__, __FEATURES_11_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_11_CONV_2__KERNEL_SIZE__, __FEATURES_11_CONV_2__STRIDE_SIZE__, __FEATURES_11_CONV_2__PADDING_SIZE__,
						 __FEATURES_11_CONV_2__WEIGHT_LENGTH__, __FEATURES_11_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_11_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_11_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_11_CONV_2_B_SCALE_LENGTH__, __FEATURES_11_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_11_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_11_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_11_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_11_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_11_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_11_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_11_CONV_4__INPUT_CHAN__, __FEATURES_11_CONV_4__INPUT_SIZE__,
						 __FEATURES_11_CONV_4__OUTPUT_CHAN__, __FEATURES_11_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_11_CONV_4__KERNEL_SIZE__, __FEATURES_11_CONV_4__STRIDE_SIZE__, __FEATURES_11_CONV_4__PADDING_SIZE__,
						 __FEATURES_11_CONV_4__WEIGHT_LENGTH__, __FEATURES_11_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_11_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_11_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_11_CONV_4_B_SCALE_LENGTH__, __FEATURES_11_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_11_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_11_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_11_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_11_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_11_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_11_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_body10.npy", __FEATURES_11_CONV_4__OUTPUT_CHAN__, __FEATURES_11_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_body10.npy", "Body10", __FEATURES_11_CONV_4__OUTPUT_CHAN__, __FEATURES_11_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_10.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 11 call." << std::endl;
#endif

		//Body 11 - Weights

		total_weight_length = __FEATURES_12_CONV_0__WEIGHT_LENGTH__ + __FEATURES_12_CONV_2__WEIGHT_LENGTH__ + __FEATURES_12_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_12_CONV_0__WEIGHT_OFFSET__];
		}

		//skipline zps
		skipline_ipzp1 = skipline_zp[12];
		skipline_ipzp2 = skipline_zp[13];
		skipline_opzp1 = output_zeropoint[39];

		//Call to Compute Body 11
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[38];
		opzp2 = output_zeropoint[40];
		opzp3 = output_zeropoint[41];
		opzp4 = output_zeropoint[42];
		bzp_pw_ex = bias_zeropoint[33];
		bzp_dw = bias_zeropoint[34];
		bzp_pw_pj = bias_zeropoint[35];
		useSkipLine = true;
		skipLineOffset = 6;

		// copy_data_length = __FEATURES_12_CONV_0__INPUT_CHAN__ * __FEATURES_12_CONV_0__INPUT_SIZE__ * __FEATURES_12_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_11.start();
#endif
		big_compute_unit(output_0,
						 net_weight,
						 net_biases_0,
						 out_buffer,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_12_CONV_0__INPUT_CHAN__, __FEATURES_12_CONV_0__INPUT_SIZE__,
						 __FEATURES_12_CONV_0__OUTPUT_CHAN__, __FEATURES_12_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_12_CONV_0__KERNEL_SIZE__, __FEATURES_12_CONV_0__STRIDE_SIZE__, __FEATURES_12_CONV_0__PADDING_SIZE__,
						 __FEATURES_12_CONV_0__WEIGHT_LENGTH__, __FEATURES_12_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_12_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_12_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_12_CONV_0_B_SCALE_LENGTH__, __FEATURES_12_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_12_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_12_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_12_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_12_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_12_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_12_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_12_CONV_2__INPUT_CHAN__, __FEATURES_12_CONV_2__INPUT_SIZE__,
						 __FEATURES_12_CONV_2__OUTPUT_CHAN__, __FEATURES_12_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_12_CONV_2__KERNEL_SIZE__, __FEATURES_12_CONV_2__STRIDE_SIZE__, __FEATURES_12_CONV_2__PADDING_SIZE__,
						 __FEATURES_12_CONV_2__WEIGHT_LENGTH__, __FEATURES_12_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_12_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_12_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_12_CONV_2_B_SCALE_LENGTH__, __FEATURES_12_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_12_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_12_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_12_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_12_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_12_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_12_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_12_CONV_4__INPUT_CHAN__, __FEATURES_12_CONV_4__INPUT_SIZE__,
						 __FEATURES_12_CONV_4__OUTPUT_CHAN__, __FEATURES_12_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_12_CONV_4__KERNEL_SIZE__, __FEATURES_12_CONV_4__STRIDE_SIZE__, __FEATURES_12_CONV_4__PADDING_SIZE__,
						 __FEATURES_12_CONV_4__WEIGHT_LENGTH__, __FEATURES_12_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_12_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_12_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_12_CONV_4_B_SCALE_LENGTH__, __FEATURES_12_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_12_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_12_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_12_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_12_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_12_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_12_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_body11.npy", __FEATURES_12_CONV_4__OUTPUT_CHAN__, __FEATURES_12_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(out_buffer, "./data/out_body11.npy", "Body11", __FEATURES_12_CONV_4__OUTPUT_CHAN__, __FEATURES_12_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_11.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 12 call." << std::endl;
#endif

		output_length = __FEATURES_12_CONV_4__OUTPUT_CHAN__ * __FEATURES_12_CONV_4__OUTPUT_SIZE__ * __FEATURES_12_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(out_buffer,
					output_0,
					skip_line,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(out_buffer, output_0,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/
		//Body 12 - Weights

		total_weight_length = __FEATURES_13_CONV_0__WEIGHT_LENGTH__ + __FEATURES_13_CONV_2__WEIGHT_LENGTH__ + __FEATURES_13_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_13_CONV_0__WEIGHT_OFFSET__];
		}

		//skipline zps
		skipline_ipzp1 = skipline_zp[14];
		skipline_ipzp2 = skipline_zp[15];
		skipline_opzp1 = output_zeropoint[43];

		//Call to Compute Body 12
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[39];
		opzp2 = output_zeropoint[44];
		opzp3 = output_zeropoint[45];
		opzp4 = output_zeropoint[46];
		bzp_pw_ex = bias_zeropoint[36];
		bzp_dw = bias_zeropoint[37];
		bzp_pw_pj = bias_zeropoint[38];
		useSkipLine = true;
		skipLineOffset = 7;
		// copy_data_length = __FEATURES_13_CONV_0__INPUT_CHAN__ * __FEATURES_13_CONV_0__INPUT_SIZE__ * __FEATURES_13_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_12.start();
#endif
		big_compute_unit(skip_line,
						 net_weight,
						 net_biases_0,
						 output_0,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_13_CONV_0__INPUT_CHAN__, __FEATURES_13_CONV_0__INPUT_SIZE__,
						 __FEATURES_13_CONV_0__OUTPUT_CHAN__, __FEATURES_13_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_13_CONV_0__KERNEL_SIZE__, __FEATURES_13_CONV_0__STRIDE_SIZE__, __FEATURES_13_CONV_0__PADDING_SIZE__,
						 __FEATURES_13_CONV_0__WEIGHT_LENGTH__, __FEATURES_13_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_13_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_13_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_13_CONV_0_B_SCALE_LENGTH__, __FEATURES_13_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_13_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_13_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_13_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_13_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_13_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_13_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_13_CONV_2__INPUT_CHAN__, __FEATURES_13_CONV_2__INPUT_SIZE__,
						 __FEATURES_13_CONV_2__OUTPUT_CHAN__, __FEATURES_13_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_13_CONV_2__KERNEL_SIZE__, __FEATURES_13_CONV_2__STRIDE_SIZE__, __FEATURES_13_CONV_2__PADDING_SIZE__,
						 __FEATURES_13_CONV_2__WEIGHT_LENGTH__, __FEATURES_13_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_13_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_13_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_13_CONV_2_B_SCALE_LENGTH__, __FEATURES_13_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_13_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_13_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_13_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_13_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_13_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_13_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_13_CONV_4__INPUT_CHAN__, __FEATURES_13_CONV_4__INPUT_SIZE__,
						 __FEATURES_13_CONV_4__OUTPUT_CHAN__, __FEATURES_13_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_13_CONV_4__KERNEL_SIZE__, __FEATURES_13_CONV_4__STRIDE_SIZE__, __FEATURES_13_CONV_4__PADDING_SIZE__,
						 __FEATURES_13_CONV_4__WEIGHT_LENGTH__, __FEATURES_13_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_13_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_13_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_13_CONV_4_B_SCALE_LENGTH__, __FEATURES_13_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_13_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_13_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_13_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_13_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_13_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_13_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_body12.npy", __FEATURES_13_CONV_4__OUTPUT_CHAN__, __FEATURES_13_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_body12.npy", "Body12", __FEATURES_13_CONV_4__OUTPUT_CHAN__, __FEATURES_13_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_12.stop();
#endif

		output_length = __FEATURES_13_CONV_4__OUTPUT_CHAN__ * __FEATURES_13_CONV_4__OUTPUT_SIZE__ * __FEATURES_13_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(output_0,
					skip_line,
					out_buffer,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(output_0, out_buffer,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/
		//Body 13 - Weights

		total_weight_length = __FEATURES_14_CONV_0__WEIGHT_LENGTH__ + __FEATURES_14_CONV_2__WEIGHT_LENGTH__ + __FEATURES_14_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_14_CONV_0__WEIGHT_OFFSET__];
		}

		//Call to Compute Body 13
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[43];
		opzp2 = output_zeropoint[47];
		opzp3 = output_zeropoint[48];
		opzp4 = output_zeropoint[49];
		bzp_pw_ex = bias_zeropoint[39];
		bzp_dw = bias_zeropoint[40];
		bzp_pw_pj = bias_zeropoint[41];
		useSkipLine = false;
		skipLineOffset = 7;
#ifdef __HW__
		compute_body_13.start();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 13 call." << std::endl;
#endif

		big_compute_unit(out_buffer,
						 net_weight,
						 net_biases_0,
						 output_0,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_14_CONV_0__INPUT_CHAN__, __FEATURES_14_CONV_0__INPUT_SIZE__,
						 __FEATURES_14_CONV_0__OUTPUT_CHAN__, __FEATURES_14_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_14_CONV_0__KERNEL_SIZE__, __FEATURES_14_CONV_0__STRIDE_SIZE__, __FEATURES_14_CONV_0__PADDING_SIZE__,
						 __FEATURES_14_CONV_0__WEIGHT_LENGTH__, __FEATURES_14_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_14_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_14_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_14_CONV_0_B_SCALE_LENGTH__, __FEATURES_14_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_14_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_14_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_14_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_14_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_14_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_14_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_14_CONV_2__INPUT_CHAN__, __FEATURES_14_CONV_2__INPUT_SIZE__,
						 __FEATURES_14_CONV_2__OUTPUT_CHAN__, __FEATURES_14_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_14_CONV_2__KERNEL_SIZE__, __FEATURES_14_CONV_2__STRIDE_SIZE__, __FEATURES_14_CONV_2__PADDING_SIZE__,
						 __FEATURES_14_CONV_2__WEIGHT_LENGTH__, __FEATURES_14_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_14_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_14_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_14_CONV_2_B_SCALE_LENGTH__, __FEATURES_14_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_14_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_14_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_14_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_14_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_14_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_14_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_14_CONV_4__INPUT_CHAN__, __FEATURES_14_CONV_4__INPUT_SIZE__,
						 __FEATURES_14_CONV_4__OUTPUT_CHAN__, __FEATURES_14_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_14_CONV_4__KERNEL_SIZE__, __FEATURES_14_CONV_4__STRIDE_SIZE__, __FEATURES_14_CONV_4__PADDING_SIZE__,
						 __FEATURES_14_CONV_4__WEIGHT_LENGTH__, __FEATURES_14_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_14_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_14_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_14_CONV_4_B_SCALE_LENGTH__, __FEATURES_14_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_14_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_14_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_14_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_14_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_14_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_14_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_body13.npy", __FEATURES_14_CONV_4__OUTPUT_CHAN__, __FEATURES_14_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(output_0, "./data/out_body13.npy", "Body13", __FEATURES_14_CONV_4__OUTPUT_CHAN__, __FEATURES_14_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_13.stop();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 14 call." << std::endl;
#endif

		//Body 14 - Weights

		total_weight_length = __FEATURES_15_CONV_0__WEIGHT_LENGTH__ + __FEATURES_15_CONV_2__WEIGHT_LENGTH__ + __FEATURES_15_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_15_CONV_0__WEIGHT_OFFSET__];
		}

		//skipline zps
		skipline_ipzp1 = skipline_zp[16];
		skipline_ipzp2 = skipline_zp[17];
		skipline_opzp1 = output_zeropoint[50];

		//Call to Compute Body 14
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[46];
		opzp2 = output_zeropoint[51];
		opzp3 = output_zeropoint[52];
		opzp4 = output_zeropoint[53];
		bzp_pw_ex = bias_zeropoint[42];
		bzp_dw = bias_zeropoint[43];
		bzp_pw_pj = bias_zeropoint[44];
		useSkipLine = true;
		skipLineOffset = 8;

		// copy_data_length = __FEATURES_15_CONV_0__INPUT_CHAN__ * __FEATURES_15_CONV_0__INPUT_SIZE__ * __FEATURES_15_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_14.start();
#endif
		big_compute_unit(output_0,
						 net_weight,
						 net_biases_0,
						 out_buffer,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_15_CONV_0__INPUT_CHAN__, __FEATURES_15_CONV_0__INPUT_SIZE__,
						 __FEATURES_15_CONV_0__OUTPUT_CHAN__, __FEATURES_15_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_15_CONV_0__KERNEL_SIZE__, __FEATURES_15_CONV_0__STRIDE_SIZE__, __FEATURES_15_CONV_0__PADDING_SIZE__,
						 __FEATURES_15_CONV_0__WEIGHT_LENGTH__, __FEATURES_15_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_15_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_15_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_15_CONV_0_B_SCALE_LENGTH__, __FEATURES_15_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_15_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_15_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_15_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_15_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_15_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_15_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_15_CONV_2__INPUT_CHAN__, __FEATURES_15_CONV_2__INPUT_SIZE__,
						 __FEATURES_15_CONV_2__OUTPUT_CHAN__, __FEATURES_15_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_15_CONV_2__KERNEL_SIZE__, __FEATURES_15_CONV_2__STRIDE_SIZE__, __FEATURES_15_CONV_2__PADDING_SIZE__,
						 __FEATURES_15_CONV_2__WEIGHT_LENGTH__, __FEATURES_15_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_15_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_15_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_15_CONV_2_B_SCALE_LENGTH__, __FEATURES_15_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_15_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_15_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_15_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_15_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_15_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_15_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_15_CONV_4__INPUT_CHAN__, __FEATURES_15_CONV_4__INPUT_SIZE__,
						 __FEATURES_15_CONV_4__OUTPUT_CHAN__, __FEATURES_15_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_15_CONV_4__KERNEL_SIZE__, __FEATURES_15_CONV_4__STRIDE_SIZE__, __FEATURES_15_CONV_4__PADDING_SIZE__,
						 __FEATURES_15_CONV_4__WEIGHT_LENGTH__, __FEATURES_15_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_15_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_15_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_15_CONV_4_B_SCALE_LENGTH__, __FEATURES_15_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_15_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_15_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_15_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_15_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_15_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_15_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_body14.npy", __FEATURES_15_CONV_4__OUTPUT_CHAN__, __FEATURES_15_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(out_buffer, "./data/out_body14.npy", "Body14", __FEATURES_15_CONV_4__OUTPUT_CHAN__, __FEATURES_15_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_14.stop();
#endif

		output_length = __FEATURES_15_CONV_4__OUTPUT_CHAN__ * __FEATURES_15_CONV_4__OUTPUT_SIZE__ * __FEATURES_15_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(out_buffer,
					output_0,
					skip_line,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(output_0, out_buffer,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/

		//Body 15 - Weights

		total_weight_length = __FEATURES_16_CONV_0__WEIGHT_LENGTH__ + __FEATURES_16_CONV_2__WEIGHT_LENGTH__ + __FEATURES_16_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_16_CONV_0__WEIGHT_OFFSET__];
		}

		//skipline zps
		skipline_ipzp1 = skipline_zp[18];
		skipline_ipzp2 = skipline_zp[19];
		skipline_opzp1 = output_zeropoint[57];

		//Call to Compute Body 15
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[50];
		opzp2 = output_zeropoint[55];
		opzp3 = output_zeropoint[56];
		opzp4 = output_zeropoint[57];
		bzp_pw_ex = bias_zeropoint[45];
		bzp_dw = bias_zeropoint[46];
		bzp_pw_pj = bias_zeropoint[47];
		useSkipLine = true;
		skipLineOffset = 9;
		// copy_data_length = __FEATURES_16_CONV_0__INPUT_CHAN__ * __FEATURES_16_CONV_0__INPUT_SIZE__ * __FEATURES_16_CONV_0__INPUT_SIZE__;
		// copyDataSkipline(output_0, data_skipline, copy_data_length);

#ifdef __HW__
		compute_body_15.start();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 15 call." << std::endl;
#endif

		big_compute_unit(skip_line,
						 net_weight,
						 net_biases_0,
						 out_buffer,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_16_CONV_0__INPUT_CHAN__, __FEATURES_16_CONV_0__INPUT_SIZE__,
						 __FEATURES_16_CONV_0__OUTPUT_CHAN__, __FEATURES_16_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_16_CONV_0__KERNEL_SIZE__, __FEATURES_16_CONV_0__STRIDE_SIZE__, __FEATURES_16_CONV_0__PADDING_SIZE__,
						 __FEATURES_16_CONV_0__WEIGHT_LENGTH__, __FEATURES_16_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_16_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_16_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_16_CONV_0_B_SCALE_LENGTH__, __FEATURES_16_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_16_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_16_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_16_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_16_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_16_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_16_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_16_CONV_2__INPUT_CHAN__, __FEATURES_16_CONV_2__INPUT_SIZE__,
						 __FEATURES_16_CONV_2__OUTPUT_CHAN__, __FEATURES_16_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_16_CONV_2__KERNEL_SIZE__, __FEATURES_16_CONV_2__STRIDE_SIZE__, __FEATURES_16_CONV_2__PADDING_SIZE__,
						 __FEATURES_16_CONV_2__WEIGHT_LENGTH__, __FEATURES_16_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_16_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_16_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_16_CONV_2_B_SCALE_LENGTH__, __FEATURES_16_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_16_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_16_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_16_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_16_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_16_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_16_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_16_CONV_4__INPUT_CHAN__, __FEATURES_16_CONV_4__INPUT_SIZE__,
						 __FEATURES_16_CONV_4__OUTPUT_CHAN__, __FEATURES_16_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_16_CONV_4__KERNEL_SIZE__, __FEATURES_16_CONV_4__STRIDE_SIZE__, __FEATURES_16_CONV_4__PADDING_SIZE__,
						 __FEATURES_16_CONV_4__WEIGHT_LENGTH__, __FEATURES_16_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_16_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_16_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_16_CONV_4_B_SCALE_LENGTH__, __FEATURES_16_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_16_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_16_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_16_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_16_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_16_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_16_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_body15.npy", __FEATURES_16_CONV_4__OUTPUT_CHAN__, __FEATURES_16_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(out_buffer, "./data/out_body15.npy", "Body15", __FEATURES_16_CONV_4__OUTPUT_CHAN__, __FEATURES_16_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_15.stop();
#endif

		output_length = __FEATURES_16_CONV_4__OUTPUT_CHAN__ * __FEATURES_16_CONV_4__OUTPUT_SIZE__ * __FEATURES_16_CONV_4__OUTPUT_SIZE__;

		skiplinelayerParamOffset = (skipLineOffset << 2);
		xMultiplier = skipline_param[skiplinelayerParamOffset];
		xShift = skipline_param[skiplinelayerParamOffset + 1];
		yMultiplier = skipline_param[skiplinelayerParamOffset + 2];
		yShift = skipline_param[skiplinelayerParamOffset + 3];

		QVector_Add(out_buffer,
					skip_line,
					output_0,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					xMultiplier,
					xShift,
					yMultiplier,
					yShift,
					output_length);

		/*computeSkipline(out_buffer, output_0,
					skipline_ipzp1,
					skipline_ipzp2,
					skipline_opzp1,
					skipline_param,
					output_length,
					layerNo,
					skipLineOffset);*/
		//Body 16 - Weights

		total_weight_length = __FEATURES_17_CONV_0__WEIGHT_LENGTH__ + __FEATURES_17_CONV_2__WEIGHT_LENGTH__ + __FEATURES_17_CONV_4__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __FEATURES_17_CONV_0__WEIGHT_OFFSET__];
		}

		//Call to Compute Body 16
		layerNo = layerNo + 1;
		opzp1 = output_zeropoint[54];
		opzp2 = output_zeropoint[58];
		opzp3 = output_zeropoint[59];
		opzp4 = output_zeropoint[60];
		bzp_pw_ex = bias_zeropoint[48];
		bzp_dw = bias_zeropoint[49];
		bzp_pw_pj = bias_zeropoint[50];
		useSkipLine = false;
		skipLineOffset = 9;
#ifdef __HW__
		compute_body_16.start();
#endif

#ifndef __RELEASE__
		std::cout << "-> Body 16 call." << std::endl;
#endif

		big_compute_unit(output_0,
						 net_weight,
						 net_biases_0,
						 out_buffer,

						 weight_zeropoint,
						 iMult_bias_acc,
						 nShift_bias_acc,
						 iMult_output,
						 nShift_output,

						 //Layer 1 data
						 __FEATURES_17_CONV_0__INPUT_CHAN__, __FEATURES_17_CONV_0__INPUT_SIZE__,
						 __FEATURES_17_CONV_0__OUTPUT_CHAN__, __FEATURES_17_CONV_0__OUTPUT_SIZE__,
						 __FEATURES_17_CONV_0__KERNEL_SIZE__, __FEATURES_17_CONV_0__STRIDE_SIZE__, __FEATURES_17_CONV_0__PADDING_SIZE__,
						 __FEATURES_17_CONV_0__WEIGHT_LENGTH__, __FEATURES_17_CONV_0__WEIGHT_OFFSET__,
						 __FEATURES_17_CONV_0_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_17_CONV_0_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_17_CONV_0_B_SCALE_LENGTH__, __FEATURES_17_CONV_0_B_SCALE_OFFSET__,
						 __FEATURES_17_CONV_0_W_ZERO_POINT_LENGTH__, __FEATURES_17_CONV_0_W_ZERO_POINT_OFFSET__,
						 __FEATURES_17_CONV_0_ACCUM_SCALE_LENGTH__, __FEATURES_17_CONV_0_ACCUM_SCALE_OFFSET__,
						 __FEATURES_17_CONV_0_BASE_B_Q_LENGTH__, __FEATURES_17_CONV_0_BASE_B_Q_OFFSET__,

						 //Layer 2 data
						 __FEATURES_17_CONV_2__INPUT_CHAN__, __FEATURES_17_CONV_2__INPUT_SIZE__,
						 __FEATURES_17_CONV_2__OUTPUT_CHAN__, __FEATURES_17_CONV_2__OUTPUT_SIZE__,
						 __FEATURES_17_CONV_2__KERNEL_SIZE__, __FEATURES_17_CONV_2__STRIDE_SIZE__, __FEATURES_17_CONV_2__PADDING_SIZE__,
						 __FEATURES_17_CONV_2__WEIGHT_LENGTH__, __FEATURES_17_CONV_2__WEIGHT_OFFSET__,
						 __FEATURES_17_CONV_2_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_17_CONV_2_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_17_CONV_2_B_SCALE_LENGTH__, __FEATURES_17_CONV_2_B_SCALE_OFFSET__,
						 __FEATURES_17_CONV_2_W_ZERO_POINT_LENGTH__, __FEATURES_17_CONV_2_W_ZERO_POINT_OFFSET__,
						 __FEATURES_17_CONV_2_ACCUM_SCALE_LENGTH__, __FEATURES_17_CONV_2_ACCUM_SCALE_OFFSET__,
						 __FEATURES_17_CONV_2_BASE_B_Q_LENGTH__, __FEATURES_17_CONV_2_BASE_B_Q_OFFSET__,

						 //Layer 3 data
						 __FEATURES_17_CONV_4__INPUT_CHAN__, __FEATURES_17_CONV_4__INPUT_SIZE__,
						 __FEATURES_17_CONV_4__OUTPUT_CHAN__, __FEATURES_17_CONV_4__OUTPUT_SIZE__,
						 __FEATURES_17_CONV_4__KERNEL_SIZE__, __FEATURES_17_CONV_4__STRIDE_SIZE__, __FEATURES_17_CONV_4__PADDING_SIZE__,
						 __FEATURES_17_CONV_4__WEIGHT_LENGTH__, __FEATURES_17_CONV_4__WEIGHT_OFFSET__,
						 __FEATURES_17_CONV_4_OUTPUT_ZERO_POINT_LENGTH__, __FEATURES_17_CONV_4_OUTPUT_ZERO_POINT_OFFSET__,
						 __FEATURES_17_CONV_4_B_SCALE_LENGTH__, __FEATURES_17_CONV_4_B_SCALE_OFFSET__,
						 __FEATURES_17_CONV_4_W_ZERO_POINT_LENGTH__, __FEATURES_17_CONV_4_W_ZERO_POINT_OFFSET__,
						 __FEATURES_17_CONV_4_ACCUM_SCALE_LENGTH__, __FEATURES_17_CONV_4_ACCUM_SCALE_OFFSET__,
						 __FEATURES_17_CONV_4_BASE_B_Q_LENGTH__, __FEATURES_17_CONV_4_BASE_B_Q_OFFSET__,

						 layerNo,

						 //opzp
						 opzp1,
						 opzp2,
						 opzp3,
						 opzp4,
						 bzp_pw_ex,
						 bzp_dw,
						 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_body16.npy", __FEATURES_17_CONV_4__OUTPUT_CHAN__, __FEATURES_17_CONV_4__OUTPUT_SIZE__);
#endif

		check_output(out_buffer, "./data/out_body16.npy", "Body16", __FEATURES_17_CONV_4__OUTPUT_CHAN__, __FEATURES_17_CONV_4__OUTPUT_SIZE__);

#endif
#endif

#ifdef __HW__
		compute_body_16.stop();
#endif

		total_weight_length = __CONV_0__WEIGHT_LENGTH__;
		for (int idx = 0; idx < total_weight_length; idx++)
		{
			net_weight[idx] = net_weight_vect_4bit[idx + __CONV_0__WEIGHT_OFFSET__];
		}

		//Call to Compute PointWise
		layerNo = layerNo + 1;

		//opzp
		opzp1 = 126;
		opzp2 = 0;
		bzp_pw_pj = bias_zeropoint[51];
		//std::cout << "Head!" << std::endl;

#ifdef __HW__
		compute_tail_cnt.start();
#endif

#ifndef __RELEASE__
		std::cout << "-> Tail Call." << std::endl;
#endif

		compute_tail(out_buffer,
					 net_weight,
					 net_biases_0,
					 output_0,

					 weight_zeropoint,
					 iMult_bias_acc,
					 nShift_bias_acc,
					 iMult_output,
					 nShift_output,

					 opzp1,
					 opzp2,
					 bzp_pw_pj);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(output_0, "./data/out_avg.npy", __AVG_POOLING_INPUT_CHAN_SIZE__, 1);
#endif

		check_output(output_0, "./data/out_avg.npy", "tail", __AVG_POOLING_INPUT_CHAN_SIZE__, 1);

#endif
#endif

#ifdef __HW__
		compute_tail_cnt.stop();
#endif

		for (int idx = 0; idx < __CLASSIFIER__WEIGHT_LENGTH__; idx++)
		{
			net_weight_classifier[idx] = net_classifier_weight_vect[idx];
		}

		//Call to Compute PointWise
		layerNo = layerNo + 1;

		//opzp
		opzp1 = 0;
		opzp2 = 77;
		bzp_pw_pj = bias_zeropoint[52];

#ifdef __HW__
		compute_linear_cnt.start();
#endif
#ifndef __RELEASE__
		std::cout << "-> Linear call." << std::endl;
#endif

		compute_linear(output_0,
					   net_weight_classifier,
					   net_biases_0,
					   out_buffer,

					   weight_zeropoint,
					   bzp_pw_pj,
					   iMult_bias_acc,
					   nShift_bias_acc,
					   iMult_output,
					   nShift_output,

					   opzp1,
					   opzp2);

		//vector_trunc_clip<dType_16t, __LINEAR_ROW_SIZE__>(b16t_out_buffer, out_buffer);

#ifndef __RELEASE__
#ifdef __DEBUG__
#ifdef __SAVE_INTRM_RES__
		saveNPYFile(out_buffer, "./data/out_linear.npy", __LINEAR_ROW_SIZE__, 1);
#endif

		check_output(out_buffer, "./data/out_linear.npy", "Linear", __LINEAR_ROW_SIZE__, 1);

#endif
#endif

		std::cout << "-> Finished computation." << std::endl;

#ifdef __HW__
		compute_linear_cnt.stop();
		compute_Total.stop();
#endif
		//print_qnt_results(image_0, __FEATURES_0_0__OUTPUT_CHAN__, __FEATURES_0_0__OUTPUT_SIZE__);
		std::cout
			<< "-> Writing intermediate layer data." << std::endl;
		std::vector<unsigned long> norm_output_size;

#define OUT_CHAN 1000
#define IM_SIZE 1

		std::vector<dType_8u> output_norm_vect(OUT_CHAN * IM_SIZE * IM_SIZE);
		std::vector<float> fp_output(OUT_CHAN * IM_SIZE * IM_SIZE);
		std::vector<float> conf_output(OUT_CHAN * IM_SIZE * IM_SIZE);

		for (int i = 0; i < IM_SIZE; i++)
		{
			for (int j = 0; j < IM_SIZE; j++)
			{
				for (int ch = 0; ch < OUT_CHAN; ch++)
				{
					//output_norm_vect[ch * (__FEATURES_0_0__OUTPUT_SIZE__ * __FEATURES_0_0__OUTPUT_SIZE__) + (i * __FEATURES_0_0__OUTPUT_SIZE__) + j] = image_0[ch * (__FEATURES_0_0__OUTPUT_SIZE__ * __FEATURES_0_0__OUTPUT_SIZE__) + (i * __FEATURES_0_0__OUTPUT_SIZE__) + j];
					//output_norm_vect[ch * (IM_SIZE * IM_SIZE) + (i * IM_SIZE) + j] = output_0[ch * (IM_SIZE * IM_SIZE) + (i * IM_SIZE) + j];
					output_norm_vect[ch * (IM_SIZE * IM_SIZE) + (i * IM_SIZE) + j] = out_buffer[(i * IM_SIZE * OUT_CHAN) + (j * OUT_CHAN) + ch];
					fp_output[ch * (IM_SIZE * IM_SIZE) + (i * IM_SIZE) + j] = ((float)out_buffer[(i * IM_SIZE * OUT_CHAN) + (j * OUT_CHAN) + ch] - 77) / 4.4375;
				}
			}
		}
		const long unsigned int dim[3] = {OUT_CHAN, IM_SIZE, IM_SIZE};
		softmax<float>(fp_output, conf_output);

		float max = *(std::max_element(conf_output.begin(), conf_output.end())) * 100;

		int argMax = std::distance(conf_output.begin(), std::max_element(conf_output.begin(), conf_output.end()));

		std::cout << "-> The object is \033[1;31m" << classes[argMax] << " \033[0m"
				  << "and I am \033[1;34m" << std::setprecision(4) << max << "%"
				  << "\033[0m"
				  << " sure!" << std::endl;

		
		// std::string show_class = "Object: ";
		// show_class = show_class + classes[argMax];
		// std::string show_conf = "Confidence: ";
		// show_conf = show_conf + std::to_string(max);



		
#ifdef __HW__

		uint64_t compute_head_0_cycles = compute_head_0.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_0 " << compute_head_0_cycles << "\t~\t" << (int)(1 / (compute_head_0_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_1_cycles = compute_body_1.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_1 " << compute_body_1_cycles << "\t~\t" << (int)(1 / (compute_body_1_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_2_cycles = compute_body_2.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_2 " << compute_body_2_cycles << "\t~\t" << (int)(1 / (compute_body_2_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_3_cycles = compute_body_3.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_3 " << compute_body_3_cycles << "\t~\t" << (int)(1 / (compute_body_3_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_4_cycles = compute_body_4.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_4 " << compute_body_4_cycles << "\t~\t" << (int)(1 / (compute_body_4_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_5_cycles = compute_body_5.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_5 " << compute_body_5_cycles << "\t~\t" << (int)(1 / (compute_body_5_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_6_cycles = compute_body_6.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_6 " << compute_body_6_cycles << "\t~\t" << (int)(1 / (compute_body_6_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_7_cycles = compute_body_7.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_7 " << compute_body_7_cycles << "\t~\t" << (int)(1 / (compute_body_7_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_8_cycles = compute_body_8.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_8 " << compute_body_8_cycles << "\t~\t" << (int)(1 / (compute_body_8_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_9_cycles = compute_body_9.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_9 " << compute_body_9_cycles << "\t~\t" << (int)(1 / (compute_body_9_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_10_cycles = compute_body_10.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_10 " << compute_body_10_cycles << "\t~\t" << (int)(1 / (compute_body_10_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_11_cycles = compute_body_11.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_11 " << compute_body_11_cycles << "\t~\t" << (int)(1 / (compute_body_11_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_12_cycles = compute_body_12.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_12 " << compute_body_12_cycles << "\t~\t" << (int)(1 / (compute_body_12_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_13_cycles = compute_body_13.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_13 " << compute_body_13_cycles << "\t~\t" << (int)(1 / (compute_body_13_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_14_cycles = compute_body_14.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_14 " << compute_body_14_cycles << "\t~\t" << (int)(1 / (compute_body_14_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_15_cycles = compute_body_15.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_15 " << compute_body_15_cycles << "\t~\t" << (int)(1 / (compute_body_15_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_body_16_cycles = compute_body_16.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_body_16 " << compute_body_16_cycles << "\t~\t" << (int)(1 / (compute_body_16_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_tail_cycles = compute_tail_cnt.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_tail_cnt " << compute_tail_cycles << "\t~\t" << (int)(1 / (compute_tail_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_linear_cycles = compute_linear_cnt.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_linear_cnt " << compute_linear_cycles << "\t~\t" << (int)(1 / (compute_linear_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t Total_wo_host_time_cycles = compute_head_0_cycles + compute_body_1_cycles + compute_body_2_cycles + compute_body_3_cycles + compute_body_4_cycles + compute_body_5_cycles + compute_body_6_cycles + compute_body_7_cycles + compute_body_8_cycles + compute_body_9_cycles + compute_body_10_cycles + compute_body_11_cycles + compute_body_12_cycles + compute_body_13_cycles + compute_body_14_cycles + compute_body_15_cycles + compute_body_16_cycles + compute_tail_cycles + compute_linear_cycles;
		std::cout << "-> Number of CPU cycles for Total_wo_host_time " << Total_wo_host_time_cycles << "\t~\t" << (int)(1 / (Total_wo_host_time_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;
		uint64_t compute_Total_cycles = compute_Total.avg_cpu_cycles();
		std::cout << "-> Number of CPU cycles for compute_Total " << compute_Total_cycles << "\t~\t" << (int)(1 / (compute_Total_cycles * (1 / (1.5 * 1000000000)))) << " FPS" << std::endl;

		std::cout << "-> Showing the image. Please close the image window to continue..." << std::endl;

		// cv::Size size1(500, 500); //the dst image size,e.g.192x192
		// std::cout << "-> Resizing image file..." << std::endl;
		// cvtColor(src, disp_rgb, cv::COLOR_RGB2BGR);
		
		// resize(dst_rs, disp_rgb, size1); //resize image

		std::string show_class = "Object: ";
		show_class = show_class + classes[argMax];
		std::string show_conf = "Confidence: ";
		show_conf = show_conf + std::to_string(max);
		std::string show_fps = "FPS: ";
		show_fps = show_fps + std::to_string((int)(1 / (compute_Total_cycles * (1 / (1.5 * 1000000000)))));


		cv::putText(src, show_class, cv::Point(5,100), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,225), 1);
		// cv::putText(src, show_conf, cv::Point(5,130), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,225), 1);
		cv::putText(src, show_fps, cv::Point(5,155), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,225), 1);
		// cv::namedWindow("TeCSAR-DeepDive", cv::WINDOW_AUTOSIZE);
		cv::imshow("TeCSAR-DeepDive-Input", src);
		char key = (char) cv::waitKey(1);
		if (key == 27) break;
		else continue;
		// if (key == ' ') continue;
		// cv::waitKey();

#endif
	}

	std::cout << "-> Freeing allocated memory..." << std::endl;

	generic_free(image_0);
	generic_free(net_biases_0);
	generic_free(output_0);

	return EXIT_SUCCESS;
}
