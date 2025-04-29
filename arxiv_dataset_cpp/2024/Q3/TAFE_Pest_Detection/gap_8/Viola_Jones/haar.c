 /* Copyright (C) 2023-2024 University of Bologna, Italy.
 *  Created on: Apr 21, 2022
 *      Author: olmo baldoni
 *      email:  olmo.baldoni@studio.unibo.it
 *  Modified by 
 *          Carlo Montanari calo.montanari4@studio.unibo.it
*/

// #include <stdio.h>
// #include <math.h>
// #include <string.h>

// /***************************Begin User Header*******************************/
#include "haar.h"
#include "cascade.h"
#include "cluster.h"
#include "image.h"
#include "pmsis.h"
// #include "Gap.h"
//#include "check.h"
// #define DEBUG 

PI_L1 float l1_scaleFactor;
PI_L1 MyCascade l1_cascade;
PI_L1 uint8_t l1_insect_stages_array[(sizeof(insect_stages_array)/sizeof(uint8_t))];
PI_L1 int16_t l1_insect_stages_thresh_array[sizeof(insect_stages_thresh_array)/sizeof(int16_t)];
PI_L1 int16_t l1_insect_node_thresh_array[(sizeof(insect_node_thresh_array)/sizeof(int16_t))];
PI_L1 int16_t l1_insect_alpha1_array[(sizeof(insect_alpha1_array)/sizeof(int16_t))];
PI_L1 int16_t l1_insect_alpha2_array[(sizeof(insect_alpha2_array)/sizeof(int16_t))];
PI_L1 int8_t  l1_insect_num_rectangles_array[(sizeof(insect_num_rectangles_array)/sizeof(int8_t))];
PI_L1 int8_t  l1_insect_weights_array[(sizeof(insect_weights_array)/sizeof(int8_t))];
PI_L1 int8_t  l1_insect_rectangles_array[(sizeof(insect_rectangles_array)/sizeof(int8_t))];



int feature_global_counter = 0;

MyRect detections[maxsize];
/*
 * buffer stuff
 */

int isfull() {
	if (top == maxsize)
		return 1;
	else
		return 0;
}

void push (MyRect data) {
	pi_cl_team_critical_enter();
	if (!isfull()) { 
		top = top + 1;
		detections[top] = data;
	}
	pi_cl_team_critical_exit();
}

int invoked_run_cascade_class = 0;

/*
 * function to load the cascade contained in cascade.h
 */

void loadCascade(MyCascade *cascade) {
	
	//load the built-in cascade
    int i;
	
    cascade->total_nodes = 0;
	
    cascade->total_rectangles = 0;
	
    cascade->n_stages                = insect_n_stages;
	
    cascade->orig_window_size.height = insect_window_h;
	
    cascade->orig_window_size.width  = insect_window_w;
	
    cascade->stages_array            = (uint8_t*)insect_stages_array;
	
    cascade->stages_thresh_array     = (int16_t*)insect_stages_thresh_array;
	
    cascade->tree_tresh_array        = (int16_t*)insect_node_thresh_array;
	
    cascade->alpha1_array            = (int16_t*)insect_alpha1_array;
	
    cascade->alpha2_array            = (int16_t*)insect_alpha2_array;
	
    cascade->num_rectangles_array  	 = (int8_t*)insect_num_rectangles_array;
	
    cascade->weights_array           = (int8_t*)insect_weights_array;
	
    cascade->rectangles_array        = (int8_t*)insect_rectangles_array;
	
    //sum the number of features in all stages
    for (i = 0;i < cascade->n_stages;i++) {
         cascade->total_nodes += cascade->stages_array[i];
    }

    //sum the number of rectangles in all features
    for (i = 0;i < cascade->total_nodes;i++) {
    	cascade->total_rectangles += cascade->num_rectangles_array[i];
    }

}


/*
 * function to calculate integral image and square integral image
 */

void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum) {

	int i, j;
	
	//source pixel value
	unsigned char val;

	int height = src->height;
	int width = src->width;
	unsigned char *data = src->data;

	int s, sq;
	int temp_s, temp_sq;

	//loop over the number of rows
	for (i = 0;i < height;i++) {
		s = 0;
		sq = 0;
		//loop over the number of columns
		for (j = 0;j < width;j++) {
			val = data[i*width + j];

			//sum of the current row
			s += val;
			sq += val*val;

			temp_s = s;
			temp_sq = sq;

			if (i != 0) {
				temp_s += sum->data[(i - 1)*width + j];
				temp_sq += sqsum->data[(i - 1)*width + j];
			}

			sum->data[i*width + j] = temp_s;
			sqsum->data[i*width + j] = temp_sq;
		}
	}


}



// /*
//  * This function downsample an image using nearest neighbor
//  * It is used to build the image pyramid
//  * http://tech-algorithm.com/articles/nearest-neighbor-image-scaling/
//  */

void nearestNeighbor(MyImage *src, MyImage *dst) {

    int y;
    int j;
    int x;
    int i;
    unsigned char* t;
    unsigned char* p;
    int w1 = src->width;
    int h1 = src->height;
    int w2 = dst->width;
    int h2 = dst->height;

    int rat = 0;

    unsigned char* src_data = src->data;
    unsigned char* dst_data = dst->data;

	
    int x_ratio = (int)((w1<<16)/w2) +1;

    int y_ratio = (int)((h1<<16)/h2) +1;
	

    for (i=0;i<h2;i++) {
        t = dst_data + i*w2;
        y = ((i*y_ratio)>>16);
        p = src_data + y*w1;
        rat = 0;
        for (j=0;j<w2;j++) {
	        x = (rat>>16);
	        *t++ = p[x];
	        rat += x_ratio;
	    }
    }
}


/*
 * evaluates each classifier of a stage
 */

int evalWeakClassifier(MyCascade *_cascade, int variance, int x, int y, int w_index, int r_index, int t_index)	{
	// int core_id=pi_core_id();
	// if(core_id==0){
	// invoked_EWC++;}

	int feature_sum = 0;
		
	//the node threshold is multiplied by standard deviation of the sub-window

	int t1 = _cascade->tree_tresh_array[t_index];// t = 0.05*t1

	if (t1 > 0){
		t1 = t1*t1;
	}
	else{
		t1 = -t1*t1;
	}

	t1 = t1*variance;

	int feature_rects = _cascade->num_rectangles_array[t_index];

	int sum_width =_cascade->sum->width;

	int *sum_data =&_cascade->sum->data[0]; 

	int fc0, fc1, fc2, fc3;		//angles of the filter rectangle in relation to the sliding window and in relation to the integral image

	//v4s point ={x, y, x, y};

	for (int i = 0; i < feature_rects; i++) {

		int8_t * a_ptr = &_cascade->rectangles_array[r_index +i*4];


		//loads the coordinates of the filter rectangle
		int8_t node_x = a_ptr[0];
		int8_t node_y = a_ptr[1];
		int8_t node_width = a_ptr[2];
		int8_t node_height = a_ptr[3];
				
		fc0 = ((y + node_y)*sum_width) + (x + node_x);
		fc1 = fc0 + node_width;
		fc2 = ((y + node_y + node_height)*sum_width) + (x + node_x);
		fc3 = fc2 + node_width;
		
		feature_sum += (sum_data[fc0] + sum_data[fc3] - sum_data[fc1] - sum_data[fc2]) * _cascade->weights_array[w_index];
		
		w_index ++;
		
	}	

	int feature_sum_quad= 20*feature_sum * 20*feature_sum;

	if (feature_sum < 0){
		feature_sum_quad = - feature_sum_quad;
	}


	if (feature_sum_quad>= t1) {
		return _cascade->alpha2_array[t_index];
	}

	return _cascade ->alpha1_array[t_index];
}


/*
 * for each stage making up the _cascade, all classifiers
 * are evaluated using the evalWeakClassifier function
 */

int runCascadeClassifier(MyCascade *_cascade, MyPoint pt) {
	


	int t_index = 0;		//index of features array
	int w_index = 0;		//index of weights array
	int r_index = 0;		//index of rects array


	int i, j;

	int stage_sum;

	////////////////////////////////////////////////////

	int win_w = _cascade->orig_window_size.height;
	int win_h = _cascade->orig_window_size.width;


	int window_area = win_w * win_h;
	int cascade_sum_width = _cascade->sum->width;
	int cascade_sumsq_width = _cascade->sqsum->width;

	int variance;
	uint32_t std;					//standard deviation

	//corners of the detection window
	int c0, c1, c2, c3;
	int cq0, cq1, cq2, cq3;


	//angles of the detection window in relation to the integral image
	c0 = ((pt.y)*cascade_sum_width) + pt.x;
	c1 = ((pt.y)*cascade_sum_width) + pt.x + (win_w - 1);
	c2 = ((pt.y + (win_h - 1))*cascade_sum_width) + pt.x;
	c3 = ((pt.y + (win_h - 1))*cascade_sum_width) + pt.x + (win_w - 1);

	//angles of the detection window in relation to the square integral image
	cq0 = ((pt.y)*cascade_sumsq_width) + pt.x;
	cq1 = ((pt.y)*cascade_sumsq_width) + pt.x + (win_w - 1);
	cq2 = ((pt.y + (win_h - 1))*cascade_sumsq_width) + pt.x;
	cq3 = ((pt.y + (win_h - 1))*cascade_sumsq_width) + pt.x + (win_w - 1);
	/*
	Note: the indices are the same since the dimensions of the integral image and the square integral image are the same
	*/

	//detection window area in relation to the integral image and the square integral image

	uint32_t i_s = _cascade->sum->data[c0] + _cascade->sum->data[c3] - _cascade->sum->data[c1] - _cascade->sum->data[c2];
	uint32_t i_sq = _cascade->sqsum->data[cq0] + _cascade->sqsum->data[cq3] - _cascade->sqsum->data[cq1] - _cascade->sqsum->data[cq2];
	
	/*
	 * CALCULATION OF THE VARIANCE AND STANDARD DEVIATION OF THE SUB-WINDOW ACCORDING TO THE FORMULAS USED IN THE REFERENCE REPOS
	 */

	//uint32_t variance = (i_sq * window_area) - (i_s * i_s);
	//uint32_t std = sqrt(variance);


	/*
	 * CALCULATION OF THE VARIANCE ACCORDING TO THE FORMULAS FOUND AT THE FOLLOWING SITES:
	 * https://en.wikipedia.org/wiki/Summed-area_table#cite_note-Phan-April2012-7
	 * https://theailearner.com/tag/standard-deviation-integral-image-opencv/
	 */

	variance =((i_sq) - (((i_s * i_s)) / (window_area)));
	
	if (variance > 0) {
		variance = variance/(window_area);
		//std = (uint32_t)sqrtf(variance);
	}
	else {
		variance = 1;
	}
	//int core_id=pi_core_id();
	//printf("variance : %d\nstd : %u\n\n", variance, std);
	//condizione ? condizione_vera_esito : condizione_falsa_esito 

	////////////////////////////////////////////////////

	//send the shifted window through the cascade
	for (i = 0; i < _cascade->n_stages; i++) {
		stage_sum = 0;



		for (j = 0; j < _cascade->stages_array[i];j++) {


        // if(core_id==0){
		// 	pi_perf_conf( 1 << PI_PERF_CYCLES);
		// 	pi_perf_cl_start();
		// 	pi_perf_cl_reset();
		// 	}
	        
			stage_sum += evalWeakClassifier(_cascade, variance, pt.x, pt.y, w_index, r_index, t_index); 
        	// if(core_id==0){
			// pi_perf_cl_stop();
        	// //instr_cnt += pi_perf_cl_read(PI_PERF_INSTR);
        	// cycles_cnt += pi_perf_cl_read(PI_PERF_CYCLES);
    	   	// // load_ext += pi_perf_cl_read(PI_PERF_LD_EXT);
        	// // cache_miss += pi_perf_cl_read(PI_PERF_IMISS);
			// }
			w_index += _cascade->num_rectangles_array[t_index];
			r_index += _cascade->num_rectangles_array[t_index] * 4;
			t_index ++;
		}

		//if the sum is below the stage threshold, non object were detected
		if (5*stage_sum < _cascade->stages_thresh_array[i]) {
			
			return 0;
		}
		else
			#ifdef DEBUG
				if (stage_sum!=evalWeakClassifierCheck[count]){
					error_evalWClassifier++;
				}
			#endif
			count ++;
			
	}
	return 1;
}

/*
 * processes the current level in the image pyramid
 */

void processCurrentScale(MyCascade *_cascade, float _factor, MyIntImage *_sum, MyIntImage *_sqsum, MyPoint *final_offset) {

	int x, y, x2, y2;
	int step;
	int result;

	//assigning integral images to the struct cascade
	_cascade->sum = _sum;
	_cascade->sqsum = _sqsum;

	

	int total_rows = _cascade->sum->height;
	int total_colums = _cascade->sum->width;
	//printf("cascadesum\nx %d,y%d\n\n\n",total_rows,total_colums);

	y2 = total_rows - _cascade->orig_window_size.height;
	x2 = total_colums - _cascade->orig_window_size.width;
	//printf("cascade orig window size %d, %d",_cascade->orig_window_size.height,_cascade->orig_window_size.width);
	//printf("X2 %d,Y2 %d\n",x2,y2);
	step = 1;

	PCS_Params PCS_params = {_cascade, y2, x2, _factor, final_offset};

	pi_cl_team_fork(NUM_CORES, PCS_cluster, &PCS_params);

	// //shfit the filter window over the image
	// for (y = 0; y < y2; y += step) {
	// 	for (x = 0; x < x2; x += step) {

	// 		MyPoint p = {x,y};
	// 		
	// 		result = runCascadeClassifier(_cascade, p);
	// 		pi_perf_stop();
	// 	    instr_cnt += pi_perf_read(PI_PERF_INSTR);
	// 	    cycles_cnt += pi_perf_read(PI_PERF_CYCLES);
	// 		#ifdef DEBUG
	// 		if (result!=runCascadeClassifierCheck[runCascadeClassifier_counter])
	// 			error_runCascadeClassifier++;
	// 		runCascadeClassifier_counter++;
	// 		#endif

	// 		if (result > 0) {
	// 		MyRect r = {round(x*_factor), round(y*_factor), round(_cascade->orig_window_size.width*_factor), round(_cascade->orig_window_size.width*_factor)};	
	// 		push(r);
	// 		}
	// 	}
	// }
}

void detectInsects (MyImage *_nneighbor, MyIntImage *_intimage,MyIntImage *_intimagesq, MyCascade *cascade, float scaleFactor, MyPoint *final_offset) {
	
	// pi_perf_conf( 1 << PI_PERF_CYCLES | 1 << PI_PERF_INSTR | 1 << PI_PERF_LD_EXT | 1<< PI_PERF_IMISS); 
	// gap_cl_starttimer();
	// gap_cl_resethwtimer();
	
	// instr_cnt = 0;
	uint32_t cycles_cnt = 0;
	// load_ext = 0;
	// cache_miss = 0;
	//scaling factor

	
    setSumImage(_nneighbor->width, _nneighbor->height, _intimage);
	//printf("dimensioni immagine integrale %d, %d\n",_intimage->width, _intimage->height);
    setSumImage(_nneighbor->width, _nneighbor->height, _intimagesq);
	//printf("dimensioni immagine integrale quadrata %d, %d\n",_intimagesq->width, _intimagesq->height);
	integralImages(_nneighbor, _intimage, _intimagesq);
	processCurrentScale(cascade, scaleFactor, _intimage, _intimagesq, final_offset);
	
    	// float cpi = (float) cycles_cnt/instr_cnt;
    	//printf("\nfactor: %f\nCLOCK CYCLES: %d\n\n",factor,cycles_cnt);
		//printf("%f\t%d\n", factor, invoked_run_cascade_class);
		invoked_run_cascade_class = 0;
		instr_cnt = 0;
		cycles_cnt = 0;
		load_ext = 0;
		cache_miss = 0;
		count = 0;
		//printf("%f = scale factor\n", factor);
		#ifdef DEBUG
			printf("\n\n\nSCALE FACTOR := %f\nError Integral Image := %d\nError Integral Image Squared := %d\nError Nearest Neighbor Fucntion := %d\n",scaleFactor,error_integral,error_sq_integral,error_NN);
			printf("Error Run Cascade Classifier := %d\n",error_runCascadeClassifier);
			printf("Error Evaluate Weak Classifier := %d\n\n====================================================================\n",error_evalWClassifier);

			error_integral=error_NN=error_sq_integral=0;
		#endif
	

    
}

/*load cascade struct and rispectives arrays in the l1*/

void load_cl_l1_cascade(MyCascade *l2_cascade){

    
    l1_cascade.n_stages = l2_cascade->n_stages;
    l1_cascade.total_nodes = l2_cascade->total_nodes;
    l1_cascade.total_rectangles = l2_cascade->total_rectangles;
    l1_cascade.scale = l2_cascade->scale;
    l1_cascade.orig_window_size = l2_cascade->orig_window_size;
    l1_cascade.sum = l2_cascade->sum;
    l1_cascade.sqsum = l2_cascade->sqsum;

    unsigned int i = 0;

    // load insect_stages_array on l1

    for(i = 0; i < (sizeof(insect_stages_array)/sizeof(uint8_t)); i++){
        l1_insect_stages_array[i] = insect_stages_array[i];
    }

    l1_cascade.stages_array = l1_insect_stages_array;
       
    //load insect_stages_thresh_array on l1
    for(i = 0; i < (sizeof(insect_stages_thresh_array)/sizeof(int16_t)); i++){
        l1_insect_stages_thresh_array[i] = insect_stages_thresh_array[i];
    }
    l1_cascade.stages_thresh_array = l1_insect_stages_thresh_array;

    //load insect_node_thresh_array on l1
    for(i = 0; i < (sizeof(insect_node_thresh_array)/sizeof(int16_t)); i++){
        l1_insect_node_thresh_array[i] = insect_node_thresh_array[i];
    }
    l1_cascade.tree_tresh_array = l1_insect_node_thresh_array;
   

    //load insect_alpha1_array on l1
    for(i = 0; i < (sizeof(insect_alpha1_array)/sizeof(int16_t)); i++){
        l1_insect_alpha1_array[i] = insect_alpha1_array[i];
    }
    l1_cascade.alpha1_array = l1_insect_alpha1_array;
  

    //load insect_alpha2_array on l1
    for(i = 0; i < (sizeof(insect_alpha2_array)/sizeof(int16_t)); i++){
        l1_insect_alpha2_array[i] = insect_alpha2_array[i];
    }
    l1_cascade.alpha2_array = l1_insect_alpha2_array;


    //load insect_num_rectangles_array on l1
    for(i = 0; i < (sizeof(insect_num_rectangles_array)/sizeof(int8_t)); i++){
        l1_insect_num_rectangles_array[i] = insect_num_rectangles_array[i];
    }
    l1_cascade.num_rectangles_array = l1_insect_num_rectangles_array;

    //load insect_weights_array on l1
    for(i = 0; i < (sizeof(insect_weights_array)/sizeof(int8_t)); i++){
        l1_insect_weights_array[i] = insect_weights_array[i];
    }
    l1_cascade.weights_array = l1_insect_weights_array;
    
    //load insect_rectangles_array on l1
    for(i = 0; i < (sizeof(insect_rectangles_array)/sizeof(int8_t)); i++){
        l1_insect_rectangles_array[i] = insect_rectangles_array[i];
    }
    l1_cascade.rectangles_array = l1_insect_rectangles_array;


}

/*
dumping results into a .txt file (x,y,width,height) of the bounding box
*/
//  void dumpResult(MyRect results,FILE* fp, int top){ 
  
//   int i;

//   fp=fopen("detections.txt", "w+");


//   for (i=0;i<=top;i++)
// 	fprintf(fp,"%d %d %d %d\n", detections[i].x,detections[i].y, detections[i].width,detections[i].height);

//   fclose(fp);
//  }
/***************************End User Code*********************************/


