 /*Copyright (C) 2023-2024 University of Bologna, Italy.
 *  Created on: Apr 21, 2022
 *      Author: olmo baldoni
 *      email:  olmo.baldoni@studio.unibo.it
 *  Modified by 
 *          Carlo Montanari calo.montanari4@studio.unibo.it
 *          Luca Bompani luca.bompani5@unibo.it
 */


#include "pmsis.h"
/* PMSIS includes. */


/* PMSIS BSP includes. */
#include "bsp/bsp.h"
#include "bsp/camera.h"

/* Demo includes. */
#include "gaplib/ImgIO.h"

#include "haar.h"
#include "image.h"
#include "cluster.h"

#include <math.h>


/*
 * buffer stuff
 */

int top = -1;


/*
 * counter for the number of times any stage is passed
 */
int count = 0;
#define CAMERA_WIDTH        324
#define CAMERA_HEIGHT       244
#define         HIMAX_GRP_PARAM_HOLD      0x0104

// static int32_t open_camera_himax(struct pi_device *device)
// {
//     struct pi_himax_conf cam_conf;
//     pi_himax_conf_init(&cam_conf);

// #ifdef SLICE_MODE
//     cam_conf.roi.slice_en = 1;
//     cam_conf.roi.x = X;
//     cam_conf.roi.y = Y;
//     cam_conf.roi.w = CAMERA_WIDTH;
//     cam_conf.roi.h = CAMERA_HEIGHT;
// #endif

//     pi_open_from_conf(device, &cam_conf);
//     if (pi_camera_open(device))
//     {
//         return -1;
//     }
// 	uint8_t set_value = 0x03;
// 	// pi_camera_reg_set(device, IMG_ORIENTATION, &set_value); // IMG_ORIENTATION = 0101 CURRENTLY NOT WORKING WORKAROUND
	

// 	set_value = 0x0;
// 	// pi_camera_reg_set(device, HIMAX_GRP_PARAM_HOLD, &set_value); 
// 	pi_time_wait_us(100000);

//     /* Let the camera AEG work for 100ms */
//     // pi_camera_control(device, PI_CAMERA_CMD_AEG_INIT, 0);

//     return 0;
// }

void my_cluster(void*args){
    cluster_entry(args);

}

int main()
{   
    //frequency and voltage setting 
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP,800);    
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, 240*1000*1000); 
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, 240*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 240*1000*1000); 

    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP,650);
    struct pi_cluster_task cluster_task = {0};
    struct pi_device cluster_dev = {0};
    struct pi_cluster_conf conf;


    //[OPTIONAL] specify the stack size for slaves cores
    
    cluster_task.slave_stack_size = STACK_SIZE;
    
    // open the cluster
    pi_cluster_conf_init(&conf);
    conf.cc_stack_size=STACK_SIZE;
    pi_open_from_conf(&cluster_dev, &conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("ERROR: Cluster not working\n");
        return -1;
    }
    
    static uint8_t *imgBuff0;
    static struct pi_device cam;

    static char imgName[50];
    static uint32_t idx = 0;


    /*imgBuff0 = (uint8_t *) pmsis_l2_malloc((CAMERA_WIDTH * CAMERA_HEIGHT) * sizeof(uint8_t));
    if (imgBuff0 == NULL)
    {
        printf("Failed to allocate Memory for Image, asking for: %d * %d\n", CAMERA_WIDTH, CAMERA_HEIGHT);
        pmsis_exit(-1);
    }
	*/
    // int errors = open_camera_himax(&cam);
    // if (errors)
    // {
    //     printf("Failed to open camera : %ld\n", errors);
    //     pmsis_exit(-2);
    // }



    //detection parameters
    float scaleFactor = 1.2;

    MyImage *image;
    image = pi_l2_malloc(sizeof(MyImage));

	// pi_camera_control(&cam, PI_CAMERA_CMD_START, 0);
    // pi_camera_capture(&cam, image->data, CAMERA_WIDTH * CAMERA_HEIGHT);
    // pi_camera_control(&cam, PI_CAMERA_CMD_STOP, 0);
    //load jpg input image

	
	// printf("due\n");
    // for(int i =0;i<CAMERA_HEIGHT;i++){
    //   for(int j=0;j<CAMERA_WIDTH;j++){
    //     if (i<240 && j<320){
    //       image->data[idx] = image->data[i*CAMERA_WIDTH+j];
    //     idx+=1;
    //     }
    // };}
	//WriteImageToFile("image.ppm", 320, 240, sizeof(uint8_t), image->data, GRAY_SCALE_IO);
    readImage(image);

	image->height=240;
	image->width=320;
	
    



    //initialise the struct for the cascade

    MyCascade *cascade;
    cascade = pi_l2_malloc(sizeof(MyCascade));

    //minimum size of the object to be detected. Objects smaller than this size are ignored
    MySize minSize = {10,10};

    //maximum size of the object to be detected. Objects larger than this size are ignored
    MySize maxSize = {30,30};
	
    //loads the cascade contained in cascade.h
    loadCascade(cascade);
    

    //initialize structs for image and integral image//
    MyIntImage *intimage;
    MyIntImage *intimagesq;
    MyImage *nneighbor;
    MyImage *current_tile;
    intimage = pi_l2_malloc(sizeof(MyIntImage));
    intimagesq = pi_l2_malloc(sizeof(MyIntImage));
    nneighbor = pi_l2_malloc(sizeof(MyImage));
    current_tile = pi_l2_malloc(sizeof(MyImage));
	
    ClusterParams cluster_params = {image, nneighbor, current_tile, intimage, intimagesq, minSize, maxSize, cascade, scaleFactor};

    uint32_t starting=pi_time_get_us();
    pi_cluster_task(&cluster_task, (void(*)(void*)) my_cluster, &cluster_params);

	pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    uint32_t ending=pi_time_get_us();
    printf("timing %u  \n",ending-starting);
    
    //printf("finito!\n");

    
    int i;
    //printing outputs detections is a global parameter
    for(i=0;i<top;i++)
        printf("%d %d %d %d\n",detections[i].x,detections[i].y,detections[i].height,detections[i].width);
    

    //freeing L2 memory
    pi_l2_free(image,sizeof(MyImage));
    pi_l2_free(cascade,sizeof(MyCascade));
    pi_l2_free(intimage,sizeof(MyIntImage));
    pi_l2_free(intimagesq,sizeof(MyIntImage));
    pi_l2_free(nneighbor,sizeof(MyImage));
    pi_l2_free(current_tile, sizeof(MyImage));
    //closing cluster (free also L1 memory)
    pi_cluster_close(&cluster_dev);
}
