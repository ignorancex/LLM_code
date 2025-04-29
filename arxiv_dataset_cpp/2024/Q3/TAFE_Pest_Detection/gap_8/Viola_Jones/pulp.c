 /* Copyright (C) 2023-2024 University of Bologna, Italy.
 *  Created on: Apr 21, 2022
 *      Author: olmo baldoni
 *      email:  olmo.baldoni@studio.unibo.it
 *  Modified by 
 *          Carlo Montanari calo.montanari4@studio.unibo.it
*/


#include "pmsis.h"
/* PMSIS includes. */


/* PMSIS BSP includes. */
#include "bsp/bsp.h"
#include "bsp/camera.h"

/* Demo includes. */
//#include "gaplib/ImgIO.h"

#include "haar.h"
#include "image.h"
#include "cluster.h"

#include <math.h>



int top = -1;


/*
 * counter for the number of times any stage is passed
 */

int count = 0;
#define CAMERA_WIDTH        324
#define CAMERA_HEIGHT       244
#define         HIMAX_GRP_PARAM_HOLD      0x0104

/

void my_cluster(void*args){
    cluster_entry(args);


}

int main()
{   
    // pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP,800);    
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, 250*1000*1000); 
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, 175*1000*1000);
    struct pi_cluster_task cluster_task = {0};
    struct pi_device cluster_dev = {0};
    struct pi_cluster_conf conf;
    
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
    // static struct pi_device cam;

    static char imgName[50];
    static uint32_t idx = 0;


 



    //detection parameters
    float scaleFactor = 1.2;

    MyImage *image;
    image = pi_l2_malloc(sizeof(MyImage));

	
    readImage(image);
	image->height=240;
	image->width=320;
	//Image->data=imgBuff0;
    


    //initialise the struct for the cascade

    MyCascade *cascade;
    cascade = pi_l2_malloc(sizeof(MyCascade));

    //minimum size of the object to be detected. Objects smaller than this size are ignored
    MySize minSize = {10,10};

    //maximum size of the object to be detected. Objects larger than this size are ignored
    MySize maxSize = {30,30};
	
    //loads the cascade contained in cascade.h
    loadCascade(cascade);
    

    //initialize structs//
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
    pi_perf_conf( 1 << PI_PERF_CYCLES);
    pi_perf_fc_start();
    pi_perf_fc_reset();
    uint32_t cles_cnt = pi_perf_fc_read(PI_PERF_CYCLES);

	pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    uint32_t ending=pi_time_get_us();
    cles_cnt = pi_perf_fc_read(PI_PERF_CYCLES) -cles_cnt;
    printf("cicles %u  \n",cles_cnt);
    printf("timing %d \n",ending-starting);
    
    //printf("finito!\n");

    
    int i;

    for(i=0;i<top;i++)
    //     //printf("detection #%d       %d,%d   %d,%d\n\n",i,detections[i].x,detections[i].y,detections[i].height,detections[i].width);
        printf("%d %d %d %d\n",detections[i].x,detections[i].y,detections[i].height,detections[i].width);
    //printf("\nnumber of features: %d\n",feature_global_counter);

    pi_l2_free(image,sizeof(MyImage));
    pi_l2_free(cascade,sizeof(MyCascade));
    pi_l2_free(intimage,sizeof(MyIntImage));
    pi_l2_free(intimagesq,sizeof(MyIntImage));
    pi_l2_free(nneighbor,sizeof(MyImage));
    pi_l2_free(current_tile, sizeof(MyImage));
    
    pi_cluster_close(&cluster_dev);
}
