
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 * Modified by Luca Bompani luca.bompani5@unibo.it
 * Copyright (C) University of Bologna, Italy 
 * 
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 * 
 */


/* Autotiler includes. */

#include "Mobv3SSDLiteKernels.h"
#include "ResizeBasicKernels.h"
#include "modelInfos.h"
#include "gaplib/ImgIO.h"
#include "bsp/bsp.h"
#include "bsp/ram.h"
// #include "bsp/display/ili9341.h"


#define WIDTH 320
#define HEIGHT 240

#define CAMERA_WIDTH 320
#define CAMERA_HEIGHT 240
#define CAMERA_MINDIM 240

#ifdef DISPLAY
#define MAX_DISPLAY_WIDTH 320
#define MAX_DISPLAY_HEIGHT 240
#define SCORE_THRESHOLD 50
#else
#define MAX_DISPLAY_WIDTH WIDTH
#define MAX_DISPLAY_HEIGHT HEIGHT
#define SCORE_THRESHOLD 0.3*128
#endif
#define PERF 1
struct pi_device DefaultRam;
AT_HYPERFLASH_EXT_ADDR_TYPE Mobv3SSDLite_L3_Flash = 0;
AT_HYPERFLASH_EXT_ADDR_TYPE Mobv3SSDLite_L3_PrivilegedFlash = 0;

/* Global Buffers and Variables */
static L2_MEM uint32_t RamImageBuffer;
static L2_MEM short int OutputBoxes[40];
static L2_MEM signed char OutputClasses[10];
static L2_MEM signed char OutputScores[10];
static L2_MEM char labels[2][50];
static L2_MEM unsigned short colors[5] = {0x001F, 0x07E0, 0x07FF, 0xF800, 0xFFE0};
static L2_MEM char colors_names[5][6] = {"blue", "green", "cyan", "red", "yellow"};

struct pi_device ram;
// static pi_buffer_t buffer;
// struct pi_device gpio;
// pi_gpio_e gpio_out = PI_GPIO_A17_PAD_31_B11;
// int toggle=0;

// static void Resize(KerResize_ArgT *KerArg)
// {
//     AT_FORK(gap_ncore(), (void *) KerResizeBilinear, (void *) KerArg);
// }


static void cluster()
{
    #ifdef PERF
    pi_perf_conf( 1 << PI_PERF_CYCLES);
    pi_perf_cl_start();
    pi_perf_cl_reset();
    #endif
    printf("e questaltro ?\n");
    Mobv3SSDLiteCNN(Mobv3SSDLite_L2_Memory_Dyn,Mobv3SSDLite_L2_Memory_Dyn,Mobv3SSDLite_L2_Memory_Dyn,Mobv3SSDLite_L2_Memory_Dyn, Mobv3SSDLite_L2_Memory_Dyn,Mobv3SSDLite_L2_Memory_Dyn);
}

int test_ssd_mobilenet(void)
{
    // printf("Entering main controller\n");

    // /* Frequency Settings: defined in the Makefile */
    // int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    // int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    // // int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 370*1000*1000);
    // if (cur_fc_freq == -1 || cur_cl_freq == -1 )//|| cur_pe_freq == -1
    // {
    //     printf("Error changing frequency !\nTest failed...\n");
    //     pmsis_exit(-4);
    // }
    // printf("FC Frequency as %d Hz, CL Frequency = %d Hz\n", //, PERIIPH Frequency = %d Hz
    //         pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL));//, pi_freq_get(PI_FREQ_DOMAIN_PERIPH)

    /* Configure And open cluster. */
    PMU_set_voltage(1000, 0);
    pi_time_wait_us(10000);
    PMU_set_voltage(1100, 0);
    pi_time_wait_us(10000);
    PMU_set_voltage(1200, 0);
    PMU_set_voltage(1200, 0);
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_FC, 250000000);
    pi_time_wait_us(10000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, 175000000);
    pi_time_wait_us(10000);
    //cluster inizialization
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
 
    
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }

	struct pi_default_ram_conf default_conf;

    
    pi_default_ram_conf_init(&default_conf);
    default_conf.baudrate=250000000;
	printf("two\n");
    pi_open_from_conf(&DefaultRam, (void *) &default_conf);
	printf("three\n");
    if (pi_ram_open(&DefaultRam))
    {
        printf("Error ram open !\n");
        pmsis_exit(-3);
    }

 	printf("baudrate %d \n",default_conf.baudrate);

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = Mobv3SSDLiteCNN_Construct();
	printf("finished constructing\n");
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file ssd_mobilenetKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }
    unsigned char *L2ImageBuffer = Mobv3SSDLite_L2_Memory_Dyn;

    


    gap_fc_starttimer();
    gap_fc_resethwtimer();
    struct pi_device gpio;
    pi_gpio_e gpio_out = PI_GPIO_A17_PAD_31_B11;
    int toggle=0;
    // ------------------------- GPIO Setup for Gapazzo -------------------------
        int32_t gpio_errors = 0;
        struct pi_gpio_conf gpio_conf = {0};
        pi_gpio_conf_init(&gpio_conf);
        pi_open_from_conf(&gpio, &gpio_conf);
        gpio_errors = pi_gpio_open(&gpio);
        if(gpio_errors) {
        printf("Failed to initialize GPIO!\n");

        pmsis_exit(gpio_errors);
        }
        printf("GPIO inititalized\n");

        pi_gpio_flags_e cfg_flags = PI_GPIO_OUTPUT|PI_GPIO_PULL_DISABLE|PI_GPIO_DRIVE_STRENGTH_LOW;
        pi_gpio_pin_configure(&gpio, gpio_out, cfg_flags);
    // ----------------------- END: GPIO Setup for Gapazzo -----------------------
    while(1) {
		memset(L2ImageBuffer,1,320*240*3);


        
       
        

        // start = pi_time_get_us();
        struct pi_cluster_task task;
        // toggle ^= 1;
	    // pi_gpio_pin_write(&gpio, gpio_out, toggle);
        pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
        task.stack_size = STACK_SIZE+1792;

        task.slave_stack_size=1256;

        toggle = 1;
        pi_gpio_pin_write(&gpio, gpio_out, toggle);
        pi_cluster_send_task_to_cl(&cluster_dev, &task);
        toggle = 0;
        pi_gpio_pin_write(&gpio, gpio_out, toggle);
      
        
        break;

    }

#ifdef PERF
    unsigned int TotalCycles = 0, TotalOper = 0;
    printf("\n");
    for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
    }
    for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
    }
    printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
printf("\n");
#endif

    

    printf("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL YOLOX_nano Example ***\n\n");
    pmsis_kickoff((void*)(test_ssd_mobilenet));
    return 0;
}
