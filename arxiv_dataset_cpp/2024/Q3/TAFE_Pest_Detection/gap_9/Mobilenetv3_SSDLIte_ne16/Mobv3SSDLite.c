/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 * Modified by Luca Crupi luca.crupi@supsi.ch
 * Copyright (C) Dalle Molle Institute on Artifical Intelligence, Switzerland
 * 
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 * 
 */


/* Autotiler includes. */
#include "Mobv3SSDLite.h"
#include "Mobv3SSDLiteKernels.h"
#include "gaplib/fs_switch.h"

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

unsigned int GPIOs = 89;

#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs,x) // macro per accendere o spegnere fuori dal main 


AT_DEFAULTFLASH_EXT_ADDR_TYPE Mobv3SSDLite_L3_Flash = 0;


/* Inputs */
/* Outputs */

/* Copy inputs function */
void copy_inputs() {
}

/* Copy outputs function */
void write_outputs() {
    switch_fs_t fs;
    __FS_INIT(fs);

    /* Reading from file Output_1 */
    void *File_Output_1;
    int ret_Output_1 = 0;
    #ifdef __EMUL__
    File_Output_1 = __OPEN_WRITE(fs, "../Output_1.bin");
    #else
    File_Output_1 = __OPEN_WRITE(fs, "../Output_1.bin");
    #endif
    ret_Output_1 = __WRITE(File_Output_1, Output_1, 2574*sizeof(unsigned char));
    __CLOSE(File_Output_1);

    /* Reading from file Output_2 */
    void *File_Output_2;
    int ret_Output_2 = 0;
    #ifdef __EMUL__
    File_Output_2 = __OPEN_WRITE(fs, "../Output_2.bin");
    #else
    File_Output_2 = __OPEN_WRITE(fs, "../Output_2.bin");
    #endif
    ret_Output_2 = __WRITE(File_Output_2, Output_2, 2574*sizeof(unsigned char));
    __CLOSE(File_Output_2);

    /* Reading from file Output_3 */
    void *File_Output_3;
    int ret_Output_3 = 0;
    #ifdef __EMUL__
    File_Output_3 = __OPEN_WRITE(fs, "../Output_3.bin");
    #else
    File_Output_3 = __OPEN_WRITE(fs, "../Output_3.bin");
    #endif
    ret_Output_3 = __WRITE(File_Output_3, Output_3, 2574*sizeof(unsigned char));
    __CLOSE(File_Output_3);

    /* Reading from file Output_4 */
    void *File_Output_4;
    int ret_Output_4 = 0;
    #ifdef __EMUL__
    File_Output_4 = __OPEN_WRITE(fs, "../Output_4.bin");
    #else
    File_Output_4 = __OPEN_WRITE(fs, "../Output_4.bin");
    #endif
    ret_Output_4 = __WRITE(File_Output_4, Output_4, 2574*sizeof(unsigned char));
    __CLOSE(File_Output_4);

    /* Reading from file Output_5 */
    void *File_Output_5;
    int ret_Output_5 = 0;
    #ifdef __EMUL__
    File_Output_5 = __OPEN_WRITE(fs, "../Output_5.bin");
    #else
    File_Output_5 = __OPEN_WRITE(fs, "../Output_5.bin");
    #endif
    ret_Output_5 = __WRITE(File_Output_5, Output_5, 79794*sizeof(signed short));
    __CLOSE(File_Output_5);

    __FS_DEINIT(fs);
}


static void cluster()
{


    Mobv3SSDLiteCNN();
    //printf("Runner completed\n");

}

int test_Mobv3SSDLite(void)
{
    printf("Entering main controller\n");


    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    

    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP,650);
    pi_time_wait_us(100000);



    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP,650);
    pi_time_wait_us(100000);



#ifndef __EMUL__
    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;

    cl_conf.id = 0; /* Set cluster ID. */
                    // Enable the special icache for the master core
    cl_conf.icache_conf = PI_CLUSTER_MASTER_CORE_ICACHE_ENABLE |
                    // Enable the prefetch for all the cores, it's a 9bits mask (from bit 2 to bit 10), each bit correspond to 1 core
                    PI_CLUSTER_ICACHE_PREFETCH_ENABLE |
                    // Enable the icache for all the cores
                    PI_CLUSTER_ICACHE_ENABLE;

    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }


    





    /* Frequency Settings: defined in the Makefile */
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
	printf("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

#endif
    

    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = Mobv3SSDLiteCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file Mobv3SSDLiteKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }
    
    
    
    

    /*
     * Put here Your input settings
     */
    copy_inputs();

    printf("Call cluster\n");
#ifndef __EMUL__
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);



    pi_pad_function_set(GPIOs, 1);
    pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
    pi_gpio_pin_write(GPIOs, 0);
    WRITE_GPIO(0); //to zer
    
    printf("Recording trace initialized on GPIO %u.\n", GPIOs);
    pi_time_wait_us(100000);

    WRITE_GPIO(1);



    pi_cluster_send_task_to_cl(&cluster_dev, &task);

    WRITE_GPIO(0);

#else
    cluster();
#endif

    write_outputs();

    

    Mobv3SSDLiteCNN_Destruct();

#ifdef PERF
    {
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
      }
      printf("\n");
      printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
      printf("\n");
    }
#endif

    printf("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{

    
    
    printf("\n\n\t *** NNTOOL Mobv3SSDLite Example ***\n\n");
    test_Mobv3SSDLite();
    return 0;
}
