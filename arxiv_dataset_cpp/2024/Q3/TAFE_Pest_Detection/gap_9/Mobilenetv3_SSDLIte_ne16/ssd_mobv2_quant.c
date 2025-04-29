
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "ssd_mobv2_quant.h"
#include "ssd_mobv2_quantKernels.h"
#include "gaplib/fs_switch.h"
#include "measurments_utils.h"

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif



AT_DEFAULTFLASH_EXT_ADDR_TYPE ssd_mobv2_quant_L3_Flash = 0;



/* Inputs */
/* Outputs */
L2_MEM short int Output_1[40];
L2_MEM signed char Output_2[10];
L2_MEM signed char Output_3[10];

/* Copy inputs function */
void copy_inputs() {
    switch_fs_t fs;
    __FS_INIT(fs);

    /* Reading from file Input_1 */
    void *File_Input_1;
    int ret_Input_1 = 0;
    #ifdef __EMUL__
    File_Input_1 = __OPEN_READ(fs, "../Input_1.bin");
    #else
    File_Input_1 = __OPEN_READ(fs, "../Input_1.bin");
    #endif
    ret_Input_1 = __READ(File_Input_1, Input_1, 270000*sizeof(unsigned char));
    __CLOSE(File_Input_1);

    __FS_DEINIT(fs);
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
    ret_Output_1 = __WRITE(File_Output_1, Output_1, 40*sizeof(signed short));
    __CLOSE(File_Output_1);

    /* Reading from file Output_2 */
    void *File_Output_2;
    int ret_Output_2 = 0;
    #ifdef __EMUL__
    File_Output_2 = __OPEN_WRITE(fs, "../Output_2.bin");
    #else
    File_Output_2 = __OPEN_WRITE(fs, "../Output_2.bin");
    #endif
    ret_Output_2 = __WRITE(File_Output_2, Output_2, 10*sizeof(signed char));
    __CLOSE(File_Output_2);

    /* Reading from file Output_3 */
    void *File_Output_3;
    int ret_Output_3 = 0;
    #ifdef __EMUL__
    File_Output_3 = __OPEN_WRITE(fs, "../Output_3.bin");
    #else
    File_Output_3 = __OPEN_WRITE(fs, "../Output_3.bin");
    #endif
    ret_Output_3 = __WRITE(File_Output_3, Output_3, 10*sizeof(signed char));
    __CLOSE(File_Output_3);

    __FS_DEINIT(fs);
}


static void cluster()
{
    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

GPIO_HIGH();
    ssd_mobv2_quantCNN(Output_1, Output_2, Output_3);
GPIO_LOW();
    printf("Runner completed\n");

}

int test_ssd_mobv2_quant(void)
{
    printf("Entering main controller\n");

    /* Configure And open cluster. */
    OPEN_GPIO_MEAS();
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
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
	printf("FC Frequency = %d Hz CL Frequency = %d Hz PERIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

	#ifdef VOLTAGE
	pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
	pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
	#endif
	printf("Voltage: %dmV\n", pi_pmu_voltage_get(PI_PMU_VOLTAGE_DOMAIN_CHIP));

    
    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = ssd_mobv2_quantCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
        pmsis_exit(-6);
    }
    

    {
        pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_ACTIVE_CYCLES);
        gap_fc_starttimer();
        gap_fc_resethwtimer();
        int start = gap_fc_readhwtimer();
        struct pi_cluster_task task_ctor;
        pi_cluster_task(&task_ctor, (void (*)(void *)) ssd_mobv2_quantCNN_ConstructCluster, NULL);
        pi_cluster_send_task_to_cl(&cluster_dev, &task_ctor);
        int elapsed = gap_fc_readhwtimer() - start;
        printf("L1 Promotion copy took %d FC Cycles\n", elapsed);
    }

    /*
     * Put here Your input settings
     */
    copy_inputs();
    printf("Call cluster\n");

    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    pi_cluster_send_task_to_cl(&cluster_dev, &task);

    write_outputs();
    
    ssd_mobv2_quantCNN_Destruct();
#ifdef PERF
	{
		unsigned long long int TotalCycles = 0, TotalOper = 0;
		printf("\n");
		for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
		}
		for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
		}
		printf("\n");
		printf("%45s: Cycles: %12llu, Cyc%%: 100.0%%, Operations: %12llu, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
		printf("\n");
	}
#endif

    printf("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL ssd_mobv2_quant Example ***\n\n");
    test_ssd_mobv2_quant();
    return 0;
}
