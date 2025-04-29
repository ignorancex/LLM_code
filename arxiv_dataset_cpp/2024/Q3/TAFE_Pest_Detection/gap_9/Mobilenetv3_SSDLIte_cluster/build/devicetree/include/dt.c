// ############################################################################# 
// Device tree content
// Generation time : 31/01/2024 09:55:10 
//
// THIS IS A GENERATED FILE, DO NOT EDIT !
//
// This file is the static output of the device tree. It aims to define all 
// available features from the board you are using. It is generated from the 
// device tree generator python script : dtGenerator.py.
// #############################################################################

#include "dt.h"

// hyperram ********************************************************************

pi_hyperram_conf_t hyperram_conf = {
	.reserve_addr_0   = DT_HYPERRAM_RESERVE_ADDR_0,
	.baudrate         = DT_HYPERRAM_BAUDRATE,
	.ram_size         = DT_HYPERRAM_RAM_SIZE,
	.ram_start        = DT_HYPERRAM_RAM_START,
	.xip_en           = DT_HYPERRAM_XIP_EN,
	.skip_pads_config = DT_HYPERRAM_SKIP_PADS_CONFIG,
	.hyper_cs         = DT_HYPERRAM_HYPER_CS,
	.hyper_itf        = DT_HYPERRAM_HYPER_ITF,
};

pi_hyperram_data_t hyperram_data;

pi_device_t hyperram_dev = {
	.config = &hyperram_conf,
	.data   = &hyperram_data,
	.api    = (pi_device_api_t *) &hyperram_api
};

// hyperflash ******************************************************************

pi_hyperflash_conf_t hyperflash_conf = {
	.xip_en           = DT_HYPERFLASH_XIP_EN,
	.baudrate         = DT_HYPERFLASH_BAUDRATE,
	.skip_pads_config = DT_HYPERFLASH_SKIP_PADS_CONFIG,
	.hyper_cs         = DT_HYPERFLASH_HYPER_CS,
	.hyper_itf        = DT_HYPERFLASH_HYPER_ITF,
};

pi_hyperflash_data_t hyperflash_data;

pi_device_t hyperflash_dev = {
	.config = &hyperflash_conf,
	.data   = &hyperflash_data,
	.api    = (pi_device_api_t *) &hyperflash_api
};

// mram ************************************************************************

pi_mram_conf_t mram_conf = {
	.itf      = DT_MRAM_ITF,
	.baudrate = DT_MRAM_BAUDRATE,
	.xip_en   = DT_MRAM_XIP_EN,
};

pi_mram_data_t mram_data;

pi_device_t mram_dev = {
	.config = &mram_conf,
	.data   = &mram_data,
	.api    = (pi_device_api_t *) &mram_api
};

// Flash partition v2 memories *************************************************

#if defined(CONFIG_DRIVER_FLASH_PARTITION_V2)
#include <bsp/flash_partition_v2.h>
typedef struct {
    pi_fpv2_device_ref_t ref;
    pi_device_e memory;
} pi_fpv2_memory_ref_t;

#endif
#if defined(CONFIG_DRIVER_FLASH_PARTITION_V2)
pi_fpv2_memory_ref_t fpv2_memory_refs[2] = {
    { .ref = { PI_FPV2_DEV_TYPE_HYPER, 0, 1}, .memory = PI_FLASH_HYPERFLASH},
    { .ref = { PI_FPV2_DEV_TYPE_MRAM, 0, 0}, .memory = PI_FLASH_MRAM},
};
#endif
// Devices FPV2 register *******************************************************

#if defined(CONFIG_DRIVER_FLASH_PARTITION_V2)
pi_err_t pi_register_fpv2_flashs(void)
{
    pi_err_t err = PI_OK;
    for(size_t i = 0; i < sizeof(fpv2_memory_refs)/sizeof(fpv2_memory_refs[0]); i++)
    {
        err = pi_fpv2_device_register(&fpv2_memory_refs[i].ref,
                                      fpv2_memory_refs[i].memory);
        if (PI_OK != err)
        {
            break;
        }
    }
    return err;
}
#endif
// Devices API *****************************************************************

pi_err_t pi_open(pi_device_e device_enum, pi_device_t** device_ptr)
{
	pi_err_t err = PI_OK;
	switch (device_enum)
	{
		case PI_RAM_HYPERRAM:
		case PI_RAM_DEFAULT :
		{
			*device_ptr = DEVICE_PTR_RAM_HYPERRAM;
			break;
		}
		case PI_FLASH_HYPERFLASH:
		case PI_FLASH_DEFAULT :
		{
			*device_ptr = DEVICE_PTR_FLASH_HYPERFLASH;
			break;
		}
		case PI_FLASH_MRAM:
		{
			*device_ptr = DEVICE_PTR_FLASH_MRAM;
			break;
		}
		default:
		{
			err = PI_ERR_NOT_FOUND;
			break;
		}
	}
	if ((device_ptr != NULL) && (err == PI_OK))
	{
		err = (*device_ptr)->api->open(*device_ptr);
	}
	return err;
}

pi_err_t pi_close(pi_device_e device_enum)
{
	pi_err_t err = PI_OK;
	pi_device_t* device_ptr = NULL;

	switch (device_enum)
	{
		case PI_RAM_HYPERRAM:
		case PI_RAM_DEFAULT :
		{
			device_ptr = DEVICE_PTR_RAM_HYPERRAM;
			break;
		}
		case PI_FLASH_HYPERFLASH:
		case PI_FLASH_DEFAULT :
		{
			device_ptr = DEVICE_PTR_FLASH_HYPERFLASH;
			break;
		}
		case PI_FLASH_MRAM:
		{
			device_ptr = DEVICE_PTR_FLASH_MRAM;
			break;
		}
		default:
		{
			err = PI_ERR_NOT_FOUND;
			break;
		}
	}
	if ((device_ptr != NULL) && (err == PI_OK))
	{
		err = (device_ptr)->api->close(device_ptr);
	}
	return err;
}

