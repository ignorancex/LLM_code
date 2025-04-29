// ############################################################################# 
// Device tree content
// Generation time : 28/11/2023 18:09:14 
//
// THIS IS A GENERATED FILE, DO NOT EDIT !
//
// This file is the static output of the device tree. It aims to define all 
// available features from the board you are using. It is generated from the 
// device tree generator python script : dtGenerator.py.
// #############################################################################

#include "dt.h"

// aps256xxn *******************************************************************

pi_aps256xxn_conf_t aps256xxn_conf = {
	.baudrate  = DT_APS256XXN_BAUDRATE,
	.ram_size  = DT_APS256XXN_RAM_SIZE,
	.ram_start = DT_APS256XXN_RAM_START,
	.spi_cs    = DT_APS256XXN_SPI_CS,
	.spi_itf   = DT_APS256XXN_SPI_ITF,
};

pi_aps256xxn_data_t aps256xxn_data;

pi_device_t aps256xxn_dev = {
	.config = &aps256xxn_conf,
	.data   = &aps256xxn_data,
	.api    = (pi_device_api_t *) &aps256xxn_api
};

// mx25u51245g *****************************************************************

pi_mx25u51245g_conf_t mx25u51245g_conf = {
	.baudrate    = DT_MX25U51245G_BAUDRATE,
	.xip_en      = DT_MX25U51245G_XIP_EN,
	.sector_size = DT_MX25U51245G_SECTOR_SIZE,
	.size        = DT_MX25U51245G_SIZE,
	.spi_cs      = DT_MX25U51245G_SPI_CS,
	.spi_itf     = DT_MX25U51245G_SPI_ITF,
};

pi_mx25u51245g_data_t mx25u51245g_data;

pi_device_t mx25u51245g_dev = {
	.config = &mx25u51245g_conf,
	.data   = &mx25u51245g_data,
	.api    = (pi_device_api_t *) &mx25u51245g_api
};

// Devices API *****************************************************************

pi_err_t pi_open(pi_device_e device_enum, pi_device_t** device_ptr)
{
	pi_err_t err = PI_OK;
	switch (device_enum)
	{
		case PI_RAM_APS256XXN:
		case PI_RAM_DEFAULT :
		{
			*device_ptr = DEVICE_PTR_RAM_APS256XXN;
			break;
		}
		case PI_FLASH_MX25U51245G:
		case PI_FLASH_DEFAULT :
		{
			*device_ptr = DEVICE_PTR_FLASH_MX25U51245G;
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
		case PI_RAM_APS256XXN:
		case PI_RAM_DEFAULT :
		{
			device_ptr = DEVICE_PTR_RAM_APS256XXN;
			break;
		}
		case PI_FLASH_MX25U51245G:
		case PI_FLASH_DEFAULT :
		{
			device_ptr = DEVICE_PTR_FLASH_MX25U51245G;
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

