/***************************************************************************************************
**
** FileName:     main.c
** Dependencies: See INCLUDES section
** Processor:	Microchip PIC18F2550. (Possibly usable on other PIC18 or PIC24 USB Microcontrollers)
** Hardware:	Multiple-sensors interface board used to interface various sensors, resistive, inductive, capacitive, analogue, frequency, voltage. 
**				Interface to a PC with a USB connection and/or RS232 / RS485. Configurable digital outputs to trigger alarms and other devices.
**				
** Compiler:  	Microchip C18 (for PIC18) or C30 (for PIC24)
** Inventor/Developer:		David Nuno G. da Silva S. Quelhas. Lisboa, Portugal. Multiple-Sensor Interface.
**							Copyright (c) 2021 David Nuno Quelhas.
** Multiple Sensor Interface, RTU. PIC18F Firmware.
** License: GNU General Public License version 3 , 
**    In accordance with 7-b) and 7-c) of GNU-GPLv3, this license requires to preserve/include prior copyright notices with author names,
**     and also add new additional copyright notices, specifically on start/top of source code files and on 'About'(or similar)
**     dialog/windows at 'user interfaces', on the software/firmware and on any 'modified version' or 'derivative work'.
**
** This is the source file for the Firmware of the Multiple-Sensor Interface. 
**
**  This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as
** published by the Free Software Foundation.                          
**  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
**                                                                        
**  You should have received a copy of the GNU General Public License along with this program; if not, write to the  
** Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.   
**
******************************************************************************************************/


//ATTENTION: Don't exceed about 3500 lines of code (instructions not counting the comments), because this may cause the code to malfunction due to missing program memory,
//				causing the program to not allowing read of USART and not allowing interruptions

/********************************************************************
 File Description:

 Change History:
  Rev   Description
  ----  -----------------------------------------
  1.0   Initial release
  2.1   Updated for simplicity and to use common
                     coding style
  2.7b  Improvements to USBCBSendResume(), to make it easier to use.
  2.9f  Adding new part support
********************************************************************/

#ifndef MAIN_C
#define MAIN_C

/** INCLUDES *******************************************************/
#include "./USB/usb.h"
#include "HardwareProfile.h"
#include "./USB/usb_function_hid.h"

#include "i2c.h"

// Include files for the Multiple-Sensor Interface

#include <adc.h>
#include <EEP.h>
//#include <string.h>
#include <portb.h>
#include <timers.h>


// Include files for USART / serial port -----
//#include <portb.h>
#include <usart.h>
// -----------------------------------


//----------- comparator0 include files -------------
#define USE_OR_MASKS
//#include "ancomp.h"
//--------------------------------------------------

//-------- SensorBoard constants and define macros ---------
#define PIN_DE_RE_MAX485 LATBbits.LATB5
#define LCR_SENSORS_SIGNAL_IN PORTAbits.RA4
#define AN_MUX_A PORTCbits.RC0
#define AN_MUX_B PORTCbits.RC1
#define AN_MUX_C PORTCbits.RC2
/*#define OUT_0 PORTBbits.RB5
#define OUT_1 PORTBbits.RB4
#define OUT_2 PORTBbits.RB3
*/
//#define OUT_0 LATBbits.LATB5	//output OUT_0 will be a message sent by serial port
#define OUT_1 LATBbits.LATB4
#define OUT_2 LATBbits.LATB3

#define PIN_I2C_SDA LATBbits.LATB0
#define PIN_I2C_SCL LATBbits.LATB1


#define GS_ASCII 29  //"group separator" character ASCII code
#define US_ASCII 31  //"unit separator" character ASCII code

#define N_LCR_CHANNELS 6  //number of available channels for LCR sensors
#define N_ADC_CHANNELS 4  //number of available channels for the ADC
#define N_DIGITAL_OUT 3	//number of available digital outputs

#define CH_ZERO 0	// DEBUG CODE

#define USB_REPLY_WITH_INVALID_DATA 0x70
#define SERIAL_REPLY_WITH_INVALID_DATA 0x70

#define bool_func_N_bytes 54	// bool_func_N_bytes/2=(integer), MUST BE DIVISIBLE BY 2	//max. 50 characters for the boolean functions of the outputs, 2 character for ( ), 1 character for '\0', and 1 extra character
#define bool_func_max_length 50	//the actual maximum length/size for bool func is '50', because must have/add '(' at start and ')' at end for ensuring bool function / expression is evaluated correctly, plus '\0' at the end
#define CONST_NEG_ONE -1

#define RECEIVE_DATA_BUFFER_SIZE 64
#define SEND_DATA_BUFFER_SIZE 64

//-------------------------------------------------------------

// ----- Multiple Sensor board EEPROM ADDRS -----
#define EEPROM_ADDR_BOARD_ID 0x01
#define EEPROM_ADDR_CHANNELS_CONFIGS 0x03
#define EEPROM_ADDR_BOOL_FUNCTIONS 0x36
//#define EEPROM_ADDR_SENSORS_CALIB 0x04
//#define EEPROM_ADDR_CHANNELS_CONFIGS 0xB4
//#define EEPROM_ADDR_OUTPUTS_ACTIVE_LEVEL 0xE7
#define EEPROM_ADDR_MODE_CHANNEL_MULTIPLE_OR_SINGLE 0xED

#define TIMEOUT_TICKS_1 50
// ------------------------------------------

// ----- Multiple Sensor board I2C EEPROM ADDRS -----
//#define EEPROM_I2C_WRITE_ADDR 0xA1  //(1010 , chip_select: 000 Write Op: 1) , NOT CORRECT
//#define EEPROM_I2C_READ_ADDR 0xA0  //(1010 , chip_select: 000 Read Op: 0)	, NOT CORRECT
#define EEPROM_I2C_ADDR 0xA0  //(1010 , chip_select: 000 )
#define EEPROM_I2C_SIZE_KBITS 512	//512Kbits <=> 64Kbytes, 24LC512 or 24FC512
//#define SIZE_CALIB_TABLE_BYTES ( EEPROM_I2C_SIZE_KBITS / ( 8*(N_LCR_CHANNELS+N_ADC_CHANNELS) ) );
#define TABLE_HEADER_SIZE_BYTES 128  //size in bytes of the header of each calibration table, where is saved the units of calibration, the mode: multi channel or single channel, etc...
const short SIZE_CALIB_TABLE_BYTES_const = (short) ( (1024.0 *  EEPROM_I2C_SIZE_KBITS ) / ( 8.0*(N_LCR_CHANNELS+N_ADC_CHANNELS) ) );
const short LINES_CALIB_TABLE_const = (short) ( (1024.0 *  EEPROM_I2C_SIZE_KBITS ) / ( TABLE_HEADER_SIZE_BYTES*(N_LCR_CHANNELS+N_ADC_CHANNELS) ) );	//each line has 8 bytes (4 bytes the RAW_value, 4 bytes the measurement) , for 512Kbit EEPROM this is 819 lines for each calib table
#define WAIT_BETWEEN_I2C_WRITE_CMD 5000	//count value for a delay time between consecutive I2C commands
#define CALIB_UNITS_SIZE_STR 20
#define CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE 12		//maximum size/lines used from a calibration table of sensor measurements when device is in multiple channel mode, this restriction is to ensure the measurements process doesn't overload the processing by the PIC18F2550
#define CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE 32		//maximum size/lines used from a calibration table of sensor measurements when device is in single channel mode, this restriction is to ensure the measurements process doesn't overload the processing by the PIC18F2550	
#define ADC_CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE 12	//maximum size/lines used from a calibration table of sensor measurements when device is in multiple channel mode, this restriction is to ensure the measurements process doesn't overload the processing by the PIC18F2550
#define ADC_CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE 32	//maximum size/lines used from a calibration table of sensor measurements when device is in single channel mode, this restriction is to ensure the measurements process doesn't overload the processing by the PIC18F2550
#define COUNTER_CALIB_ROW_NUMBER 240	//row number where are saved the calibration constants for the counter calibration.
							//this means the counter calibration constants(C0,C1,C2,C3) are saved at the end/after each calibration table of sensor measuremnts (the calibration for RAW_value[Hz]/ADC_value[V])
#define NAN 0x7f800000 //define the NAN constant (Not-A-Number)
// --------------------------------------------------


// ----- LCR sensors ModbusRTU register codes -----
#define SELECT_MEASUREMENT_TIME_CODE_ModbusRTU 0x01
#define QUERY_FREQUENCY_MEASUREMENTS_CODE_ModbusRTU 0x10
//#define SAVE_CALIBRATE_ZERO_LOAD_ModbusRTU 0x83
#define QUERY_ADC_MEASUREMENTS_CODE_ModbusRTU 0x16
#define SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU 0x20	//from 0x20 to 0x2A (0x20 + N_LCR_CHANNELS + N_ADC_CHANNELS)
#define SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU 0x30	//from 0x30 to 0x3A (0x30 + N_LCR_CHANNELS + N_ADC_CHANNELS)
#define SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU 0x40	//from 0x40 to 0x4A (0x40 + N_LCR_CHANNELS + N_ADC_CHANNELS)
#define QUERY_MEASUREMENTS_CH_CODE_ModbusRTU 0x87
//#define SAVE_SET_ALL_OUTPUTS_ACTIVE_STATE_ModbusRTU 0x02		//OBSOLETE
//#define SAVE_OR_QUERY_OUTPUT_ACTIVE_LEVEL_CONFIG_ModbusRTU 0x91	//OBSOLETE
#define SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU 0x91
#define QUERY_OUTPUTS_CURRENT_VALUE_CODE_ModbusRTU 0x95
#define SAVE_OR_QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_CODE_ModbusRTU 0x96
#define QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU 0x50
#define RESET_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU 0x58
#define QUERY_COUNTERS_MEASUREMENTS_CH_CODE_ModbusRTU 0x60
#define SET_OR_GET_BOARD_ID_CODE_ModbusRTU 0x98

#define MULTIPLE_SENSOR_FIRMWARE_ERROR_ModbusRTU 0xFF
// ---------------------------------------

/** CONFIGURATION **************************************************/
#if defined(PICDEM_FS_USB)      // Configuration bits for PICDEM FS USB Demo Board (based on PIC18F4550)
		#warning "INFO: PICDEM-FS-USB is defined, setting configuration accordingly"		//information useful for debug purposes
        #pragma config PLLDIV   = 5         // (20 MHz crystal on PICDEM FS USB board)
        #pragma config CPUDIV   = OSC1_PLL2   
        #pragma config USBDIV   = 2         // Clock source from 96MHz PLL/2
        #pragma config FOSC     = HSPLL_HS
        #pragma config FCMEN    = OFF
        #pragma config IESO     = OFF
        #pragma config PWRT     = OFF
        #pragma config BOR      = ON		//enable Brown-out Reset to avoid wrong behavior of the PIC processor during power-up and power-down, like for example overwriting memory with trash
        #pragma config BORV     = 3			//#pragma config BORV = 3	//BORV1:BORV0=11 (2.05V); BORV1:BORV0=10 (2.79V); BORV1:BORV0=01  (4.33V); BORV1:BORV0=11 (4.59V)
        #pragma config VREGEN   = ON      //USB Voltage Regulator
        #pragma config WDT      = OFF
        #pragma config WDTPS    = 32768
        #pragma config MCLRE    = ON
        #pragma config LPT1OSC  = OFF
        #pragma config PBADEN   = OFF
//      #pragma config CCP2MX   = ON
        #pragma config STVREN   = ON
        #pragma config LVP      = OFF
//      #pragma config ICPRT    = OFF       // Dedicated In-Circuit Debug/Programming
        #pragma config XINST    = OFF       // Extended Instruction Set
        #pragma config CP0      = OFF
        #pragma config CP1      = OFF
//      #pragma config CP2      = OFF
//      #pragma config CP3      = OFF
        #pragma config CPB      = OFF
//      #pragma config CPD      = OFF
        #pragma config WRT0     = OFF
        #pragma config WRT1     = OFF
//      #pragma config WRT2     = OFF
//      #pragma config WRT3     = OFF
        #pragma config WRTB     = OFF       // Boot Block Write Protection
        #pragma config WRTC     = OFF
//      #pragma config WRTD     = OFF
        #pragma config EBTR0    = OFF
        #pragma config EBTR1    = OFF
//      #pragma config EBTR2    = OFF
//      #pragma config EBTR3    = OFF
        #pragma config EBTRB    = OFF

#elif defined(PICDEM_FS_USB_K50)
        #pragma config PLLSEL   = PLL3X     // 3X PLL multiplier selected
        #pragma config CFGPLLEN = OFF       // PLL turned on during execution
        #pragma config CPUDIV   = NOCLKDIV  // 1:1 mode (for 48MHz CPU)
        #pragma config LS48MHZ  = SYS48X8   // Clock div / 8 in Low Speed USB mode
        #pragma config FOSC     = INTOSCIO  // HFINTOSC selected at powerup, no clock out
        #pragma config PCLKEN   = OFF       // Primary oscillator driver
        #pragma config FCMEN    = OFF       // Fail safe clock monitor
        #pragma config IESO     = OFF       // Internal/external switchover (two speed startup)
        #pragma config nPWRTEN  = OFF       // Power up timer
        #pragma config BOREN    = SBORDIS   // BOR enabled
        #pragma config nLPBOR   = ON        // Low Power BOR
        #pragma config WDTEN    = SWON      // Watchdog Timer controlled by SWDTEN
        #pragma config WDTPS    = 32768     // WDT postscalar
        #pragma config PBADEN   = OFF       // Port B Digital/Analog Powerup Behavior
        #pragma config SDOMX    = RC7       // SDO function location
        #pragma config LVP      = OFF       // Low voltage programming
        #pragma config MCLRE    = ON        // MCLR function enabled (RE3 disabled)
        #pragma config STVREN   = ON        // Stack overflow reset
        //#pragma config ICPRT  = OFF       // Dedicated ICPORT program/debug pins enable
        #pragma config XINST    = OFF       // Extended instruction set

#elif defined(PIC18F87J50_PIM)				// Configuration bits for PIC18F87J50 FS USB Plug-In Module board
        #pragma config XINST    = OFF   	// Extended instruction set
        #pragma config STVREN   = ON      	// Stack overflow reset
        #pragma config PLLDIV   = 3         // (12 MHz crystal used on this board)
        #pragma config WDTEN    = OFF      	// Watch Dog Timer (WDT)
        #pragma config CP0      = OFF      	// Code protect
        #pragma config CPUDIV   = OSC1      // OSC1 = divide by 1 mode
        #pragma config IESO     = OFF      	// Internal External (clock) Switchover
        #pragma config FCMEN    = OFF      	// Fail Safe Clock Monitor
        #pragma config FOSC     = HSPLL     // Firmware must also set OSCTUNE<PLLEN> to start PLL!
        #pragma config WDTPS    = 32768
//      #pragma config WAIT     = OFF      	// Commented choices are
//      #pragma config BW       = 16      	// only available on the
//      #pragma config MODE     = MM      	// 80 pin devices in the 
//      #pragma config EASHFT   = OFF      	// family.
        #pragma config MSSPMSK  = MSK5
//      #pragma config PMPMX    = DEFAULT
//      #pragma config ECCPMX   = DEFAULT
        #pragma config CCP2MX   = DEFAULT   
        
// Configuration bits for PIC18F97J94 PIM and PIC18F87J94 PIM
#elif defined(PIC18F97J94_PIM) || defined(PIC18F87J94_PIM)
        #pragma config STVREN   = ON      	// Stack overflow reset
        #pragma config XINST    = OFF   	// Extended instruction set
        #pragma config BOREN    = ON        // BOR Enabled
        #pragma config BORV     = 0         // BOR Set to "2.0V" nominal setting
        #pragma config CP0      = OFF      	// Code protect disabled
        #pragma config FOSC     = FRCPLL    // Firmware should also enable active clock tuning for this setting
        #pragma config SOSCSEL  = LOW       // SOSC circuit configured for crystal driver mode
        #pragma config CLKOEN   = OFF       // Disable clock output on RA6
        #pragma config IESO     = OFF      	// Internal External (clock) Switchover
        #pragma config PLLDIV   = NODIV     // 4 MHz input (from 8MHz FRC / 2) provided to PLL circuit
        #pragma config POSCMD   = NONE      // Primary osc disabled, using FRC
        #pragma config FSCM     = CSECMD    // Clock switching enabled, fail safe clock monitor disabled
        #pragma config WPDIS    = WPDIS     // Program memory not write protected
        #pragma config WPCFG    = WPCFGDIS  // Config word page of program memory not write protected
        #pragma config IOL1WAY  = OFF       // IOLOCK can be set/cleared as needed with unlock sequence
        #pragma config LS48MHZ  = SYSX2     // Low Speed USB clock divider
        #pragma config WDTCLK   = LPRC      // WDT always uses INTOSC/LPRC oscillator
        #pragma config WDTEN    = ON        // WDT disabled; SWDTEN can control WDT
        #pragma config WINDIS   = WDTSTD    // Normal non-window mode WDT.
        #pragma config VBTBOR   = OFF       // VBAT BOR disabled
      
#elif defined(PIC18F46J50_PIM) || defined(PIC18F_STARTER_KIT_1) || defined(PIC18F47J53_PIM)
     #pragma config WDTEN = OFF          //WDT disabled (enabled by SWDTEN bit)
     #pragma config PLLDIV = 3           //Divide by 3 (12 MHz oscillator input)
     #pragma config STVREN = ON          //stack overflow/underflow reset enabled
     #pragma config XINST = OFF          //Extended instruction set disabled
     #pragma config CPUDIV = OSC1        //No CPU system clock divide
     #pragma config CP0 = OFF            //Program memory is not code-protected
     #pragma config OSC = HSPLL          //HS oscillator, PLL enabled, HSPLL used by USB
     #pragma config FCMEN = OFF          //Fail-Safe Clock Monitor disabled
     #pragma config IESO = OFF           //Two-Speed Start-up disabled
     #pragma config WDTPS = 32768        //1:32768
     #pragma config DSWDTOSC = INTOSCREF //DSWDT uses INTOSC/INTRC as clock
     #pragma config RTCOSC = T1OSCREF    //RTCC uses T1OSC/T1CKI as clock
     #pragma config DSBOREN = OFF        //Zero-Power BOR disabled in Deep Sleep
     #pragma config DSWDTEN = OFF        //Disabled
     #pragma config DSWDTPS = 8192       //1:8,192 (8.5 seconds)
     #pragma config IOL1WAY = OFF        //IOLOCK bit can be set and cleared
     #pragma config MSSP7B_EN = MSK7     //7 Bit address masking
     #pragma config WPFP = PAGE_1        //Write Protect Program Flash Page 0
     #pragma config WPEND = PAGE_0       //Start protection at page 0
     #pragma config WPCFG = OFF          //Write/Erase last page protect Disabled
     #pragma config WPDIS = OFF          //WPFP[5:0], WPEND, and WPCFG bits ignored 
     #if defined(PIC18F47J53_PIM)
        #pragma config CFGPLLEN = OFF
     #else
        #pragma config T1DIG = ON           //Sec Osc clock source may be selected
        #pragma config LPT1OSC = OFF        //high power Timer1 mode
     #endif
#elif defined(LOW_PIN_COUNT_USB_DEVELOPMENT_KIT)
		#warning "INFO: LOW-PIN-COUNT-USB-DEVELOPMENT-KIT is defined, setting configuration accordingly"
        #pragma config CPUDIV = NOCLKDIV
        #pragma config USBDIV = OFF
        #pragma config FOSC   = HS
        #pragma config PLLEN  = ON
        #pragma config FCMEN  = OFF
        #pragma config IESO   = OFF
        #pragma config PWRTEN = OFF
        #pragma config BOREN  = OFF
        #pragma config BORV   = 30
        #pragma config WDTEN  = OFF
        #pragma config WDTPS  = 32768
        #pragma config MCLRE  = OFF
        #pragma config HFOFST = OFF
        #pragma config STVREN = ON
        #pragma config LVP    = OFF
        #pragma config XINST  = OFF
        #pragma config BBSIZ  = OFF
        #pragma config CP0    = OFF
        #pragma config CP1    = OFF
        #pragma config CPB    = OFF
        #pragma config WRT0   = OFF
        #pragma config WRT1   = OFF
        #pragma config WRTB   = OFF
        #pragma config WRTC   = OFF
        #pragma config EBTR0  = OFF
        #pragma config EBTR1  = OFF
        #pragma config EBTRB  = OFF                                                  // CONFIG7H

#elif	defined(PIC16F1_LPC_USB_DEVELOPMENT_KIT)
    // PIC 16F1459 fuse configuration:
    // Config word 1 (Oscillator configuration)
    // 20Mhz crystal input scaled to 48Mhz and configured for USB operation
    #if defined (USE_INTERNAL_OSC)
#warning Using Internal Oscillator
        __CONFIG(FOSC_INTOSC & WDTE_OFF & PWRTE_OFF & MCLRE_OFF & CP_OFF & BOREN_OFF & CLKOUTEN_ON & IESO_OFF & FCMEN_OFF);
        __CONFIG(WRT_OFF & CPUDIV_NOCLKDIV & USBLSCLK_48MHz & PLLMULT_3x & PLLEN_ENABLED & STVREN_ON &  BORV_LO & LPBOR_OFF & LVP_OFF);
    #else
#warning Using Crystal Oscillator
        __CONFIG(FOSC_HS & WDTE_OFF & PWRTE_OFF & MCLRE_OFF & CP_OFF & BOREN_OFF & CLKOUTEN_ON & IESO_OFF & FCMEN_OFF);
        __CONFIG(WRT_OFF & CPUDIV_NOCLKDIV & USBLSCLK_48MHz & PLLMULT_4x & PLLEN_ENABLED & STVREN_ON &  BORV_LO & LPBOR_OFF & LVP_OFF);
    #endif

#elif defined(EXPLORER_16)
    #if defined(__PIC24FJ256GB110__)
        _CONFIG1( JTAGEN_OFF & GCP_OFF & GWRP_OFF & FWDTEN_OFF & ICS_PGx2) 
        _CONFIG2( PLL_96MHZ_ON & IESO_OFF & FCKSM_CSDCMD & OSCIOFNC_ON & POSCMOD_HS & FNOSC_PRIPLL & PLLDIV_DIV2 & IOL1WAY_ON)
    #elif defined(PIC24FJ256GB210_PIM)
        _CONFIG1(FWDTEN_OFF & ICS_PGx2 & GWRP_OFF & GCP_OFF & JTAGEN_OFF)
        _CONFIG2(POSCMOD_HS & IOL1WAY_ON & OSCIOFNC_ON & FCKSM_CSDCMD & FNOSC_PRIPLL & PLL96MHZ_ON & PLLDIV_DIV2 & IESO_OFF)
    #elif defined(__PIC24FJ64GB004__)
        _CONFIG1(WDTPS_PS1 & FWPSA_PR32 & WINDIS_OFF & FWDTEN_OFF & ICS_PGx1 & GWRP_OFF & GCP_OFF & JTAGEN_OFF)
        _CONFIG2(POSCMOD_HS & I2C1SEL_PRI & IOL1WAY_OFF & OSCIOFNC_ON & FCKSM_CSDCMD & FNOSC_PRIPLL & PLL96MHZ_ON & PLLDIV_DIV2 & IESO_OFF)
        _CONFIG3(WPFP_WPFP0 & SOSCSEL_SOSC & WUTSEL_LEG & WPDIS_WPDIS & WPCFG_WPCFGDIS & WPEND_WPENDMEM)
        _CONFIG4(DSWDTPS_DSWDTPS3 & DSWDTOSC_LPRC & RTCOSC_SOSC & DSBOREN_OFF & DSWDTEN_OFF)
    #elif defined(__32MX460F512L__) || defined(__32MX795F512L__)
        #pragma config UPLLEN   = ON        // USB PLL Enabled
        #pragma config FPLLMUL  = MUL_15        // PLL Multiplier
        #pragma config UPLLIDIV = DIV_2         // USB PLL Input Divider
        #pragma config FPLLIDIV = DIV_2         // PLL Input Divider
        #pragma config FPLLODIV = DIV_1         // PLL Output Divider
        #pragma config FPBDIV   = DIV_1         // Peripheral Clock divisor
        #pragma config FWDTEN   = OFF           // Watchdog Timer
        #pragma config WDTPS    = PS1           // Watchdog Timer Postscale
        #pragma config FCKSM    = CSDCMD        // Clock Switching & Fail Safe Clock Monitor
        #pragma config OSCIOFNC = OFF           // CLKO Enable
        #pragma config POSCMOD  = HS            // Primary Oscillator
        #pragma config IESO     = OFF           // Internal/External Switch-over
        #pragma config FSOSCEN  = OFF           // Secondary Oscillator Enable (KLO was off)
        #pragma config FNOSC    = PRIPLL        // Oscillator Selection
        #pragma config CP       = OFF           // Code Protect
        #pragma config BWP      = OFF           // Boot Flash Write Protect
        #pragma config PWP      = OFF           // Program Flash Write Protect
        #pragma config ICESEL   = ICS_PGx2      // ICE/ICD Comm Channel Select
    #elif defined(__dsPIC33EP512MU810__) || defined (__PIC24EP512GU810__)
        _FOSCSEL(FNOSC_FRC);
        _FOSC(FCKSM_CSECMD & OSCIOFNC_OFF & POSCMD_XT);
        _FWDT(FWDTEN_OFF);
    
    #else
        #error No hardware board defined, see "HardwareProfile.h" and __FILE__
    #endif
#elif defined(PIC24F_STARTER_KIT)
    _CONFIG1( JTAGEN_OFF & GCP_OFF & GWRP_OFF & FWDTEN_OFF & ICS_PGx2) 
    _CONFIG2( PLL_96MHZ_ON & IESO_OFF & FCKSM_CSDCMD & OSCIOFNC_OFF & POSCMOD_HS & FNOSC_PRIPLL & PLLDIV_DIV3 & IOL1WAY_ON)
#elif defined(PIC24FJ256DA210_DEV_BOARD)
    _CONFIG1(FWDTEN_OFF & ICS_PGx2 & GWRP_OFF & GCP_OFF & JTAGEN_OFF)
    _CONFIG2(POSCMOD_HS & IOL1WAY_ON & OSCIOFNC_ON & FCKSM_CSDCMD & FNOSC_PRIPLL & PLL96MHZ_ON & PLLDIV_DIV2 & IESO_OFF)
#elif defined(PIC32_USB_STARTER_KIT)
    #pragma config UPLLEN   = ON        // USB PLL Enabled
    #pragma config FPLLMUL  = MUL_15        // PLL Multiplier
    #pragma config UPLLIDIV = DIV_2         // USB PLL Input Divider
    #pragma config FPLLIDIV = DIV_2         // PLL Input Divider
    #pragma config FPLLODIV = DIV_1         // PLL Output Divider
    #pragma config FPBDIV   = DIV_1         // Peripheral Clock divisor
    #pragma config FWDTEN   = OFF           // Watchdog Timer
    #pragma config WDTPS    = PS1           // Watchdog Timer Postscale
    #pragma config FCKSM    = CSDCMD        // Clock Switching & Fail Safe Clock Monitor
    #pragma config OSCIOFNC = OFF           // CLKO Enable
    #pragma config POSCMOD  = HS            // Primary Oscillator
    #pragma config IESO     = OFF           // Internal/External Switch-over
    #pragma config FSOSCEN  = OFF           // Secondary Oscillator Enable (KLO was off)
    #pragma config FNOSC    = PRIPLL        // Oscillator Selection
    #pragma config CP       = OFF           // Code Protect
    #pragma config BWP      = OFF           // Boot Flash Write Protect
    #pragma config PWP      = OFF           // Program Flash Write Protect
    #pragma config ICESEL   = ICS_PGx2      // ICE/ICD Comm Channel Select
#elif defined(DSPIC33E_USB_STARTER_KIT)
        _FOSCSEL(FNOSC_FRC);
        _FOSC(FCKSM_CSECMD & OSCIOFNC_OFF & POSCMD_XT);
        _FWDT(FWDTEN_OFF);
#elif defined(PIC24FJ64GB502_MICROSTICK)
    _CONFIG1(WDTPS_PS1 & FWPSA_PR32 & WINDIS_OFF & FWDTEN_OFF & ICS_PGx1 & GWRP_OFF & GCP_OFF & JTAGEN_OFF)
    _CONFIG2(I2C1SEL_PRI & IOL1WAY_OFF & FCKSM_CSDCMD & FNOSC_PRIPLL & PLL96MHZ_ON & PLLDIV_DIV2 & IESO_OFF)
    _CONFIG3(WPFP_WPFP0 & SOSCSEL_SOSC & WUTSEL_LEG & WPDIS_WPDIS & WPCFG_WPCFGDIS & WPEND_WPENDMEM)
    _CONFIG4(DSWDTPS_DSWDTPS3 & DSWDTOSC_LPRC & RTCOSC_SOSC & DSBOREN_OFF & DSWDTEN_OFF)
#else
    #error No hardware board defined, see "HardwareProfile.h" and __FILE__
#endif

/** VARIABLES ******************************************************/
#if defined(__18CXX)
    #pragma udata

    //The ReceivedDataBuffer[] and ToSendDataBuffer[] arrays are used as
    //USB packet buffers in this firmware.  Therefore, they must be located in
    //a USB module accessible portion of microcontroller RAM.
    #if defined(__18F14K50) || defined(__18F13K50) || defined(__18LF14K50) || defined(__18LF13K50) 
        #pragma udata USB_VARIABLES=0x260
    #elif defined(__18F2455) || defined(__18F2550) || defined(__18F4455) || defined(__18F4550)\
        || defined(__18F2458) || defined(__18F2553) || defined(__18F4458) || defined(__18F4553)\
        || defined(__18LF24K50) || defined(__18F24K50) || defined(__18LF25K50)\
        || defined(__18F25K50) || defined(__18LF45K50) || defined(__18F45K50)
        #pragma udata USB_VARIABLES=0x500
    #elif defined(__18F4450) || defined(__18F2450)
        #pragma udata USB_VARIABLES=0x480
    #else
        #pragma udata
    #endif
#endif

#if defined(__XC8)
    #if defined(_18F14K50) || defined(_18F13K50) || defined(_18LF14K50) || defined(_18LF13K50)
        #define RX_DATA_BUFFER_ADDRESS @0x260
        #define TX_DATA_BUFFER_ADDRESS @0x2A0
    #elif  defined(_18F2455)   || defined(_18F2550)   || defined(_18F4455)  || defined(_18F4550)\
        || defined(_18F2458)   || defined(_18F2453)   || defined(_18F4558)  || defined(_18F4553)\
        || defined(_18LF24K50) || defined(_18F24K50)  || defined(_18LF25K50)\
        || defined(_18F25K50)  || defined(_18LF45K50) || defined(_18F45K50)
        #define RX_DATA_BUFFER_ADDRESS @0x500
        #define TX_DATA_BUFFER_ADDRESS @0x540
    #elif defined(_18F4450) || defined(_18F2450)
        #define RX_DATA_BUFFER_ADDRESS @0x480
        #define TX_DATA_BUFFER_ADDRESS @0x4C0
    #elif defined(_16F1459)
        #define RX_DATA_BUFFER_ADDRESS @0x2050
        #define TX_DATA_BUFFER_ADDRESS @0x20A0
    #else
        #define RX_DATA_BUFFER_ADDRESS
        #define RX_DATA_BUFFER_ADDRESS
    #endif
#else
    #define RX_DATA_BUFFER_ADDRESS
    #define TX_DATA_BUFFER_ADDRESS
#endif


unsigned char ReceivedDataBuffer[RECEIVE_DATA_BUFFER_SIZE] RX_DATA_BUFFER_ADDRESS;
unsigned char ToSendDataBuffer[SEND_DATA_BUFFER_SIZE] TX_DATA_BUFFER_ADDRESS;


#if defined(__18CXX)
#pragma udata
#endif

USB_HANDLE USBOutHandle = 0;    //USB handle.  Must be initialized to 0 at startup.
USB_HANDLE USBInHandle = 0;     //USB handle.  Must be initialized to 0 at startup.


// ----------------------------------

// ----- USART Global variables -----
//char message_char;   //  Used for receiving commands from the computer
unsigned char message[RECEIVE_DATA_BUFFER_SIZE];
unsigned char this_device_ID;
long int counter_inactive_read_cycles_usart=0;
int usart_counter_char=0;   //int counter_usart=0;

 
// ----------------------------------


//--------- Timer Interrupt Routine variables and constants ----------
unsigned int tmr3_counter=0;
long int sensor_signal_cycles_counter_for_current_channel;
long int SENSOR_SIGNAL_CYCLES_COUNTERS[N_LCR_CHANNELS]={0,0,0,0,0,0};		//a set of counters that for each sensor channel count the number of cycles of the sensor signal (number of cycles observed on the square wave during the measurement time of the sensor channel)
char ch_number,next_ch_number;
char CALCULATE_SENSORS_FREQ=-1;	//variable used to signal the main loop to calculate the sensors frequencies for the 6 channels plues the 4 ADC channels
#define ZERO_CONSTANT 0x00
//int measurement_range_vector[N_LCR_CHANNELS]={4,4,4,4,4,4};	//vector were are saved the measurement ranges for the various channels
unsigned int count_high,count_low,time_high,time_low;
unsigned char TMR0H_local,TMR0L_local;
#define TMR3_MEASUREMENT_TIME_SELECTION 2

//ATTENTION: Declare mode_is_single_channel_bool with the "volatile" keyword to assure its value is updated inside the ISR routine
volatile unsigned char mode_is_single_channel_bool;	//indicates if working with multiple LCR sensors simultaneously or only a single channel with improved precision
unsigned char selected_channel_number;	//only valid when working single channel mode
//char TMR0H_last_value;
char USART_cmd_received_bool;
long int counters_LCR_sensors[N_LCR_CHANNELS];	// counters for storing the number of total cycles (over long time) of the output signal from a sensor with square wave voltage output,
													//useful for sensors with square wave digital output, low frequency, Ex flowmeter, windspeed meter
//-----------------------------------


//--------- various other global variables and constants ----------
#pragma udata calib_table_copy_on_RAM_0_section
float calib_table_copy_RAW_values_on_RAM[32];		//copy on the RAM of the 1st column (RAW_values) of a calibration table for a sensor channel
float calib_table_copy_measurements_on_RAM[32];		//copy on the RAM of the 2nd column (measurements) of a calibration table for a sensor channel

#pragma udata udata_main

//-----------------------------------


/** PRIVATE PROTOTYPES *********************************************/
static void InitializeSystem(void);
void ProcessIO(void);
void UserInit(void);
void YourHighPriorityISRCode();
void YourLowPriorityISRCode();
void USBCBSendResume(void);



void USART_ProcessModbusRTU(unsigned char *message);
char ProcessModbusRTU(unsigned char *message, signed char *byte_number_to_be_sent_ptr);
char BoolExpressionEvaluator(char *bool_expr_str_ptr, unsigned char str_max_size);
void ReplaceBoolVariablesByValue(char *bool_func_str_ptr, char *bool_expr_str_ptr, unsigned char str_max_size);


//wait_ticks , Description: halts the execution of the program during the ticks in argument "ticks_number"
void wait_ticks(int ticks_number) {
	int i;
	for(i=0;i<ticks_number;i++);
}

/*
// FUNCTION: void SelectTimer0Prescaler - Timer0 prescaler selection function,sets the prescaler division to be used on the timer0 counter
// SelectTimer0Prescaler is called on the interrupt service routine
//tmr0_prescaler: Timer0 Prescaler selection , =7 => 1:256 Prescale, =6 => 1:128 Prescale value,  =3 => 1:16 Prescale value , =1 => 1:4 Prescale value, =0 => 1:2 Prescale value
void SelectTimer0Prescaler() {

	char tmr0_prescaler_selection=4;	//is best to use a fixed value of the presacler to avoid program bugs, best value for prescaler is 4

	//T0PS?: Timer0 Prescaler Select bits , =111 => 1:256 Prescale, =110 => 1:128 Prescale value,  =011 => 1:16 Prescale value , =001 => 1:4 Prescale value
	if((tmr0_prescaler_selection&0x4)>0) {
		T0CONbits.T0PS2=1;	
	}
	else {
		T0CONbits.T0PS2=0;
	}
	if((tmr0_prescaler_selection&0x2)>0) {
		T0CONbits.T0PS1=1;
	}
	else {
		T0CONbits.T0PS1=0;
	}
	if((tmr0_prescaler_selection&0x1)>0) {
		T0CONbits.T0PS0=1;
	}
	else {
		T0CONbits.T0PS0=0;
	}
}
*/

/*	//the function SelectSensorChannel(...) is not used because its code was placed directly inside interrupt function 'YourHighPriorityISRCode', so it runs faster not having to call a function
// FUNCTION: void SelectSensorChannel - Selects the current sensor channel, changes the selected channel on the multiplexer 
void SelectSensorChannel(int channel_number) {

	AN_MUX_A=0x1&channel_number;
	AN_MUX_B=(0x2&channel_number)>>1;
	AN_MUX_C=(0x4&channel_number)>>2;

//	LATAbits.LATA1=0x1&channel_number;
//	LATAbits.LATA2=(0x2&channel_number)>>1;
//	LATAbits.LATA3=(0x4&channel_number)>>2;
}
// ------------------------------------------------------------------------------------
*/


/** VECTOR REMAPPING ***********************************************/
#if defined(__18CXX)
	//On PIC18 devices, addresses 0x00, 0x08, and 0x18 are used for
	//the reset, high priority interrupt, and low priority interrupt
	//vectors.  However, the current Microchip USB bootloader 
	//examples are intended to occupy addresses 0x00-0x7FF or
	//0x00-0xFFF depending on which bootloader is used.  Therefore,
	//the bootloader code remaps these vectors to new locations
	//as indicated below.  This remapping is only necessary if you
	//wish to program the hex file generated from this project with
	//the USB bootloader.  If no bootloader is used, edit the
	//usb_config.h file and comment out the following defines:
	//#define PROGRAMMABLE_WITH_USB_HID_BOOTLOADER
	//#define PROGRAMMABLE_WITH_USB_LEGACY_CUSTOM_CLASS_BOOTLOADER
	
	#if defined(PROGRAMMABLE_WITH_USB_HID_BOOTLOADER)
		#define REMAPPED_RESET_VECTOR_ADDRESS			0x1000
		#define REMAPPED_HIGH_INTERRUPT_VECTOR_ADDRESS	0x1008
		#define REMAPPED_LOW_INTERRUPT_VECTOR_ADDRESS	0x1018
	#elif defined(PROGRAMMABLE_WITH_USB_MCHPUSB_BOOTLOADER)	
		#define REMAPPED_RESET_VECTOR_ADDRESS			0x800
		#define REMAPPED_HIGH_INTERRUPT_VECTOR_ADDRESS	0x808
		#define REMAPPED_LOW_INTERRUPT_VECTOR_ADDRESS	0x818
	#else	
		#define REMAPPED_RESET_VECTOR_ADDRESS			0x00
		#define REMAPPED_HIGH_INTERRUPT_VECTOR_ADDRESS	0x08
		#define REMAPPED_LOW_INTERRUPT_VECTOR_ADDRESS	0x18
	#endif
	
	#if defined(PROGRAMMABLE_WITH_USB_HID_BOOTLOADER)||defined(PROGRAMMABLE_WITH_USB_MCHPUSB_BOOTLOADER)
	extern void _startup (void);        // See c018i.c in your C18 compiler dir
	#pragma code REMAPPED_RESET_VECTOR = REMAPPED_RESET_VECTOR_ADDRESS
	void _reset (void)
	{
	    _asm goto _startup _endasm
	}
	#endif
	#pragma code REMAPPED_HIGH_INTERRUPT_VECTOR = REMAPPED_HIGH_INTERRUPT_VECTOR_ADDRESS
	void Remapped_High_ISR (void)
	{
	     _asm goto YourHighPriorityISRCode _endasm
	}
	#pragma code REMAPPED_LOW_INTERRUPT_VECTOR = REMAPPED_LOW_INTERRUPT_VECTOR_ADDRESS
	void Remapped_Low_ISR (void)
	{
	     _asm goto YourLowPriorityISRCode _endasm
	}
	
	#if defined(PROGRAMMABLE_WITH_USB_HID_BOOTLOADER)||defined(PROGRAMMABLE_WITH_USB_MCHPUSB_BOOTLOADER)
	//Note: If this project is built while one of the bootloaders has
	//been defined, but then the output hex file is not programmed with
	//the bootloader, addresses 0x08 and 0x18 would end up programmed with 0xFFFF.
	//As a result, if an actual interrupt was enabled and occurred, the PC would jump
	//to 0x08 (or 0x18) and would begin executing "0xFFFF" (unprogrammed space).  This
	//executes as nop instructions, but the PC would eventually reach the REMAPPED_RESET_VECTOR_ADDRESS
	//(0x1000 or 0x800, depending upon bootloader), and would execute the "goto _startup".  This
	//would effective reset the application.
	
	//To fix this situation, we should always deliberately place a 
	//"goto REMAPPED_HIGH_INTERRUPT_VECTOR_ADDRESS" at address 0x08, and a
	//"goto REMAPPED_LOW_INTERRUPT_VECTOR_ADDRESS" at address 0x18.  When the output
	//hex file of this project is programmed with the bootloader, these sections do not
	//get bootloaded (as they overlap the bootloader space).  If the output hex file is not
	//programmed using the bootloader, then the below goto instructions do get programmed,
	//and the hex file still works like normal.  The below section is only required to fix this
	//scenario.
	#pragma code HIGH_INTERRUPT_VECTOR = 0x08
	void High_ISR (void)
	{
	     _asm goto REMAPPED_HIGH_INTERRUPT_VECTOR_ADDRESS _endasm
	}
	#pragma code LOW_INTERRUPT_VECTOR = 0x18
	void Low_ISR (void)
	{
	     _asm goto REMAPPED_LOW_INTERRUPT_VECTOR_ADDRESS _endasm
	}
	#endif	//end of "#if defined(PROGRAMMABLE_WITH_USB_HID_BOOTLOADER)||defined(PROGRAMMABLE_WITH_USB_LEGACY_CUSTOM_CLASS_BOOTLOADER)"

	#pragma code
	
	
	//These are your actual interrupt handling routines.
	#pragma interrupt YourHighPriorityISRCode
	void YourHighPriorityISRCode()
	{
		//Check which interrupt flag caused the interrupt.
		//Service the interrupt
		//Clear the interrupt flag
		//Etc.

		int counter;	//counter must be integer (int) type

        #if defined(USB_INTERRUPT)
	        USBDeviceTasks();
        #endif


		//check if a new character is available to be read on the serial port buffer (1 byte, 8 bits)
		if (PIR1bits.RCIF == 1) { // check if this bit is set, meaning a byte received

			// program to read string from USART instead of a character each time, so you can read strings written at once at the serial port by the Master
			message[usart_counter_char] = (unsigned char) ReadUSART();
			usart_counter_char++;
			PIR1bits.RCIF=0; //clear the USART interrupt flag

			//try to read a string to the USART
			for(counter=0; usart_counter_char<RECEIVE_DATA_BUFFER_SIZE && counter<=13; counter++) {
				if( PIR1bits.RCIF == 1 ) {
					message[usart_counter_char] = (unsigned char) ReadUSART();
					PIR1bits.RCIF=0; //clear the USART interrupt flag
					usart_counter_char = usart_counter_char + 1;
					counter=0;		
				}
				wait_ticks(50);
				//the time of each character at baud_rate=57600 is about 250 ticks to 300 ticks
				//wait a maximum of 600 ticks before declaring the end of the string
			} 

			//check for commands from the USART in the format of multiple-sensor
			for(counter=0; USART_cmd_received_bool<1; counter++) {
				if( message[counter]=='\r' || message[counter]=='\n') {
					USART_cmd_received_bool=1;
				}
			}

			if( usart_counter_char > 5 ) {
				USART_cmd_received_bool=1;  //in case more than 5 characters were received declare a new command
			}

			if(usart_counter_char>=RECEIVE_DATA_BUFFER_SIZE) {
				//clear the buffer in case the buffer is full, any messages in the buffer should have been processed by now
				message[0]='\0';
				usart_counter_char=0;
			}


			PIR1bits.RCIF = 0;

		}

		else {

			if((PIR2&(0x02))>ZERO_CONSTANT) {	//check if timer3 has reached it's final value (TMR3IF is the bit nº1 of PIR2 register, TMR3IF is set to 1 when timer3 reaches it's final value)
	
				T0CONbits.TMR0ON = 0;    // disable timer0 counter for reading the freq count
				TMR0L_local=TMR0L;
				TMR0H_local=TMR0H;	//always read TMR0H after reading TMR0L, because TMR0H has a hardware buffer, and TMR0H is updated when TMR0L is read
				TMR0H =0x00;	//reset hardware counter (HIGH byte) for TMR0
				TMR0L =0x00;	//reset hardware counter (LOW byte) for TMR0
				T0CONbits.TMR0ON = 1;    // enable immediately timer0 counter to continue the freq measurement 

				// ----- read the value of the frequency count -----
				count_high = ( ( (unsigned int) TMR0H_local )<<8 );
				count_low = ((unsigned int) TMR0L_local)&0x00FF;

				

				if(INTCON&0x04) {   //check if TMR0IF is 1, TMR0IF is the bit2 of the INTCON register. In case TMR0IF is 1 then TMR0 counter has overflow and so should not be used to fill the 'SENSOR_SIGNAL_CYCLES_COUNTERS' ; in case the input signal has a frequency too high to be measured it will cause the TMR0 counter to overflow. 
					sensor_signal_cycles_counter_for_current_channel = 99999999;	
					INTCON=INTCON&0xFB;		//clear the TMR0IF flag (that indicates that an overflow occurred on TMR0 counter), TMR0IF is the bit2 of the INTCON register.
				}
				else {
					//check if is required to read the TMR0L and TMR0H to increment the SENSOR_SIGNAL_CYCLES_COUNTERS
					sensor_signal_cycles_counter_for_current_channel= sensor_signal_cycles_counter_for_current_channel + (count_high | count_low);  //save the final count of the sensor signals	
				}


				if(tmr3_counter>TMR3_MEASUREMENT_TIME_SELECTION) {		//tmr3_counter>2
					
					if(mode_is_single_channel_bool==0) {	//if mode_is_single_channel_bool==0 then is selected multiple channels working simultaneously
						next_ch_number=ch_number+1;
						if(next_ch_number>=N_LCR_CHANNELS) {	//check if next_ch_number is in the range of available channels
							next_ch_number=0;
							CALCULATE_SENSORS_FREQ=2;	//(=2) signals the main loop to calculate the sensors frequencies for the 6 channels
						}						
					}
					else {
						if(selected_channel_number<N_LCR_CHANNELS) {
							next_ch_number=selected_channel_number;
							ch_number=selected_channel_number;
							CALCULATE_SENSORS_FREQ=2;	//(=2) signals the main loop to calculate the sensor frequency of the channel that is being read
						}
						else {
							CALCULATE_SENSORS_FREQ=0;	//in case selected channel number is 6,7,8,9 this is an ADC channels, so just signal to read the ADC values
						}
					}

					if(selected_channel_number<N_LCR_CHANNELS) {	

						T0CONbits.TMR0ON = 0;    // disable timer0 counter for reading the freq count
						// ----- SelectSensorChannel(next_ch_number); ----- //replaced the call of a function by the code to make ISR faster
						AN_MUX_A=0x1&next_ch_number;
						AN_MUX_B=(0x2&next_ch_number)>>1;
						AN_MUX_C=(0x4&next_ch_number)>>2;
						// ------------------------------------------------
						TMR0H =0x00;	//reset hardware counter (HIGH byte) for TMR0
						TMR0L =0x00;	//reset hardware counter (LOW byte) for TMR0
						T0CONbits.TMR0ON = 1;    // enable immediately timer0 counter for the freq measurement of the next channel
		
						SENSOR_SIGNAL_CYCLES_COUNTERS[ch_number]= sensor_signal_cycles_counter_for_current_channel;  //save the final count of the sensor signals	
						sensor_signal_cycles_counter_for_current_channel = 0;	//reset the software counter for the current sensor channel
						
						//update the counters of the total cycles of the sensors signals
						counters_LCR_sensors[ch_number]=counters_LCR_sensors[ch_number] + SENSOR_SIGNAL_CYCLES_COUNTERS[ch_number];			
		
		
						ch_number=next_ch_number;  //change to the next channel
						// -------------------------------------------------
					}

					tmr3_counter=0;
				
				}

				tmr3_counter++;
				PIR2bits.TMR3IF = 0;            // clear timer3 interrupt flag TMR3IF
			} 
	
			PIR1=PIR1&0xF8; //always clean the interrupt flags related to the timers (bit0:TMR1IF, bit1:TMR2IF, bit2:CCP1IF)
			PIR2=PIR2&0xFC; //always clean the interrupt flags related to the timers (bit0:CCP2IF, bit1:TMR3IF)		
		}

	}	//This return will be a "retfie fast", since this is in a #pragma interrupt section 
	#pragma interruptlow YourLowPriorityISRCode
	void YourLowPriorityISRCode()
	{
		//Check which interrupt flag caused the interrupt.
		//Service the interrupt
		//Clear the interrupt flag
		//Etc.
	
	}	//This return will be a "retfie", since this is in a #pragma interruptlow section 
#elif defined(_PIC14E)
    	//These are your actual interrupt handling routines.
	void interrupt ISRCode()
	{
		//Check which interrupt flag caused the interrupt.
		//Service the interrupt
		//Clear the interrupt flag
		//Etc.
        #if defined(USB_INTERRUPT)

	        USBDeviceTasks();
        #endif
	}
#endif




// ----- STACK LIBRARY (SIMPLIFIED) ON C LANGUAGE, BY USING A CHAR ARRAY FOR STORING CHARACTERS ON A LIFO STACK --- START: -----
#define STACK_SIZE bool_func_N_bytes
struct stack_array_ {
	char sArray[STACK_SIZE];
	char stackTop;		//integer between -128 and +127
};
typedef struct stack_array_ stack_array;

#define init_stack(s_arr) s_arr.stackTop=CONST_NEG_ONE

#define top_stack(s_arr) s_arr.sArray[s_arr.stackTop]

void push_stack(char element, stack_array *s_arr_ptr){
 if(s_arr_ptr->stackTop == CONST_NEG_ONE){
  s_arr_ptr->sArray[STACK_SIZE - 1] = element;
  s_arr_ptr->stackTop = STACK_SIZE - 1;
 }
 else if(s_arr_ptr->stackTop > 0){
	s_arr_ptr->sArray[(s_arr_ptr->stackTop) - 1] = element;
	(s_arr_ptr->stackTop)--;
 }
 //else{
	//printf("The stack is already full. \n");
 //}
}


void pop_stack(stack_array *s_arr_ptr){
	if(s_arr_ptr->stackTop >= 0){
		//printf("Element popped: %c \n", s_arr_ptr->sArray[(s_arr_ptr->stackTop)]);
		// If the element popped was the last element in the stack, then set top to -1 to show that the stack is empty
		if((s_arr_ptr->stackTop) == STACK_SIZE - 1){
			(s_arr_ptr->stackTop) = CONST_NEG_ONE;
		}
		else{
			(s_arr_ptr->stackTop)++;
		}
	}
	//else{
		//printf("The stack is empty. \n");
	//}
}
// ----- STACK LIBRARY (SIMPLIFIED) ON C LANGUAGE, BY USING A CHAR ARRAY FOR STORING CHARACTERS ON A LIFO STACK --- END. -----


/*
// FUNCTION: power_n executes base^n, n must be integer, and return the result
double power_n(double base, int n) {
	double result;
	int i;

	for(i=1,result=base;i<n;i++) {
		result=result*base;
	}

	return result;
} */


// ----- internal EEPROM memory allocation diagram (PIC18F2550 internal memory) ---------------------------------------------------------------------------------------------
//  CHANNEL CONFIGS    |US| DEVICE_ID |US|  SENSOR  |   SENSOR  |  ...  |  SENSOR  |   SENSOR    |US| BOOL_FUNC | BOOL_FUNC | BOOL_FUNC |
//  AND BOOL FUNCTIONS |  |           |  |CONFIG CH0|TRIGGER CH0|  ...  |CONFIG CH9| TRIGGER CH9 |  |  OUT.0    |   OUT.1   |   OUT.2   |
// -----------------------------------------------------------------------------------------------------------------------------------
//   SIZE(bytes):      |1 |  1 byte   |1 |  1 byte  |  4 bytes  |  ...  |  1 byte  |   4 bytes   |1 |  50 bytes | 50 bytes  |  50 bytes |
// (PIC18F2550 internal EEPROM) TOTAL:  3+(10x5)+1+3x53=213 bytes
// -----------------------------------------------------------------------------------------------------------------------------------


// ----- Save and Load functions for the sensors channels configurations (LCR sensors (6x), and ADC sensors (4x)) ------------------
// Each sensor has 1 byte for configurations and float constant(4 bytes) for event trigger reference value
// the float reference value for event trigger is used to detect an event(situation) detected by the sensor
// and thus activating an output pin (3 outputs available),
// used for example: for activating and alarm system when sensor is triggered, or signaling an event such as weight applied on sensor passed requested load weight so dispatch the load

// FUNCTION: WriteSensorConfig - Writes to the EEPROM the sensor configurations for the specified sensor channel number
/*
void WriteSensorConfig(unsigned int start_eeprom_addr, unsigned int config_channel_number, char sensor_configs, float sensor_trigger_value) {

	unsigned char * sensor_trigger_data;
	unsigned char data_byte;
	unsigned char i;	//"I" number smaller than 255

	unsigned int config_sensor_channel_address=start_eeprom_addr+config_channel_number*5;  //address on the EEPROM for the sensor configuration and for sensor trigger value 

	//the sensor configuration plus the sensor trigger value occupies in memory 1+4*1=5bytes
	// (1 byte, ((1 bit) select RAW/units, (1bit) select comparison > or <, (3bits) sensor prescaler, (1bit) used for OUT0, (1bit) used for OUT1, (1bit) used for OUT2)

	//write sensor configurations values on the EEPROM starting on address start_eeprom_addr

	Write_b_eep( config_sensor_channel_address, sensor_configs );  //writes a byte to the EEPROM, with the sensor configuration
	Busy_eep ();     // Wait until write is complete   

	sensor_trigger_data = (unsigned char *) &sensor_trigger_value;	//get a pointer for the data of the float variable

	for(i=0; i<4; i++) {
		memmove((void *) &data_byte, (void *) &(sensor_trigger_data[i]), sizeof(unsigned char));
		Write_b_eep( config_sensor_channel_address + i + 1, data_byte );  //writes a byte to the EEPROM
		Busy_eep ();     // Wait until write is complete   
	}	

	//memmove((void *) &data_byte, (void *) &(sensor_trigger_data[0]), sizeof(unsigned char));
	//Write_b_eep( config_sensor_channel_address + 1, data_byte );  //writes a byte to the EEPROM
	//Busy_eep ();     // Wait until write is complete   
	//memmove((void *) &data_byte, (void *) &(sensor_trigger_data[1]), sizeof(unsigned char));
	//Write_b_eep( config_sensor_channel_address + 2, data_byte );  //writes a byte to the EEPROM
	//Busy_eep ();     // Wait until write is complete  
	//memmove((void *) &data_byte, (void *) &(sensor_trigger_data[2]), sizeof(unsigned char));
	//Write_b_eep( config_sensor_channel_address + 3, data_byte );  //writes a byte to the EEPROM
	//Busy_eep ();     // Wait until write is complete  
	//memmove((void *) &data_byte, (void *) &(sensor_trigger_data[3]), sizeof(unsigned char));
	//Write_b_eep( config_sensor_channel_address + 4, data_byte );  //writes a byte to the EEPROM
	//Busy_eep ();     // Wait until write is complete  
}
*/

// FUNCTION: ReadSensorConfig Reads from the EEPROM the configurations for the selected sensor channel
/*
void ReadSensorConfig(unsigned int start_eeprom_addr, unsigned int config_channel_number, char *sensor_configs_ptr, float *sensor_trigger_value_ptr) {

	unsigned char *sensor_trigger_value_data;
	unsigned char data_byte;
	float sensor_trigger_value_local;
	unsigned char i;	//"I" number smaller than 255

	unsigned int config_sensor_channel_address=start_eeprom_addr+config_channel_number*5;  //address on the EEPROM for the sensor configuration and for sensor trigger value 
	//the sensor configuration plus the sensor trigger value occupies in memory 1+4*1=5bytes
	// (1 byte, ((1 bit) select RAW/units, (1bit) select comparison > or <, (3bits) sensor prescaler, (1bit) used for OUT0, (1bit) used for OUT1, (1bit) used for OUT2)

	//read sensor configurations values on the EEPROM starting on address start_eeprom_addr

	*sensor_configs_ptr=Read_b_eep( config_sensor_channel_address );  //writes a byte from the EEPROM, with the sensor configuration
	Busy_eep ();     // Wait until write is complete   

	//Read the sensor configuration from the EEPROM starting on address start_eeprom_addr

	sensor_trigger_value_data = (unsigned char *) &sensor_trigger_value_local;

	for(i=0; i<4; i++) {
		data_byte=Read_b_eep( config_sensor_channel_address + i + 1 ); //reads a byte from the EEPROM
		memmove((void *) (sensor_trigger_value_data + i), (void *) &data_byte, sizeof(unsigned char));
	}

	//data_byte=Read_b_eep( config_sensor_channel_address + 1 ); //reads a byte from the EEPROM
	//memmove((void *) (sensor_trigger_value_data + 0), (void *) &data_byte, sizeof(unsigned char));
	//data_byte=Read_b_eep( config_sensor_channel_address + 2 ); //reads a byte from the EEPROM
	//memmove((void *) (sensor_trigger_value_data + 1), (void *) &data_byte, sizeof(unsigned char));
	//data_byte=Read_b_eep( config_sensor_channel_address + 3 ); //reads a byte from the EEPROM
	//memmove((void *) (sensor_trigger_value_data + 2), (void *) &data_byte, sizeof(unsigned char));
	//data_byte=Read_b_eep( config_sensor_channel_address + 4 ); //reads a byte from the EEPROM
	//memmove((void *) (sensor_trigger_value_data + 3), (void *) &data_byte, sizeof(unsigned char));

	*sensor_trigger_value_ptr=sensor_trigger_value_local;
}
*/


// FUNCTION: ReadOrWriteSensorConfig - Writes to the EEPROM the sensor configurations for the specified sensor channel number
void ReadOrWriteSensorConfig(unsigned char is_write_bool, unsigned int start_eeprom_addr, unsigned int config_channel_number, char *sensor_configs_ptr, float *sensor_trigger_value_ptr) {

	unsigned char *sensor_trigger_value_data;
	unsigned char data_byte;
	unsigned char i;	//"I" number smaller than 255

	unsigned int config_sensor_channel_address=start_eeprom_addr+config_channel_number*5;  //address on the EEPROM for the sensor configuration and for sensor trigger value 

	//the sensor configuration plus the sensor trigger value occupies in memory 1+4*1=5bytes
	// (1 byte, ((1 bit) select RAW/units, (1bit) select comparison > or <, (3bits) sensor prescaler, (1bit) used for OUT0, (1bit) used for OUT1, (1bit) used for OUT2)

	//read or write sensor configurations values on the EEPROM starting on address start_eeprom_addr
	if(is_write_bool==1) {
		Write_b_eep( config_sensor_channel_address, *sensor_configs_ptr );  //writes a byte to the EEPROM, with the sensor configuration  
	}
	else {
		*sensor_configs_ptr=Read_b_eep( config_sensor_channel_address );  //reads a byte from the EEPROM, with the sensor configuration
	}
	Busy_eep ();     // Wait until write is complete
	
	sensor_trigger_value_data = (unsigned char *) sensor_trigger_value_ptr;
	for(i=0; i<4; i++) {
		if(is_write_bool==1) {
			memmove((void *) &data_byte, (void *) &(sensor_trigger_value_data[i]), sizeof(unsigned char));
			Write_b_eep( config_sensor_channel_address + i + 1, data_byte );  //writes a byte to the EEPROM
			Busy_eep ();     // Wait until write is complete 
		}
		else {
			data_byte=Read_b_eep( config_sensor_channel_address + i + 1 ); //reads a byte from the EEPROM
			memmove((void *) (sensor_trigger_value_data + i), (void *) &data_byte, sizeof(unsigned char));
		}
	}

}

#define WRITE_SENSOR_CONFIG(start_eeprom_addr, config_channel_number, sensor_configs_ptr, sensor_trigger_value_ptr) ReadOrWriteSensorConfig(1, start_eeprom_addr, config_channel_number, sensor_configs_ptr, sensor_trigger_value_ptr)
#define READ_SENSOR_CONFIG(start_eeprom_addr, config_channel_number, sensor_configs_ptr, sensor_trigger_value_ptr) ReadOrWriteSensorConfig(0, start_eeprom_addr, config_channel_number, sensor_configs_ptr, sensor_trigger_value_ptr)

// -----------------------------------------------------------------------------------


// FUNCTION: ReadOrWriteOutputsConfig - Read or write from the EEPROM the output configuration of an OUT pin (boolean function)
void ReadOrWriteOutputsConfig(unsigned char is_write_bool, unsigned char out_number, char *output_bool_function_str) {
	unsigned char i;
	unsigned char addr = EEPROM_ADDR_BOOL_FUNCTIONS+(out_number*bool_func_N_bytes);	

	for(i=0; i<bool_func_N_bytes; i++) {
		if(is_write_bool==1) {
			Write_b_eep( addr+i, output_bool_function_str[i] );  //writes a byte to the EEPROM
		}
		else {
			output_bool_function_str[i] = output_bool_function_str[i]=Read_b_eep( addr+i ); //read a byte from the EEPROM
		}
		Busy_eep ();     // Wait until write is complete   
	}
}




// ----- Save and Load functions for the CHANNEL_MODO_MULTIPLE_OR_SINGLE ------------------
// FUNCTION: WriteModeCH - Writes to the EEPROM the selected channel mode (multiple or single) and if single writes also the selected channel 
void WriteModeCH(unsigned int start_eeprom_addr, unsigned char mode_is_single_channel_bool, unsigned char selected_channel_number) {

	Write_b_eep( start_eeprom_addr, mode_is_single_channel_bool );  //writes a byte to the EEPROM, with the channel mode 
	Busy_eep ();     // Wait until write is complete   
	Write_b_eep( start_eeprom_addr+1, selected_channel_number );  //writes a byte to the EEPROM, with the selected channel
	Busy_eep ();     // Wait until write is complete 

}

// FUNCTION: ReadModeCH - Read from the EEPROM the selected channel mode (multiple or single) and if single read also the selected channel 
void ReadModeCH(unsigned int start_eeprom_addr, unsigned char *mode_is_single_channel_bool_ptr, unsigned char *selected_channel_number_ptr) {

	*mode_is_single_channel_bool_ptr=Read_b_eep( start_eeprom_addr );  //read a byte from the EEPROM, with the selected channel mode
	Busy_eep ();     // Wait until write is complete
	*selected_channel_number_ptr=Read_b_eep( start_eeprom_addr+1 );  //read a byte from the EEPROM, with the selected channel number
	Busy_eep ();     // Wait until write is complete
 
}


// ---------------------------------

// -----------------------------------------------------------------------------------

// VECTOR: char GetTimer0PrescalerValue - Timer0 prescaler value vector, returns the division constant of timer0 prescaler for the selected channel
//tmr0_prescaler: Timer0 Prescaler selection , =7 => 1:256 Prescale, =6 => 1:128 Prescale value,  =3 => 1:16 Prescale value , =1 => 1:4 Prescale value, =0 => 1:2 Prescale value
//int GetTimer0PrescalerValue[8]={2,4,8,16,32,64,128,256};	//OBSOLETE: replace by PRESCALER_VAL (define constant)

/*
	// ----- DEBUG CODE -----
	//putrsUSART("\r\n byte received !! \r\n");
	putrsUSART("\r\n ReadCommandUSART Called\r\n usart_counter_char:");
	sprintf(test_str,"%d \r\n", *usart_counter_char_ptr);
	putsUSART(test_str);
	// ----------------------
*/


// FUNCTION: Find_Start_Modbus_RTU - Find any start of ModbusRTU command of the type: ID|function_code
//		     Return value: Returns the index number were is the first byte of the ModbusRTU command
signed char Find_Start_Modbus_RTU(unsigned char ID, unsigned char function_code, unsigned char *str_msg, int str_msg_length) {

	unsigned char i;		//"i" number smaller than 255
 	signed char ret_value=-1;	//"" number smaller than 127

	for( i=0, ret_value=-1; i<str_msg_length && ret_value<0; i++) {
		if(str_msg[i]==ID || str_msg[i]==255) {	//search for this_device_ID or for the broadcast_ID (device_ID=255)
			if(str_msg[i+1]==function_code) {
				ret_value=i;
			}
		}
	}

	return ret_value;
}


// FUNCTION: Find_Any_Start_Modbus_RTU, Description: Find any start of ModbusRTU command of the type: ID|function_code, where function_code can be: {4; 6; 16}
//		     Return value: Returns the index number were is the first byte of the ModbusRTU command
signed char Find_Any_Start_Modbus_RTU(unsigned char *str_msg, int str_msg_length, unsigned char ID) {

	char start_of_cmd_found=0;

 	signed char ret_value=-1;	//"" number smaller than 127
	unsigned char i_min=255;		//"i_min" number smaller than 255

	unsigned char i;	//"i" number smaller than 255
	const unsigned char function_codes_to_search[4]={3, 4, 6, 16};


	i_min=9999;
	start_of_cmd_found=0;
	str_msg_length=str_msg_length+5; //some extra length to be searched

	//repeated code of Find_Start_Modbus_RTU(...), to avoid calling another function, and so increasing the call depth, reducing the risk of stack overflow

	for(i=0; i<4; i++) {
		ret_value=Find_Start_Modbus_RTU(ID, function_codes_to_search[i], str_msg, str_msg_length);
		if(ret_value<i_min && ret_value>=0) {
			i_min=ret_value; start_of_cmd_found=1;
		}
	}

	if( start_of_cmd_found == 0)
		i_min=-1;	//wasn't found any start of a Modbus RTU command

	return i_min;
}


#define NEG_INF -99999999
#define POS_INF 99999999


// -----------------------------------------------------------------------------------


//********************************************************************
//     Function Name:    ReadI2C_w_timeout                           *
//     Return Value:     contents of SSPBUF register                 *
//     Parameters:       void                                        *
//     Description:      Read single byte from I2C bus.              *
//********************************************************************
#if defined (I2C_V1)
unsigned char ReadI2C_w_timeout( int timeout_ticks )
{
	int time_counter;

if( ((SSPCON1&0x0F)==0x08) || ((SSPCON1&0x0F)==0x0B) )	//master mode only
  SSPCON2bits.RCEN = 1;           // enable master for 1 byte reception
  for(time_counter=0; !SSPSTATbits.BF && time_counter < timeout_ticks; time_counter++);	// wait until byte received    
  return ( SSPBUF );              // return with read byte 
}
#endif

#if defined (I2C_V4)
unsigned char ReadI2C( void )
{
  for(time_counter=0; !SSPSTATbits.BF && time_counter < timeout_ticks; time_counter++);	// wait until byte received
  return ( SSPBUF );              // return with read byte 
}
#endif



//********************************************************************
//     Function Name:    getsI2C1_w_timeout                          *
//     Return Value:     error condition status                      *
//     Parameters:       address of read string storage location     *
//                       length of string bytes to read              *
//     Description:      This routine reads a predetermined string   *
//                       length in from the I2C1 bus. The routine is *
//                       developed for the Master mode. The bus ACK  *
//                       condition is generated within this routine. *
//********************************************************************
/* unsigned char getsI2C_w_timeout( unsigned char *rdptr, unsigned char length, int timeout_ticks )
{
	int time_counter;

    while ( length-- )            // perform getcI2C1() for 'length' number of bytes
    {
      
	  *rdptr++ = ReadI2C_w_timeout(timeout_ticks);       // save byte received    //*rdptr++ = getcI2C1();  // save byte received 
	  for(time_counter=0; SSPCON2bits.RCEN && time_counter < timeout_ticks; time_counter++);  // check that receive sequence is over 

      if ( PIR2bits.BCLIF )       // test for bus collision
      {
        return ( -1 );             // return with Bus Collision error 
      }

	  
	if( ((SSPCON1&0x0F)==0x08) || ((SSPCON1&0x0F)==0x0B) )	//master mode only
	{	
      if ( length )               // test if 'length' bytes have been read
      {
        SSPCON2bits.ACKDT = 0;    // set acknowledge bit state for ACK        
        SSPCON2bits.ACKEN = 1;    // initiate bus acknowledge sequence
		for(time_counter=0; SSPCON2bits.ACKEN && time_counter < timeout_ticks; time_counter++);   // wait until ACK sequence is over 
      } 
	} 
	  
    }
    return ( 0 );                  // last byte received so don't send ACK      
} */


//******************************************************************
//Macro       : NotAckI2C1_w_timeout()
//
//Include     : i2c.h
//
//Description : Macro to initiate negative acknowledgment sequence
//
//Arguments   : None
//
//Remarks     : This macro initiates negative acknowledgment condition and 
//		waits till the acknowledgment sequence is terminated.
//		This macro is applicable only to master
//*******************************************************************
void NotAckI2C_w_timeout( int timeout_ticks ) {
	int time_counter;

    SSPCON2bits.ACKDT=1;
	SSPCON2bits.ACKEN=1;
	for(time_counter=0; SSPCON2bits.ACKEN && time_counter < timeout_ticks; time_counter++);	// wait until byte received
}


//**********************************************************************************************
//Function :  IdleI2C_w_timeout(int timeout_ticks)
//
//Include            : i2c.h 
//
//Description        : This Function generates Wait condition until I2C bus is Idle.
//
//Arguments          : None 
//
//Remarks            : This Macro will be in a wait state until Start Condition Enable bit,
//                     Stop Condition Enable bit, Receive Enable bit, Acknowledge Sequence
//                     Enable bit of I2C Control register and Transmit Status bit I2C Status
//                     register are clear. The IdleI2C function is required since the hardware
//                     I2C peripheral does not allow for spooling of bus sequence. The I2C
//                     peripheral must be in Idle state before an I2C operation can be initiated
//                     or write collision will be generated.
//************************************************************************************************
void IdleI2C_w_timeout(int timeout_ticks) {
	int time_counter;

	for(time_counter=0; ((SSPCON2 & 0x1F) | (SSPSTATbits.R_W)) && time_counter < timeout_ticks; time_counter++);

}


//********************************************************************
//     Function Name:    WriteI2C_w_timeout                          *
//     Return Value:     Status byte for WCOL detection.             *
//     Parameters:       Single data byte for I2C bus.               *
//     Description:      This routine writes a single byte to the    * 
//                       I2C bus.                                    *
//********************************************************************
char WriteI2C_w_timeout( unsigned char data_out, int timeout_ticks )
{
	int time_counter;

  SSPBUF = data_out;           // write single byte to SSPBUF
  if ( SSPCON1bits.WCOL )      // test if write collision occurred
   return ( -1 );              // if WCOL bit is set return negative #
  else
  {
	if( ((SSPCON1&0x0F)!=0x08) && ((SSPCON1&0x0F)!=0x0B) )	//Slave mode only
	{
	      SSPCON1bits.CKP = 1;        // release clock line 
	      while ( !PIR1bits.SSPIF );  // wait until ninth clock pulse received

	      if ( ( !SSPSTATbits.R_W ) && ( !SSPSTATbits.BF ) )// if R/W=0 and BF=0, NOT ACK was received
	      {
	        return ( -2 );           //return NACK
	      }
		  else
		  {
			return ( 0 );				//return ACK
		  }	
	}
	else if( ((SSPCON1&0x0F)==0x08) || ((SSPCON1&0x0F)==0x0B) )	//master mode only
	{  
		for(time_counter=0; SSPSTATbits.BF && time_counter < timeout_ticks; time_counter++);	// wait until write cycle is complete
	    IdleI2C_w_timeout( timeout_ticks );                 // ensure module is idle
	    if ( SSPCON2bits.ACKSTAT ) // test for ACK condition received
	    	 return ( -2 );			// return NACK
		else return ( 0 );              //return ACK
	}
	
  }
}

//************************************************************************
//     Function Name:    HDByteWriteI2C_w_timeout                        *   
//     Parameters:       EE memory ControlByte, address and data         *
//     Description:      Writes data one byte at a time to I2C EE        *
//                       device. This routine can be used for any I2C    *
//                       EE memory device, which only uses 1 byte of     *
//                       address data as in the 24LC01B/02B/04B/08B/16B. *
//                                                                       *     
//************************************************************************
void HDByteWriteI2C_w_timeout( unsigned char ControlByte, unsigned short address, unsigned char data, int timeout_ticks )
{
	short i;
 	int time_counter;

  unsigned char HighAdd = (address >> 8) & 0x00FF;
  unsigned char LowAdd = address & 0x00FF;

  IdleI2C_w_timeout( timeout_ticks );                      // ensure module is idle
  SSPCON2bits.SEN=1; 		//StartI2C();         // initiate START condition
  for(time_counter=0; SSPCON2bits.SEN && time_counter < timeout_ticks; time_counter++);   // wait until start condition is over 
  WriteI2C_w_timeout( ControlByte, timeout_ticks );        // write 1 byte - R/W bit should be 0
  WriteI2C_w_timeout( HighAdd, timeout_ticks );            // write address byte to EEPROM
  WriteI2C_w_timeout( LowAdd, timeout_ticks );             // write address byte to EEPROM
  WriteI2C_w_timeout( data, timeout_ticks );              // Write data byte to EEPROM
  SSPCON2bits.PEN=1;	//StopI2C();                      // send STOP condition
  for(time_counter=0; SSPCON2bits.PEN && time_counter < timeout_ticks; time_counter++);     // wait until stop condition is over 
  for(i=0; i<WAIT_BETWEEN_I2C_WRITE_CMD; i++);   //while (EEAckPolling(ControlByte));  //Wait for write cycle to complete, wait for: WAIT_BETWEEN_I2C_WRITE_CMD
  return ;                   // return with no error
}



//************************************************************************
//     Function Name:    HDByteWriteI2C                                  *   
//     Parameters:       EE memory ControlByte, address and data         *
//     Description:      Writes data one byte at a time to I2C EE        *
//                       device. This routine can be used for any I2C    *
//                       EE memory device, which only uses 1 byte of     *
//                       address data as in the 24LC01B/02B/04B/08B/16B. *
//                                                                       *     
//************************************************************************
/*	//INFO: WORKING BUT NOT USED BY THE PROGRAM
void HDByteWriteI2C( unsigned char ControlByte, unsigned char HighAdd, unsigned char LowAdd, unsigned char data )
{
	char i;

  IdleI2C();                      // ensure module is idle
  StartI2C();                     // initiate START condition
  while ( SSPCON2bits.SEN );      // wait until start condition is over 
  WriteI2C( ControlByte );        // write 1 byte - R/W bit should be 0
  IdleI2C();                      // ensure module is idle
  WriteI2C( HighAdd );            // write address byte to EEPROM
  IdleI2C();                      // ensure module is idle
  WriteI2C( LowAdd );             // write address byte to EEPROM
  IdleI2C();                      // ensure module is idle
  WriteI2C ( data );              // Write data byte to EEPROM
  IdleI2C();                      // ensure module is idle
  StopI2C();                      // send STOP condition
  while ( SSPCON2bits.PEN );      // wait until stop condition is over 
  for(i=0;i<50;i++);   //while (EEAckPolling(ControlByte));  //Wait for write cycle to complete
  return ;                   // return with no error
}
*/


//********************************************************************
//     Function Name:    HDByteReadI2C_w_timeout                     *
//     Parameters:       EE memory ControlByte, address, pointer and *
//                       length bytes.                               *
//     Description:      Reads data string from I2C EE memory        *
//                       device. This routine can be used for any I2C*
//                       EE memory device, which only uses 1 byte of *
//                       address data as in the 24LC01B/02B/04B/08B. *
//                                                                   *  
//********************************************************************
unsigned char HDByteReadI2C_w_timeout( unsigned char ControlByte, unsigned short address, int timeout_ticks )
{
  int time_counter;
  unsigned char data;

  unsigned char HighAdd = (address >> 8) & 0x00FF;
  unsigned char LowAdd = address & 0x00FF;

  IdleI2C_w_timeout( timeout_ticks );                      // ensure module is idle
  SSPCON2bits.SEN=1; 		//StartI2C();         // initiate START condition
  for(time_counter=0; SSPCON2bits.SEN && time_counter < timeout_ticks; time_counter++);   // wait until start condition is over 
  WriteI2C_w_timeout( ControlByte, timeout_ticks );        // write 1 byte 
  WriteI2C_w_timeout( HighAdd, timeout_ticks );            // WRITE word address to EEPROM
  while ( SSPCON2bits.RSEN );     // wait until re-start condition is over 
  WriteI2C_w_timeout( LowAdd, timeout_ticks );             // WRITE word address to EEPROM
  SSPCON2bits.RSEN=1;	//RestartI2C();                   // generate I2C bus restart condition
  for(time_counter=0; SSPCON2bits.RSEN && time_counter < timeout_ticks; time_counter++);     // wait until re-start condition is over 
  WriteI2C_w_timeout( ControlByte | 0x01 , timeout_ticks ); // WRITE 1 byte - R/W bit should be 1 for read
  //getsI2C_w_timeout( data, length, timeout_ticks );	// read in multiple bytes
  data = ReadI2C_w_timeout(timeout_ticks);
  NotAckI2C_w_timeout( timeout_ticks );                    // send not ACK condition
  for(time_counter=0; SSPCON2bits.ACKEN && time_counter < timeout_ticks; time_counter++);	// wait until ACK sequence is over 
  SSPCON2bits.PEN=1;	//StopI2C();                      // send STOP condition
  for(time_counter=0; SSPCON2bits.PEN && time_counter < timeout_ticks; time_counter++);     // wait until stop condition is over 
  return ( data );                   // return the data byte that was read from the EEPROM by I2C
}



//********************************************************************
//     Function Name:    HDByteReadI2C                               *
//     Parameters:       EE memory ControlByte, address, pointer and *
//                       length bytes.                               *
//     Description:      Reads data string from I2C EE memory        *
//                       device. This routine can be used for any I2C*
//                       EE memory device, which only uses 1 byte of *
//                       address data as in the 24LC01B/02B/04B/08B. *
//                                                                   *  
//********************************************************************
/*     //INFO: WORKING BUT NOT USED BY THE PROGRAM
void HDByteReadI2C( unsigned char ControlByte, unsigned char HighAdd, unsigned char LowAdd, unsigned char *data, unsigned char length )
{
  IdleI2C();                      // ensure module is idle
  StartI2C();                     // initiate START condition
  while ( SSPCON2bits.SEN );      // wait until start condition is over 
  WriteI2C( ControlByte );        // write 1 byte 
  IdleI2C();                      // ensure module is idle
  WriteI2C( HighAdd );            // WRITE word address to EEPROM
  IdleI2C();                      // ensure module is idle
  while ( SSPCON2bits.RSEN );     // wait until re-start condition is over 
  WriteI2C( LowAdd );             // WRITE word address to EEPROM
  IdleI2C();                      // ensure module is idle
  RestartI2C();                   // generate I2C bus restart condition
  while ( SSPCON2bits.RSEN );     // wait until re-start condition is over 
  WriteI2C( ControlByte | 0x01 ); // WRITE 1 byte - R/W bit should be 1 for read
  IdleI2C();                      // ensure module is idle
  getsI2C( data, length );       // read in multiple bytes
  NotAckI2C();                    // send not ACK condition
  while ( SSPCON2bits.ACKEN );    // wait until ACK sequence is over 
  StopI2C();                      // send STOP condition
  while ( SSPCON2bits.PEN );      // wait until stop condition is over 
  return ;                   // return with no error
}
------------------------------------------------------- */



//********************************************************************
//     Function Name:  HDBytesReadI2C_w_timeout                      *
//     Parameters:     EE memory ControlByte, address, local_mem_ptr *
//                     length bytes, timeout_ticks                   *
//     Description:    Reads data string from I2C EE memory          *
//                     device. This routine can be used for any I2C  *
//                     EE memory device, which only uses 1 byte of   *
//                     address data as in the 24LC01B/02B/04B/08B.   *
//                                                                   *  
//********************************************************************
/*	//INFO: WORKING BUT NOT USED BY THE PROGRAM
unsigned char HDBytesReadI2C_w_timeout( unsigned char ControlByte, unsigned short address, unsigned char *local_mem_ptr, unsigned char length, int timeout_ticks )
{
  int time_counter;
  unsigned char byte_counter = 0;

  unsigned char HighAdd = (address >> 8) & 0x00FF;
  unsigned char LowAdd = address & 0x00FF;

  IdleI2C_w_timeout( timeout_ticks );                      // ensure module is idle
  SSPCON2bits.SEN=1; 		//StartI2C();         // initiate START condition
  for(time_counter=0; SSPCON2bits.SEN && time_counter < timeout_ticks; time_counter++);   // wait until start condition is over 
  WriteI2C_w_timeout( ControlByte, timeout_ticks );        // write 1 byte 
  WriteI2C_w_timeout( HighAdd, timeout_ticks );            // WRITE word address to EEPROM
  while ( SSPCON2bits.RSEN );     // wait until re-start condition is over 
  WriteI2C_w_timeout( LowAdd, timeout_ticks );             // WRITE word address to EEPROM
  SSPCON2bits.RSEN=1;	//RestartI2C();                   // generate I2C bus restart condition
  for(time_counter=0; SSPCON2bits.RSEN && time_counter < timeout_ticks; time_counter++);     // wait until re-start condition is over 
  WriteI2C_w_timeout( ControlByte | 0x01 , timeout_ticks ); // WRITE 1 byte - R/W bit should be 1 for read

  local_mem_ptr[0] = ReadI2C_w_timeout(timeout_ticks);	//read the first byte from the EEPROM
  byte_counter=1;	//the first byte was read from the EEPROM, initialize byte_counter on 1
  if( length > 1 ) {
	  do {
	
		//---- AckI2C_w_timeout { ...
	  	SSPCON2bits.ACKDT = 0;    // set acknowledge bit state for ACK        
	    SSPCON2bits.ACKEN = 1;    // initiate bus acknowledge sequence
		for(time_counter=0; SSPCON2bits.ACKEN && time_counter < timeout_ticks; time_counter++);   // wait until ACK sequence is over
		// ... } AckI2C_w_timeout  -----
		
		local_mem_ptr[byte_counter] = ReadI2C_w_timeout(timeout_ticks);	//read the next byte from the EEPROM
	    byte_counter++;
	 }
	 while(byte_counter < length);
  }

  NotAckI2C_w_timeout( timeout_ticks );                    // send not ACK condition
  for(time_counter=0; SSPCON2bits.ACKEN && time_counter < timeout_ticks; time_counter++);	// wait until ACK sequence is over 
  SSPCON2bits.PEN=1;	//StopI2C();                      // send STOP condition
  for(time_counter=0; SSPCON2bits.PEN && time_counter < timeout_ticks; time_counter++);     // wait until stop condition is over 
  return ( byte_counter );                   // return the data byte that was read from the EEPROM by I2C
}
*/



//********************************************************************
// Function Name:   ReadCalibTableColumnI2C                          *
// Parameters:   EE memory ControlByte, channel_number,column_number *
//                calib_table_RAM_ptr, number_of_rows                *
// Description:  Reads a column of a calib_table from I2C EE memory  *
//                     device. This routine can be used for any I2C  *
//                     EE memory device, which only uses 1 byte of   *
//                     address data as in the 24LC01B/02B/04B/08B.   *
//                                                                   *  
//********************************************************************
unsigned char ReadCalibTableColumnI2C( unsigned char channel_number, unsigned char column_number, float *calib_table_RAM_ptr, unsigned char number_of_rows )
{
  int time_counter;
  unsigned char byte_counter = 0;
  unsigned char current_row, current_column;

  unsigned char ControlByte = EEPROM_I2C_ADDR;
  int timeout_ticks = TIMEOUT_TICKS_1;

  unsigned short address;
  unsigned char HighAdd, LowAdd;

  float value;
  unsigned char *data_ptr;

  if( column_number == 0 ) {
    address = (channel_number*SIZE_CALIB_TABLE_BYTES_const) + TABLE_HEADER_SIZE_BYTES;  	//the read address is calculated as the start of each calibration table plus and an offset of 0 bytes in case is requested the 1st column (the RAW_values column) or an offset of (SIZE_CALIB_TABLE_BYTES_const/2) bytes in case is requested the 2nd column (the measurements column)
  }	
  else {
	address = (channel_number*SIZE_CALIB_TABLE_BYTES_const) + (SIZE_CALIB_TABLE_BYTES_const/2);  	//the read address is calculated as the start of each calibration table plus and an offset of 0 bytes in case is requested the 1st column (the RAW_values column) or an offset of (SIZE_CALIB_TABLE_BYTES_const/2) bytes in case is requested the 2nd column (the measurements column)
  }
																																//in case column_number=0 then it was requested the 1st column (the RAW_values column), in case column_number=1 then it was requested the 2nd column (the measurements column)
  HighAdd = (address >> 8) & 0x00FF;
  LowAdd = address & 0x00FF;

  data_ptr = (unsigned char *) &value;	//data_ptr is the byte pointer of the 4 byte float number saved in value

  if( number_of_rows > 0 ) {

	  IdleI2C_w_timeout( timeout_ticks );                      // ensure module is idle
	  SSPCON2bits.SEN=1; 		//StartI2C();         // initiate START condition
	  for(time_counter=0; SSPCON2bits.SEN && time_counter < timeout_ticks; time_counter++);   // wait until start condition is over 
	  WriteI2C_w_timeout( ControlByte, timeout_ticks );        // write 1 byte 
	  WriteI2C_w_timeout( HighAdd, timeout_ticks );            // WRITE word address to EEPROM
	  while ( SSPCON2bits.RSEN );     // wait until re-start condition is over 
	  WriteI2C_w_timeout( LowAdd, timeout_ticks );             // WRITE word address to EEPROM
	  SSPCON2bits.RSEN=1;	//RestartI2C();                   // generate I2C bus restart condition
	  for(time_counter=0; SSPCON2bits.RSEN && time_counter < timeout_ticks; time_counter++);     // wait until re-start condition is over 
	  WriteI2C_w_timeout( ControlByte | 0x01 , timeout_ticks ); // WRITE 1 byte - R/W bit should be 1 for read
	

	  data_ptr[0] = ReadI2C_w_timeout(timeout_ticks);	//read the first byte from the EEPROM
	  byte_counter=1;	//the first byte was read from the EEPROM, initialize byte_counter on 1
	
	  for(current_row=0; current_row<number_of_rows; current_row++) {		//read a value from the column of the calib_table at a time, until all requested values are read (number_of_rows)
	 
		  do {
			//---- AckI2C_w_timeout { ...
		  	SSPCON2bits.ACKDT = 0;    // set acknowledge bit state for ACK        
		    SSPCON2bits.ACKEN = 1;    // initiate bus acknowledge sequence
			for(time_counter=0; SSPCON2bits.ACKEN && time_counter < timeout_ticks; time_counter++);   // wait until ACK sequence is over
			// ... } AckI2C_w_timeout  -----
			
			data_ptr[byte_counter] = ReadI2C_w_timeout(timeout_ticks);	//read the next byte from the EEPROM
		    byte_counter++;
		 }
		 while(byte_counter < 4);	//each set of 4 byte read is a floating point
		 byte_counter = 0;	//reset to zero the 'byte_counter' so the 4 bytes of the next float number read from the EEPROM are placed in the 'value' variable correctly
		 calib_table_RAM_ptr[current_row]=value;
	  }
	
	  NotAckI2C_w_timeout( timeout_ticks );                    // send not ACK condition
	  for(time_counter=0; SSPCON2bits.ACKEN && time_counter < timeout_ticks; time_counter++);	// wait until ACK sequence is over 
	  SSPCON2bits.PEN=1;	//StopI2C();                      // send STOP condition
	  for(time_counter=0; SSPCON2bits.PEN && time_counter < timeout_ticks; time_counter++);     // wait until stop condition is over 
  }
  return ( byte_counter );                   // return the data byte that was read from the EEPROM by I2C
}




// --------- External EEPROM (I2C) Calibration table allocation ----------
//	The external EEPROM (I2C) is 512Kbits <=> 64KBytes , example 24LC512 (or 24FC512)
//	Line of a calibration table (total of 8bytes): | frequency / ADC_voltage (float, 4bytes) | Measurement (float, 4 bytes) |
//	Each sensor channel has a corresponding calibration table (CH_0, ..., CH_5, ADC_0(CH_6), ..., ADC_3(CH_9).
//	For each sensor channel is available 6400bytes, that is a calibration table with 800 lines.
//	However to increase the speed of searching for the two lines that are closest to a RAW_value, the RAW_values are in 1 table and the Measurements in other table
//   ---------------------------------------------------------------------

//*******************************************************************
//     Function Name:    WriteCalibTableLine                        *
//     Parameters:       channel_number, line_number, RAW_value,	*
//						measurement                              	*
//     Description:      writes to the EEPROM_I2C memory the		*
//						calibration table used to calculate the		*
//						measurements from the RAW_values   			*
//						  								 			*
//                                                                  *  
//********************************************************************
/*	//INFO: WORKING BUT NOT USED BY THE PROGRAM
void WriteCalibTableLine( unsigned char channel_number, unsigned short line_number, float RAW_value, float measurement )
{
	
	unsigned short write_addr_table_RAW_value;
	unsigned short write_addr_table_measurement;	

	unsigned char write_addr_HIGH;
	unsigned char write_addr_LOW;

	unsigned char data_byte;	//data to be written to the EEPROM_I2C memory
	unsigned char *data;
	char i;

	write_addr_table_RAW_value = channel_number*SIZE_CALIB_TABLE_BYTES_const + line_number*4;		//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number
	write_addr_table_measurement = channel_number*SIZE_CALIB_TABLE_BYTES_const + (SIZE_CALIB_TABLE_BYTES_const/2) + line_number*4;	//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number

	data = (unsigned char *) &RAW_value;
	for(i=0; i<4; i++) {
		data_byte = *data;
		HDByteWriteI2C_w_timeout( EEPROM_I2C_READ_ADDR, write_addr_table_RAW_value+i, data_byte, TIMEOUT_TICKS_1 );
	}

	data = (unsigned char *) &measurement;
	for(i=0; i<4; i++) {
		data_byte = *data;
		HDByteWriteI2C_w_timeout( EEPROM_I2C_READ_ADDR, write_addr_table_measurement+i, data_byte, TIMEOUT_TICKS_1 );
	}

return ;
}
*/




//********************************************************************
//     Function Name:    ReadCalibTableLine                          *
//     Parameters:       channel_number, line_number, *RAW_value_ptr,*
//						*measurement_ptr                             *
//     Description:      read from the EEPROM_I2C memory the		 *
//						calibration table used to calculate the		 *
//						measurements from the RAW_values   			 *
//						  								 			 *
//                                                                   *  
//********************************************************************
void ReadCalibTableLine( unsigned char channel_number, unsigned short line_number, float *RAW_value_ptr, float *measurement_ptr )
{
	//unsigned short read_addr_table_RAW_value;	
	//unsigned short read_addr_table_measurement;
    unsigned short read_addr_table_cycle_k;

	//float RAW_value_local, measurement_local;
	float value_float_local;

	unsigned char *data_ptr;
	char i,k;

	// code to read the RAW_value and measurements of the corresponding line from the EEPROM I2C memory and convert the data from byte array to floating point
	for(k=0; k<2; k++) {
		read_addr_table_cycle_k = channel_number*SIZE_CALIB_TABLE_BYTES_const + line_number*4 + ((!k)*TABLE_HEADER_SIZE_BYTES) + (k*(SIZE_CALIB_TABLE_BYTES_const/2));	//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number

		data_ptr = (unsigned char *) &value_float_local;
		for(i=0; i<4; i++) {
			data_ptr[i] = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_cycle_k+i, TIMEOUT_TICKS_1 );
		}

		//save the read values from the EEPROM I2C memory to the variables indicated on the function argument
		if( k==0 ) {
			*RAW_value_ptr = value_float_local;
		}
		else {
			*measurement_ptr = value_float_local;
		}
	}


	// ----- START --- Old source code more readable/understandable but not optimized for saving most program memory space -----
	//read_addr_table_RAW_value = channel_number*SIZE_CALIB_TABLE_BYTES_const + line_number*4 + TABLE_HEADER_SIZE_BYTES;	//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number
	//read_addr_table_measurement = channel_number*SIZE_CALIB_TABLE_BYTES_const + (SIZE_CALIB_TABLE_BYTES_const/2) + line_number*4;	//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number

	//  // code to read the RAW_value of the corresponding line from the EEPROM I2C memory and convert the data from byte array to floating point
	//data_ptr = (unsigned char *) &RAW_value_local;
	//for(i=0; i<4; i++) {
	//	data_ptr[i] = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_RAW_value+i, TIMEOUT_TICKS_1 );
	//}

	//  // code to read the measurements of the corresponding line from the EEPROM I2C memory and convert the data from byte array to floating point
	//data_ptr = (unsigned char *) &measurement_local;
	//for(i=0; i<4; i++) {
	//	data_ptr[i] = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_measurement+i, TIMEOUT_TICKS_1 );
	//}

	//  //save the read values from the EEPROM I2C memory to the variables indicated on the function arguments
	//*RAW_value_ptr = RAW_value_local;
	//*measurement_ptr = measurement_local;
	// ----- END --- Old source code more readable/understandable but not optimized for saving most program memory space -----

return ;
}


//*******************************************************************
//     Function Name:    WriteCalibTableLine_Bytes                  *
//     Parameters:       channel_number, line_number,				*
//						*RAW_value_bytes_ptr,*measurement_bytes_ptr *
//     Description:      writes to the EEPROM_I2C memory a line of	*
//						calibration table used to calculate the		*
//						measurements from the RAW_values   			*
//						  								 			* 
//*******************************************************************
/*		//INFO: WORKING BUT NOT USED BY THE PROGRAM
void WriteCalibTableLine_Bytes( unsigned char channel_number, unsigned short line_number, unsigned char *RAW_value_bytes_ptr, unsigned char *measurement_bytes_ptr)
{
	unsigned short write_addr_table_RAW_value;
	unsigned short write_addr_table_measurement;	

	unsigned char write_addr_HIGH;
	unsigned char write_addr_LOW;

	unsigned char *data;
	char i; 

	write_addr_table_RAW_value = channel_number*SIZE_CALIB_TABLE_BYTES_const + line_number*4 + TABLE_HEADER_SIZE_BYTES;		//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number
	write_addr_table_measurement = channel_number*SIZE_CALIB_TABLE_BYTES_const + (SIZE_CALIB_TABLE_BYTES_const/2) + line_number*4;	//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number

	for(i=0; i<4; i++) {
		HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, write_addr_table_RAW_value+i, RAW_value_bytes_ptr[i], TIMEOUT_TICKS_1 );
	}

	for(i=0; i<4; i++) {
		HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, write_addr_table_measurement+i, measurement_bytes_ptr[i], TIMEOUT_TICKS_1 );
	}

return ;
}
*/


//*******************************************************************
//     Function Name:    ReadCalibTableLine_Bytes                   *
//     Parameters:       channel_number, line_number,				*
//						*RAW_value_bytes_ptr,*measurement_bytes_ptr *
//     Description:      reads from the EEPROM_I2C memory a line of	*
//						calibration table used to calculate the		*
//						measurements from the RAW_values   			*
//						  								 			* 
//*******************************************************************
/*		//INFO: WORKING BUT NOT USED BY THE PROGRAM
void ReadCalibTableLine_Bytes( unsigned char channel_number, unsigned short line_number, unsigned char *RAW_value_bytes_ptr, unsigned char *measurement_bytes_ptr)
{
	unsigned short read_addr_table_RAW_value;
	unsigned short read_addr_table_measurement;	

	unsigned char read_addr_HIGH;
	unsigned char read_addr_LOW;

	char i;

	read_addr_table_RAW_value = channel_number*SIZE_CALIB_TABLE_BYTES_const + line_number*4 + TABLE_HEADER_SIZE_BYTES;		//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number
	read_addr_table_measurement = channel_number*SIZE_CALIB_TABLE_BYTES_const + (SIZE_CALIB_TABLE_BYTES_const/2) + line_number*4;	//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number

	for(i=0; i<4; i++) {
		*(RAW_value_bytes_ptr+i) = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_RAW_value+i, TIMEOUT_TICKS_1 );
	}

	for(i=0; i<4; i++) {
		*(measurement_bytes_ptr+i) = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_measurement+i, TIMEOUT_TICKS_1 );
	}

return ;
}
*/



//*******************************************************************
//     Function Name:    ReadOrWriteCalibTableLine_Bytes            *
//     Parameters:       channel_number, line_number,				*
//						*RAW_value_bytes_ptr,*measurement_bytes_ptr *
//     Description:      reads or write on the EEPROM_I2C memory a  *
//						line of calibration table used to calculate *
//						the measurements from the RAW_values   		*
//						  								 			* 
//*******************************************************************
void ReadOrWriteCalibTableLine_Bytes(unsigned char is_write_bool, unsigned char channel_number, unsigned short line_number, unsigned char *RAW_value_bytes_ptr, unsigned char *measurement_bytes_ptr)
{
	unsigned short addr_table_RAW_value;
	unsigned short addr_table_measurement;	

	unsigned char read_addr_HIGH;
	unsigned char read_addr_LOW;

	char i;

	addr_table_RAW_value = channel_number*SIZE_CALIB_TABLE_BYTES_const + line_number*4 + TABLE_HEADER_SIZE_BYTES;		//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number
	addr_table_measurement = channel_number*SIZE_CALIB_TABLE_BYTES_const + (SIZE_CALIB_TABLE_BYTES_const/2) + line_number*4;	//the write address is calculated from the start of each calibration table plus the offset in bytes of the current line_number

	for(i=0; i<4; i++) {
		if(is_write_bool==1) {
			HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, addr_table_RAW_value+i, RAW_value_bytes_ptr[i], TIMEOUT_TICKS_1 );
			HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, addr_table_measurement+i, measurement_bytes_ptr[i], TIMEOUT_TICKS_1 );
		}
		else {
			*(RAW_value_bytes_ptr+i) = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, addr_table_RAW_value+i, TIMEOUT_TICKS_1 );
			*(measurement_bytes_ptr+i) = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, addr_table_measurement+i, TIMEOUT_TICKS_1 );
		}
	}

return ;
}



//*******************************************************************
//     Function Name: WriteCalibTableHeader                         *
//     Parameters:    calib_table_is_active_bool					*
//					 number_table_lines_HighByte,					*
//					 number_table_lines_LowByte,					*
//					 calibration_mode_multi_or_single_ch,			*
//					 *units_str										*
//     Description:   writes to the EEPROM_I2C memory the header  	*
//					 of a calibration table used to calculate    	*
//					 the measurements from the RAW_values,			*
//					 the header has the units of the calibration and*
//					 the mode where the calibration is valid		*
//					(0-multiple_channel_mode; 1-single_channel_mode)* 
//						  								 			* 
//*******************************************************************
/*	//INFO: WORKING BUT NOT USED BY THE PROGRAM
void WriteCalibTableHeader( unsigned char channel_number, unsigned char calib_table_is_active_bool, unsigned char number_table_lines_HighByte, unsigned char number_table_lines_LowByte, unsigned char calibration_mode_multi_or_single_ch, char *units_str)
{
	unsigned short write_addr_table_header;	

	unsigned char write_addr_HIGH;
	unsigned char write_addr_LOW;

	unsigned char i; 
	short t;

	write_addr_table_header = channel_number*SIZE_CALIB_TABLE_BYTES_const;		//the write address for the table header is on the start of each calibration table

	//1st save the the byte that indicates if calibration table is active (boolean)
	HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, write_addr_table_header+1, calib_table_is_active_bool, TIMEOUT_TICKS_1 );

	//2nd save the size of the calibration table
	HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, write_addr_table_header+2, number_table_lines_HighByte, TIMEOUT_TICKS_1 );
	HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, write_addr_table_header+3, number_table_lines_LowByte, TIMEOUT_TICKS_1 );

	//3rd save the mode of operation
	HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, write_addr_table_header+4, calibration_mode_multi_or_single_ch, TIMEOUT_TICKS_1 );

	//4th save the units string
	for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
		HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, write_addr_table_header+6+i, units_str[i], TIMEOUT_TICKS_1 );
		if(units_str[i]=='\0') { i=2*CALIB_UNITS_SIZE_STR;  }	//in case the end of string ('\0') was written, then stop to write data of the units string
	}

return ;
}
*/


//***********************************************************************************
//     Function Name: WriteCalibTableHeader                          				*
//     Parameters:    channel_number,												*
//					 write_calib_table_arguments_ptr,								*
//					 write_calib_table_arguments: calib_table_is_active_bool(1 byte)*
//					| number_table_lines_HighByte (1 byte) | 						*
//					number_table_lines_LowByte (1 byte) | 							*
//					calibration_mode_multi_or_single_ch (1 byte) |					*
//					units_str (CALIB_UNITS_SIZE_STR bytes)							*
//     Description:   writes to the EEPROM_I2C memory the header  					*
//					 of a calibration table used to calculate    					*
//					 the measurements from the RAW_values,							*
//					 the header has the units of the calibration and				*
//					 the mode where the calibration is valid						*
//					(0-multiple_channel_mode; 1-single_channel_mode)				* 
//						  								 							* 
//***********************************************************************************

//write_calib_table_arguments(data): | 0x00(1 byte), calib_table_is_active_bool (1byte) |  number_table_lines (2 bytes) | 0x00(1 byte), calibration_mode_multi_or_single_ch (1byte) | units_str ( CALIB_UNITS_SIZE_STR bytes)
/*		//INFO: WORKING BUT NOT USED BY THE PROGRAM
void WriteCalibTableHeader( unsigned char channel_number, unsigned char *write_calib_table_arguments_ptr)
{
	unsigned short write_addr_table_header;	

	unsigned char i; 

	write_addr_table_header = channel_number*SIZE_CALIB_TABLE_BYTES_const;		//the write address for the table header is on the start of each calibration table

	//write_calib_table_arguments(data): | 0x00(1 byte), calib_table_is_active_bool (1byte) |  number_table_lines (2 bytes) | 0x00(1 byte), calibration_mode_multi_or_single_ch (1byte) | units_str ( CALIB_UNITS_SIZE_STR bytes)
	for(i=0; i<(4+(2*CALIB_UNITS_SIZE_STR)); i++) {
		HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, write_addr_table_header+i, write_calib_table_arguments_ptr[i], TIMEOUT_TICKS_1 );
	}

return ;
}
*/


//*******************************************************************
//     Function Name: ReadCalibTableHeader                          *
//     Parameters:	 *calib_table_is_active_bool_ptr, 				*
//					 *number_table_lines_HighByte_ptr,				*
//					 *number_table_lines_LowByte_ptr				*
//				     *calibration_mode_multi_or_single_ch_ptr, 		*
//					 *units_str_ptr									*
//     Description:   reads from the EEPROM_I2C memory the header  	*
//					 of a calibration table used to calculate    	*
//					 the measurements from the RAW_values,			*
//					 the header has the units of the calibration and*
//					 the mode where the calibration is valid		*
//					(0-multiple_channel_mode; 1-single_channel_mode)* 
//						  								 			* 
//*******************************************************************
/*	//INFO: WORKING BUT NOT USED BY THE PROGRAM
void ReadCalibTableHeader( unsigned char channel_number, unsigned char *calib_table_is_active_bool_ptr, unsigned char *number_table_lines_HighByte_ptr, unsigned char *number_table_lines_LowByte_ptr, unsigned char *calibration_mode_multi_or_single_ch_ptr, char *units_str_ptr)
{
	unsigned short read_addr_table_header;	

	unsigned char read_addr_HIGH;
	unsigned char read_addr_LOW;

	unsigned char i; 
	short t;

	read_addr_table_header = channel_number*SIZE_CALIB_TABLE_BYTES_const;		//the read address for the table header is on the start of each calibration table

	//1st read the byte that indicates if calibration table is active (boolean)
	*calib_table_is_active_bool_ptr = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+1, TIMEOUT_TICKS_1 );

	//2nd read the size of the calibration table
	*number_table_lines_HighByte_ptr = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+2, TIMEOUT_TICKS_1 );
	*number_table_lines_LowByte_ptr = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+3, TIMEOUT_TICKS_1 );

	//3rd read the mode of operation
	*calibration_mode_multi_or_single_ch_ptr = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+4, TIMEOUT_TICKS_1 );

	//4th read the units string
	if( units_str_ptr !=NULL) {
		for(i=0; i<CALIB_UNITS_SIZE_STR; i++) {
			*(units_str_ptr+i) = HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+6+i, TIMEOUT_TICKS_1 );
		}
	}

return ;
}
*/


//*******************************************************************
//     Function Name: ReadCalibTableHeader                          *
//     Parameters:	 channel_number,								*
//																	*
//     Description:   reads from the EEPROM_I2C memory the header  	*
//					 of a calibration table used to calculate    	*
//					 the measurements from the RAW_values,			*
//					 the header has the units of the calibration and*
//					 the mode where the calibration is valid		*
//					(0-multiple_channel_mode; 1-single_channel_mode)* 
//						  								 			* 
//*******************************************************************

//read_calib_table_arguments(data): | 0x00(1 byte), calib_table_is_active_bool (1byte) |  number_table_lines (2 bytes) | 0x00(1 byte), calibration_mode_multi_or_single_ch (1byte) | units_str ( CALIB_UNITS_SIZE_STR bytes)
/*		//INFO: WORKING BUT NOT USED BY THE PROGRAM
void ReadCalibTableHeader( unsigned char channel_number, unsigned char *read_calib_table_arguments_ptr)
{
	unsigned short read_addr_table_header;	

	unsigned char i; 

	read_addr_table_header = channel_number*SIZE_CALIB_TABLE_BYTES_const;		//the read address for the table header is on the start of each calibration table

	for(i=0; i<(4+(2*CALIB_UNITS_SIZE_STR)); i++) {
		read_calib_table_arguments_ptr[i] =	HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+i, TIMEOUT_TICKS_1 );
	}

return ;
}
*/




//************************************************************************
//     Function Name: ReadOrWriteCalibTableHeader                        *
//     Parameters:	 is_write_bool, channel_number,                      *
//                    calib_table_arguments_ptr							 *
//																		 *
//     Description:   read or write on the EEPROM_I2C memory the header  *
//					 of a calibration table used to calculate    		 *
//					 the measurements from the RAW_values,				 *
//					 the header has the units of the calibration ,   	 *
//					 the mode where the calibration is valid			 *
//					(0-multiple_channel_mode; 1-single_channel_mode),    *
//                  if the calib table is active, and                    *
//                  the number of valid table lines .                    *
//						  								 				 *
//************************************************************************

//calib_table_arguments(data): | 0x00(1 byte), calib_table_is_active_bool (1byte) |  number_table_lines (2 bytes) | 0x00(1 byte), calibration_mode_multi_or_single_ch (1byte) | units_str ( CALIB_UNITS_SIZE_STR bytes)

void ReadOrWriteCalibTableHeader(unsigned char is_write_bool, unsigned char channel_number, unsigned char *calib_table_arguments_ptr)
{
	unsigned short addr_table_header;	

	unsigned char i; 

	addr_table_header = channel_number*SIZE_CALIB_TABLE_BYTES_const;		//the read address for the table header is on the start of each calibration table

	for(i=0; i<(4+(2*CALIB_UNITS_SIZE_STR)); i++) {
		if(is_write_bool==1) {
			HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, addr_table_header+i, calib_table_arguments_ptr[i], TIMEOUT_TICKS_1 );
		}
		else {
			calib_table_arguments_ptr[i] =	HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, addr_table_header+i, TIMEOUT_TICKS_1 );
		}
	}
return ;
}



//************************************************************************
//     Function Name: ReadCalibTableHeaderActiveNRows                    *
//     Parameters:	 channel_number, *calib_table_is_active_bool_ptr,    *
//					 *number_table_lines_ptr .                           *
//																		 *
//     Description: read the information about if: calib table is active,*
//                  and the number of valid table lines,                 *
//                  on the EEPROM_I2C memory from the header of a        * 
//                  calibration table used to calculate          		 *
//					the measurements from the RAW_values .				 *
//************************************************************************
void ReadCalibTableHeaderActiveNRows( unsigned char channel_number, unsigned char *calib_table_is_active_bool_ptr, unsigned short *number_table_lines_ptr)
{
	unsigned short read_addr_table_header;	

	read_addr_table_header = channel_number*SIZE_CALIB_TABLE_BYTES_const;		//the read address for the table header is on the start of each calibration table
	
	*calib_table_is_active_bool_ptr =	HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+1, TIMEOUT_TICKS_1 );

	*number_table_lines_ptr =	( (HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+2, TIMEOUT_TICKS_1 )) << 8 ) & 0xFF00 | ( HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, read_addr_table_header+3, TIMEOUT_TICKS_1 ) & 0x00FF );

	return ;
}

// -----------------------------------------------------



//*******************************************************************
//     Function Name: CalculateMeasurement                          *
//     Parameters:    channel_number, calib_table_number_of_rows,   *
//					 RAW_value 										*
//																	*
//	Return Value: returns 1 case the RAW_value was inside the range *
//				  of calibration table and so the measurement was	*
//				  calculated, returns 0 otherwise					*
//     Description:   Calculates the measurement  of the selected	*
//					channel number using the RAW_value,				*
//					and corresponding calibration table 			*
//						  								 			* 
//*******************************************************************

signed char CalculateMeasurement( unsigned char channel_number, unsigned short calib_table_number_of_rows, float RAW_value, float *measurement_ptr )
{
	unsigned short i;		//"i" number smaller than 255
	float RAW_value_1, RAW_value_2, measurement_1, measurement_2;	//variables to save the RAW_values and measurements read from the calibration table
	float C0, C1;

	*measurement_ptr=NAN;

	ReadCalibTableColumnI2C( channel_number, 0, calib_table_copy_RAW_values_on_RAM, calib_table_number_of_rows );
    ReadCalibTableColumnI2C( channel_number, 1, calib_table_copy_measurements_on_RAM, calib_table_number_of_rows );
	
	for(i=0, RAW_value_1=999999999.0, RAW_value_2=-999999999.0; i < calib_table_number_of_rows; i++) {
		RAW_value_2 = calib_table_copy_RAW_values_on_RAM[i];
		measurement_2 = calib_table_copy_measurements_on_RAM[i];	//always read the calib_table from the EEPROM at once because is much faster than reading line by line.
		//check if the 'RAW_value' is inside the range [RAW_value_1, RAW_value_2]
		if( RAW_value >= RAW_value_1 && RAW_value < RAW_value_2 ) {
			//the measured RAW_value is inside the range of this line of the calibration table, so calculate the measurement using the last 2 lines
			C1 = ( measurement_2 - measurement_1 ) / ( RAW_value_2 - RAW_value_1 );	// "C1"
			C0 = measurement_2 - C1*RAW_value_2;	// "C0"
			*measurement_ptr = C1*RAW_value + C0;
			i=32700;	//set i very large to signal to quit the search loop because the measurement was calculated 
			return 1;	
		}
		else {
			RAW_value_1 = RAW_value_2;	measurement_1 = measurement_2;	//pass the last read values to: RAW_value_1, measurement_1
		}

	}
return -1;
}



//***********************************************************************
//     Function Name: CalculateCounterMeasurement                   	*
//     Parameters:    channel_number, counter_RAW_value,				*
//					 counter_measurement_ptr 							*
//																		*
//	Return Value: 														*
//     Description:   Calculates the measurement  of the selected		*
//				 counter channel number using the counter_RAW_value		*
//					and corresponding calibration constants			 	*
//						  								 			 	* 
//***********************************************************************
void CalculateCounterMeasurement( unsigned char channel_number, long int counter_RAW_value, float *counter_measurement_ptr )
{
	float constants[4];
	float value, power_func;
	unsigned char i, k;	//"I" number smaller than 255
	ReadCalibTableLine( channel_number, COUNTER_CALIB_ROW_NUMBER, constants, constants+1 );
	ReadCalibTableLine( channel_number, COUNTER_CALIB_ROW_NUMBER+1, constants+2, constants+3 );

	//calculates the value of the function: counter_measurement=C3*(counter_RAW_value^3)+C2*(counter_RAW_value^2)+C1*counter_RAW_value+C0
	for(i=0, value=0; i<4; i++) {
		for(k=0, power_func=1; k<i; k++) {
			power_func = power_func*counter_RAW_value;
		}
		value = value + constants[i]*power_func;
	} 

	*counter_measurement_ptr = value;

return ;
}



/** DECLARATIONS ***************************************************/
#if defined(__18CXX)
#pragma code
#endif




//unsigned char config=0x00;

unsigned char i;		//"i" number smaller than 255
char ch_number_var2=0;
long int SENSORS_FREQ[N_LCR_CHANNELS];	//vector were are saved the calculated sensors frequencies [Hz] for the 6 channels
//char tmr0_prescaler_selection;	//selection  number for the timer0 prescaler
//float prescaler_val;	//OBSOLETE: replace by PRESCALER_VAL (define constant)
//#define PRESCALER_VAL 32.0
//#define CONST_AND_PRESCALER_VAL 641.0	//#define CONST_AND_PRESCALER_VAL 22.84		//NOTE: use for CONST_AND_PRESCALER_VAL=641.0 in case of using a prescaler of 1/256 between the input signal T0CKI and the timer0 counter
#define CONST_AND_PRESCALER_VAL_MULTIPLE_CH_MODE 15.29		//constant used to calculate the sensor channel frequency when operating in multiple_channel_mode
#define CONST_AND_PRESCALER_VAL_SINGLE_CH_MODE 15.29	//constant used to calculate the sensor channel frequency when operating in single_channel_mode
float measurements[N_LCR_CHANNELS+N_ADC_CHANNELS];	//calculated measurement from the freq reading for the selected sensor; for example the measurement on a sensor on the selected units(Ex: Lux, Newton)
float ADC_measurements[N_ADC_CHANNELS];	//saves the last read values from the ADC (raw value, [V] - Volt units)
float counters_measurements[N_LCR_CHANNELS];	//measurements for the sensors counters
float ADC_volt_constant;	// constant used to convert ADC raw values to volt, ADC(analogue to digital converter)

float var1_float;
unsigned char var1_uchar;
//int var2_int;
unsigned short var1_ushort;

char sensor_configs;	//byte with the sensor configurations, on each byte is saved a configuration for one sensor channel
//float sensor_trigger_value;	//saved value of the sensor trigger,
//char sensor_state;	//indicates if the sensor was activated by its trigger value (1-ACTIVATED, 0-INACTIVE)
unsigned char is_active_bool, compare_is_greater_than;
char OUT[N_DIGITAL_OUT];	//values of the outputs of the LCR sensors board

unsigned short calib_table_number_of_rows_to_process;		//variable(integer) were are saved the maximum number of valid rows the PIC processor will try to process for calculating the measurement of each channel
char calib_table_is_active_bool;	//boolean variable (0 or 1) that indicates if the calibration table is active and measurement calculation should be processed 

char bool_func_str[bool_func_N_bytes];	//char string used to read from the memory the boolean function used to that determines the values of OUT0, OUT1, OUT2

//unsigned char test_i2c_str[10];	//DEBUG CODE



/********************************************************************
 * Function:        void main(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        Main program entry point.
 *
 * Note:            None
 *******************************************************************/

#if defined(__18CXX)
void main(void)
#else
int main(void)
#endif
{   
    char bool_expr_str[bool_func_N_bytes];  //char string used to store temporarily a boolean expression to be evaluated for determining values of OUT0, OUT1, OUT2

    InitializeSystem();

    #if defined(USB_INTERRUPT)
        USBDeviceAttach();
    #endif


// ------------- INPUTS/OUTPUTS for the ADC -----------------------------------------
TRISAbits.TRISA0=1;		//RA0 is input, ADC_0 on Sensors_Board
TRISAbits.TRISA1=1;		//RA1 is input, ADC_1 on Sensors_Board
TRISAbits.TRISA2=1;		//RA2 is input, ADC_voltage_reference (2.5V reference) on Sensors_Board
TRISAbits.TRISA3=1;		//RA3 is input, VCC_voltage_reference on Sensors_Board
//-----------------------------------------------------------------------------------

// ------------- INPUTS/OUTPUTS for the Sensors -----------------------------------------
TRISAbits.TRISA4=1;		//RA4/T0CKI is input, LCR sensors input on Sensors_Board
TRISAbits.TRISA5=1;		//RA5 is input, RES_SENSOR_1 Resistive_sensor 1, ADC: (AN2)
TRISCbits.TRISC0=0;		//RC0 is output, analogue multiplexer ch selection, pin A
TRISCbits.TRISC1=0;		//RC1 is output, analogue multiplexer ch selection, pin B
TRISCbits.TRISC2=0;		//RC2 is output, analogue multiplexer ch selection, pin C
//SENSOR_4 on Sensors_Board
TRISCbits.TRISC6=0;		//RC6 is output, pin TX for USART / serial port
TRISCbits.TRISC7=1;		//RC7 is input, pin RX for USART / serial port
//TRISBbits.TRISB0=0;		//RB0 is input, SDA on Sensors_Board
//TRISBbits.TRISB1=1;		//RB1 is input, SCL on Sensors_Board
TRISBbits.TRISB2=1;		//RB2 is input, RES_SENSOR_2 Resistive_sensor 2, ADC: (AN8)
//---------------------------------------------------------------------------------------

// ------------- INPUTS/OUTPUTS DIGITAL  -----------------------------------------------
TRISBbits.TRISB0=0;		//RB0 is output, I2C_SDA
TRISBbits.TRISB1=0;		//RB1 is output, I2C_SCK
TRISBbits.TRISB5=0;		//RB0 is output, DE, /RE pin of IC MAX485 for implementing RS485 protocol
TRISBbits.TRISB4=0;		//RB4 is output, OUT_1 on Sensors_Board
TRISBbits.TRISB3=0;		//RB3 is output, OUT_2 on Sensors_Board
//---------------------------------------------------------------------------------------

//configure the ADCON1 registers to select the analogue input pins and the digital I/O pins
ADCON1 = 0x06;       // config ADCON1, set AN0:AN8 to analogue input, AN9:AN12 to digital output

// ----- Configure timer0 to work as counter for the frequency -----

T0CONbits.T08BIT = 0;    //T08BIT=1 8bit counter/timer, T08BIT=0 16bit counter/timer
T0CONbits.T0CS = 1;    // using external CLK source, Clock Source Select bit...0 = Internal clock (FOSC/4) , 1 = external signal from T0CKI
T0CONbits.T0SE=0;  	//=1 "Increment on high-to-low transition on T0CKI pin", =0 "low-to-high"
T0CONbits.PSA=0;		//=1 "Timer0 prescaler is NOT assigned. Timer0 clock input bypasses prescaler.", =0 "Timer0 clock input comes from prescaler output."

T0CONbits.T0PS2=0; 		//T0PS?: Timer0 Prescaler Select bits , =000 => 1:2 Prescale value , =111 => 1:256 Prescale, =110 => 1:128 Prescale value,  =011 => 1:16 Prescale value , =001 => 1:4 Prescale value
T0CONbits.T0PS1=0;
T0CONbits.T0PS0=0;	

T0CONbits.TMR0ON = 1;    // , =1 enables timer
TMR0H = 0x00;             // preset for timer3 MSB register
TMR0L = 0x00;             // preset for timer3 LSB register

INTCONbits.TMR0IF = 0;            // clear timer0 interrupt flag TMR1IF
INTCONbits.TMR0IE  = 0;         // enable(=1) Timer0 interrupts
INTCONbits.GIE = 1;           // bit7 global interrupt enable
INTCONbits.PEIE = 1;          // bit6 Peripheral Interrupt Enable bit...1 = Enables all unmasked peripheral interrupts
// timer0 clock input is RA4/T0CKI, pin 6
// -------------------------------------------------------------------


// ----- Configure timer3 to work as a timer for measurement range of each channel -----
T3CONbits.T3CKPS1 = 1;   // bits 5-4  Prescaler Rate Select bits
T3CONbits.T3CKPS0 = 1;   // bit 4
T3CONbits.T3SYNC = 1;    // bit 2 Timer3 External Clock Input Synchronization Control bit...1 = Do not synchronize external clock input
T3CONbits.TMR3CS = 0;    // bit 1 Timer3 Clock Source Select bit...0 = Internal clock (FOSC/4)
T3CONbits.TMR3ON = 1;    // bit 0 enables timer
TMR3H = 0x00;             // preset for timer3 MSB register
TMR3L = 0x00;             // preset for timer3 LSB register

//INTCON = 0;           // clear the interrupt control register
PIR2bits.TMR3IF = 0;            // clear timer3 interrupt flag TMR3IF
PIE2bits.TMR3IE  =   1;         // enable(=1) Timer3 interrupts
INTCONbits.GIE = 1;           // bit7 global interrupt enable
INTCONbits.PEIE = 1;          // bit6 Peripheral Interrupt Enable bit...1 = Enables all unmasked peripheral interrupts
// -------------------------------------------------------------------


//--------------------------------------

//****************Configure Analog Comparator **********************************
//*** Analog comparator is configured for:
//	* Output is non-inverted
//	* comparator 1 & 2 are configured as independent comparators
//	* comparator interrupt in enabled
	
	//TRISAbits.TRISA0=1;
	// TRISAbits.TRISA3=1;
//	CVRCON=0b10101000; //setting up comparator reference voltage 2.8125V
//	config = COMP_1_2_OP_INV | COMP_INT_REF_SAME_IP | COMP_INT_EN;	//comparators with int ref and default IPS
//	Open_ancomp(config);

//******************************************************************************

// -----  Configure USART -----

// -----  CONFIG FOR 57600 BAUD on USART / serial port  -----
//Initializations set according to the PIC datasheet requirements for USART receive/trans

    RCSTAbits.SPEN = 1;        //setting serial port enabled 
    TXSTAbits.SYNC = 0;            //asynchronous mode
    TXSTAbits.BRGH = 1;        //BRGH high
    TXSTAbits.TXEN = 1;        //enable transmit

  // ----- Open the USART configured as 8N1, 57600 baud, in polled mode -----
  OpenUSART (USART_TX_INT_OFF & USART_RX_INT_OFF & USART_ASYNCH_MODE &
             USART_EIGHT_BIT & USART_CONT_RX & USART_BRGH_HIGH, 51);   //[48000000/(57600*16)] - 1 = 51
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------

//**************** Initialize the Analog to Digital Converter (ADC) **********************************
 OpenADC(  ADC_FOSC_32 &       // A/D clock source set to 32Tosc
           ADC_RIGHT_JUST&     // write the Digital result(10bits) from right, into ADRESH:ADRESL (16bits).
		   ADC_20_TAD,		   // A/D acquisition time: 20TAD (for 10bit conversion at least 12TAD)
           ADC_CH0 &           //set input channel (0), AN0
           ADC_INT_OFF&        //ADC interrupt off
           ADC_VREFPLUS_VDD&   // set the supply voltage VDD as reference voltage, V+
           ADC_VREFMINUS_VSS,  // set the supply voltage VSS as reference voltage, V
	   	   7				   // this value is used for setting the Analog and Digital I/O. Make sure that AN0 is chosen as analog input.
);
//****************************************************************************************************
// Always set again the usart interrupt flags after calling OpenUSART(...)
INTCONbits.GIE = 1; // enable general interrupts
INTCONbits.PEIE = 1;// enable peripheral interrupts
RCONbits.IPEN=0;	//Enable only 1(one) level of Interrupt, enable HighISR and disable LowISR
PIE1bits.RCIE = 1;	// Interrupt Enable for the USART, when a byte is received is called interrupt routine
IPR1bits.RCIP=1;	// Interrupt priority bit of RX-USART set to 1 - high priority. 		<--- (changed)

USART_cmd_received_bool=0;  //init value of USART_cmd_received_bool

	// ----- Start I2C Module Setup -----
	OpenI2C(MASTER, SLEW_OFF);	
	SSPADD = 0x63;         		//100kHz Baud clock(0x63) @40MHz
	SSPCON2 = 0x00;   //Clear MSSP Control Bits 
	// ---------------------------------


    while(1)
    {
        #if defined(USB_POLLING)
		// Check bus status and service USB interrupts.
        USBDeviceTasks(); // Interrupt or polling method.  If using polling, must call
        				  // this function periodically.  This function will take care
        				  // of processing and responding to SETUP transactions 
        				  // (such as during the enumeration process when you first
        				  // plug in).  USB hosts require that USB devices should accept
        				  // and process SETUP packets in a timely fashion.  Therefore,
        				  // when using polling, this function should be called 
        				  // regularly (such as once every 1.8ms or faster** [see 
        				  // inline code comments in usb_device.c for explanation when
        				  // "or faster" applies])  In most cases, the USBDeviceTasks() 
        				  // function does not take very long to execute (ex: <100 
        				  // instruction cycles) before it returns.
        #endif
    				  

		// Application-specific tasks.
		// Application related code may be added here, or in the ProcessIO() function.


if(CALCULATE_SENSORS_FREQ >= 0) {	//CALCULATE SENSORS_FREQ IN 2 PARTS; "CALCULATE_SENSORS_FREQ=2" calculate channels N. 0,1,2; "CALCULATE_SENSORS_FREQ=1" calculate channels N. 3,4,5; 



			ReadModeCH(EEPROM_ADDR_MODE_CHANNEL_MULTIPLE_OR_SINGLE, &mode_is_single_channel_bool, &selected_channel_number);
			//check if the read mode_is_single_channel_bool has a valid value(0, 1), if not valid then set the default value that is 0
			if(mode_is_single_channel_bool!=1 && mode_is_single_channel_bool!=0) {
				mode_is_single_channel_bool=0;
				WriteModeCH(EEPROM_ADDR_MODE_CHANNEL_MULTIPLE_OR_SINGLE, mode_is_single_channel_bool, 0);	//save mode configuration to the EEPROM		
			}


			if(CALCULATE_SENSORS_FREQ > 0) {
				// freq=sensor_count/(2*counting_time), if counting_time=100ms, freq=5*sensor_count	
				for(ch_number_var2=(N_LCR_CHANNELS*(2-CALCULATE_SENSORS_FREQ))/2; ch_number_var2<(N_LCR_CHANNELS/CALCULATE_SENSORS_FREQ); ch_number_var2++) {
					ReadCalibTableHeaderActiveNRows( ch_number_var2, &is_active_bool, &calib_table_number_of_rows_to_process);
	
					if(mode_is_single_channel_bool==0) {
						if( calib_table_number_of_rows_to_process > CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE) {
							calib_table_number_of_rows_to_process = CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE;	//select an appropriate (small) maximum number of rows to process when calculating the measurement to avoid overloading the PIC processor
						}
						
						SENSORS_FREQ[ch_number_var2]= CONST_AND_PRESCALER_VAL_MULTIPLE_CH_MODE*(SENSOR_SIGNAL_CYCLES_COUNTERS[ch_number_var2]) ;
					}
					else {
						if( calib_table_number_of_rows_to_process > CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE) {
							calib_table_number_of_rows_to_process = CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE;	//select an appropriate (small) maximum number of rows to process when calculating the measurement to avoid overloading the PIC processor
						}
						if(ch_number_var2==selected_channel_number) {
							//if mode_is_single_channel_bool==1 then is selected single channel mode, only one channel read a a time,
							SENSORS_FREQ[selected_channel_number]= CONST_AND_PRESCALER_VAL_SINGLE_CH_MODE*(SENSOR_SIGNAL_CYCLES_COUNTERS[selected_channel_number]) ;
						}
						else {
							SENSORS_FREQ[ch_number_var2]=0;
						}
					}
				   

					if(mode_is_single_channel_bool==0 || (mode_is_single_channel_bool==1 && ch_number_var2==selected_channel_number) ) {	//call 'CalculateMeasurement(...)' (or 'CalculateCounterMeasurement(...)') for all LCR channels if in multiple_channel_mode, in case of single_channel_mode only call 'CalculateMeasurement(...)' (or 'CalculateCounterMeasurement(...)') if current value of ch_number_var2 matches the value in selected_channel_number
						//only calculate the sensor measurement if the calibration table of this channel is active
						if( SENSORS_FREQ[ch_number_var2] >= 0 && is_active_bool == 1 ) {	//Calculate the measurement of the selected channel number using the correspondent calibration table
							CalculateMeasurement( ch_number_var2, calib_table_number_of_rows_to_process, SENSORS_FREQ[ch_number_var2], &measurements[ch_number_var2] );
						}
		
						if( counters_LCR_sensors[ch_number_var2] >= 0) {	//calculate the measurements for the sensor counters
							CalculateCounterMeasurement( ch_number_var2, counters_LCR_sensors[ch_number_var2], &counters_measurements[ch_number_var2]);						
						} 
					}				
	
	
				}
			}



			if( CALCULATE_SENSORS_FREQ == 0 && ( mode_is_single_channel_bool==0 || (mode_is_single_channel_bool==1 && selected_channel_number>=N_LCR_CHANNELS) ) ) {		//"CALCULATE_SENSORS_FREQ=0" read and calculate ADC channels 0,1,2,3 (channels 6,7,8,9)
				// used this cyclic event to make a periodic reading of the ADC channels
				for(i=0; i<5; i++) {
						
					switch(i) { case 0: var1_uchar=ADC_CH2; break;  case 1: var1_uchar=ADC_CH0; break;  case 2: var1_uchar=ADC_CH1; break; case 3: var1_uchar=ADC_CH4; break; case 4: var1_uchar=ADC_CH8; break; default: var1_uchar=ADC_CH2; }	// ADC_voltage_reference(2.5V): ADC_CH2 ; ADC_0(CH6): ADC_CH0 ; ADC_1(CH7): ADC_CH1 ; ADC_2(CH8):  ADC_CH4 ; ADC_3(CH9): ADC_CH8 ;
					SetChanADC (var1_uchar);    // choose the ADC channel to read.
					ConvertADC ();             // Start an A/D conversion.
	   				while( BusyADC());         // Wait for completion. when BusyADC is cleared, the conversion finished.
					var1_float=ReadADC();	  //ATTENTION: it is required to save the return value of ReadADC() first on a variable before using it on a calculation because by doing this it avoids a strange compiler error that would make the function return value be always zero '0' !!
					if(i==0) {						
	   					ADC_volt_constant = 2.5 / var1_float;        // Read result, ADC_CH2, and calculate the constant to convert raw ADC to volt
															// The voltage reference used is 2.5V from a precision shunt volt ref
					}
					else {	
						

						if( mode_is_single_channel_bool==0 || (i-1+N_LCR_CHANNELS)==selected_channel_number ) {
							ReadCalibTableHeaderActiveNRows( N_LCR_CHANNELS+i-1, &is_active_bool, &calib_table_number_of_rows_to_process);	
							ADC_measurements[i-1] = ADC_volt_constant * var1_float;        // Read result, ADC_CH0, ADC_CH1, ADC_CH4, ADC_CH8

							if( is_active_bool == 1 ) { 	//only calculate the sensor measurement if the calibration table of this channel is active
								if( mode_is_single_channel_bool==1 ) {
									var1_ushort = ADC_CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_SINGLE_MODE;
								}	
								else {
									var1_ushort = ADC_CALIB_TABLE_MAX_ROWS_TO_PROCESS_ON_MULTI_MODE;
								}
								if( calib_table_number_of_rows_to_process > var1_ushort ) {
									calib_table_number_of_rows_to_process = var1_ushort;
								}
								CalculateMeasurement( N_LCR_CHANNELS+i-1, calib_table_number_of_rows_to_process, ADC_measurements[i-1], &measurements[N_LCR_CHANNELS+i-1] );
							}
						}
						else {
							if( i > 0) {
								ADC_measurements[i-1] = 0;		//set '0' for the ADC channel (i-1) because it is disabled, since in single_channel_mode only on channel is read at once
								measurements[i-1+N_LCR_CHANNELS] =	0;	//set '0' for the ADC channel (i-1) because it is disabled, since in single_channel_mode only on channel is read at once
							}
						}

					}				

				}

			}

			CALCULATE_SENSORS_FREQ--;	//signal that the frequencies for the LCR channels were calculated	



			/// ------ TEST I2C ---- DEBUG CODE - START -----
			//EEByteWrite(EEPROM_I2C_ADDR, 0x16, 0x5C);
			//
			//HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, 0x0016, 0x5C, 50 );	 //test a write operations
			//for(i=0; i<10000; i++);	//wait some ms between write and read
			//HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, 0x0014, test_i2c_str, 8 , 100 );
			//for(i=0; i<10000; i++);	//wait some ms between write and read
			//HDByteWriteI2C_w_timeout( EEPROM_I2C_ADDR, 0xFFFD, 0xE9, 50 );	 //test a write operations
			//for(i=0; i<10000; i++);	//wait some ms between write and read
			//HDByteReadI2C_w_timeout( EEPROM_I2C_ADDR, 0xFFFA, test_i2c_str, 8, 100 ); 
			/// ------ TEST I2C ---- DEBUG CODE - END -----
	


			// ----- update the calibrated measurement values of the LCR sensors ----------------
			for(i=0; i < N_DIGITAL_OUT; i++) {
				ReadOrWriteOutputsConfig(0, i, bool_func_str);
				ReplaceBoolVariablesByValue(bool_func_str, bool_expr_str, bool_func_N_bytes);
				OUT[i] = BoolExpressionEvaluator(bool_expr_str, bool_func_N_bytes);
			}
			//OUT_0=N/A;	//OUT_0 currently not assigned to a pin of the microcontroller, but still can be read by the 'master' by USB or RS485/serial
			if(OUT[1]>=0)
				OUT_1=0x01&OUT[1];
			if(OUT[2]>=0)
				OUT_2=0x01&OUT[2];
			// --------------------------------------------------------------


		}

		


		if( USART_cmd_received_bool > 0 ) {
			//a command was completely received, process it now		
			USART_ProcessModbusRTU(message);  //check if received data is a Modbus command (Multiple-Sensor protocol) and process it

			usart_counter_char=0;
			counter_inactive_read_cycles_usart=0;
			USART_cmd_received_bool=0;

	//		// ---- DEBUG CODE -----	
	//		PIN_DE_RE_MAX485=1; //RS485 protocol: before starting the transmission set the pin DE(/RE) on "driver output enable" (DE=1)	
	//		//write to the USART the reply to the command
	//		for(i=0;i<10;i++) {
	//			putcUSART(message[i]);
	//			while(BusyUSART());  //wait for USART to process each character before next character
	//		}
	//		//putsUSART(msg_str);  // DEBUG CODE
	//		putrsUSART("\r\n");	
	//
	//	   wait_ticks(1000); //wait about 1ms before changing the RS485 driver back to listen mode, to assure the entire command is transmitted on the RS485 bus		
	//		PIN_DE_RE_MAX485=0; //RS485 protocol: after finishing the transmission set the pin DE(/RE) back to listen (DE=0)
	//	 // ----------------------------

		}
		else{
			counter_inactive_read_cycles_usart++;
		}

		//in case a byte hasn't been receive for a long time CloseUSART and OpenUSART, to flush buffer and avoid jammed USART or errors
		if( counter_inactive_read_cycles_usart > 500000 ) {
			
			CloseUSART();
			 // ----- Open the USART configured as 8N1, 57600 baud, in polled mode -----
  			OpenUSART (USART_TX_INT_OFF & USART_RX_INT_OFF & USART_ASYNCH_MODE &
             USART_EIGHT_BIT & USART_CONT_RX & USART_BRGH_HIGH, 51);   //[48000000/(57600*16)] - 1 = 51
			// Always set again the usart interrupt flags after calling OpenUSART(...)
			INTCONbits.GIE = 1; // enable general interrupts
			INTCONbits.PEIE = 1;// enable peripheral interrupts
			RCONbits.IPEN=0;	//Enable only 1(one) level of Interrupt, enable HighISR and disable LowISR
			PIE1bits.RCIE = 1;	// Interrupt Enable for the USART, when a byte is received is called interrupt routine
			IPR1bits.RCIP=1;	// Interrupt priority bit of RX-USART set to 1 - high priority. 		<--- (changed)

			PIN_DE_RE_MAX485=0; //RS485 protocol: periodically set the pin DE(/RE) of MAX485 to listen mode (DE=0)
			// --------------------------------------------------------------------------
			counter_inactive_read_cycles_usart=0;
		}

        ProcessIO();        
    }//end while
}//end main



/********************************************************************
 * Function:        static void InitializeSystem(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        InitializeSystem is a centralize initialization
 *                  routine. All required USB initialization routines
 *                  are called from here.
 *
 *                  User application initialization routine should
 *                  also be called from here.                  
 *
 * Note:            None
 *******************************************************************/
static void InitializeSystem(void)
{
    #if defined(_PIC14E)
        ANSELA = 0x00;
        ANSELB = 0x00;
        ANSELC = 0x00;
        TRISA  = 0x00;
        TRISB  = 0x00;
        TRISC  = 0x00;
        OSCTUNE = 0;
        #if defined (USE_INTERNAL_OSC)
            OSCCON = 0x7C;   // PLL enabled, 3x, 16MHz internal osc, SCS external
            OSCCONbits.SPLLMULT = 1;   // 1=3x, 0=4x
            ACTCON = 0x90;   // Clock recovery on, Clock Recovery enabled; SOF packet
        #else
            OSCCON = 0x3C;   // PLL enabled, 3x, 16MHz internal osc, SCS external
            OSCCONbits.SPLLMULT = 0;   // 1=3x, 0=4x
            ACTCON = 0x00;   // Clock recovery off, Clock Recovery enabled; SOF packet
        #endif
    #endif

    #if (defined(__18CXX) & !defined(PIC18F87J50_PIM) & !defined(PIC18F97J94_FAMILY))
        ADCON1 |= 0x0F;                 // Default all pins to digital
    #elif defined(__C30__) || defined __XC16__
    	#if defined(__PIC24FJ256DA210__) || defined(__PIC24FJ256GB210__)
    		ANSA = 0x0000;
    		ANSB = 0x0000;
    		ANSC = 0x0000;
    		ANSD = 0x0000;
    		ANSE = 0x0000;
    		ANSF = 0x0000;
    		ANSG = 0x0000;
     #elif defined(__dsPIC33EP512MU810__) || defined (__PIC24EP512GU810__)
        	ANSELA = 0x0000;
    		ANSELB = 0x0000;
    		ANSELC = 0x0000;
    		ANSELD = 0x0000;
    		ANSELE = 0x0000;
    		ANSELG = 0x0000;
            
            // The dsPIC33EP512MU810 features Peripheral Pin
            // select. The following statements map UART2 to 
            // device pins which would connect to the the 
            // RX232 transceiver on the Explorer 16 board.

             RPINR19 = 0;
             RPINR19 = 0x64;
             RPOR9bits.RP101R = 0x3;

        #else
        	AD1PCFGL = 0xFFFF;
        #endif        
    #elif defined(__C32__)
        AD1PCFG = 0xFFFF;
    #endif

    #if defined(PIC18F87J50_PIM) || defined(PIC18F46J50_PIM) || defined(PIC18F_STARTER_KIT_1) || defined(PIC18F47J53_PIM)
    	//On the PIC18F87J50 Family of USB microcontrollers, the PLL will not power up and be enabled
    	//by default, even if a PLL enabled oscillator configuration is selected (such as HS+PLL).
    	//This allows the device to power up at a lower initial operating frequency, which can be
    	//advantageous when powered from a source which is not guarantee to be adequate for 48MHz
    	//operation.  On these devices, user firmware needs to manually set the OSCTUNE<PLLEN> bit to
    	//power up the PLL.
        {
            unsigned int pll_startup_counter = 600;
            OSCTUNEbits.PLLEN = 1;  //Enable the PLL and wait 2+ms until the PLL locks before enabling USB module
            while(pll_startup_counter--);
        }
        //Device switches over automatically to PLL output after PLL is locked and ready.
    #endif

    #if defined(PIC18F87J50_PIM)
    	//Configure all I/O pins to use digital input buffers.  The PIC18F87J50 Family devices
    	//use the ANCONx registers to control this, which is different from other devices which
    	//use the ADCON1 register for this purpose.
        WDTCONbits.ADSHR = 1;			// Select alternate SFR location to access ANCONx registers
        ANCON0 = 0xFF;                  // Default all pins to digital
        ANCON1 = 0xFF;                  // Default all pins to digital
        WDTCONbits.ADSHR = 0;			// Select normal SFR locations
    #endif

    #if defined(PIC18F97J94_FAMILY)
        //Configure I/O pins for digital input mode.
        ANCON1 = 0xFF;
        ANCON2 = 0xFF;
        ANCON3 = 0xFF;
        #if(USB_SPEED_OPTION == USB_FULL_SPEED)
            //Enable INTOSC active clock tuning if full speed
            ACTCON = 0x90; //Enable active clock self tuning for USB operation
            while(OSCCON2bits.LOCK == 0);   //Make sure PLL is locked/frequency is compatible
                                            //with USB operation (ex: if using two speed 
                                            //startup or otherwise performing clock switching)
        #endif
    #endif
    
    #if defined(PIC18F45K50_FAMILY)
        //Configure oscillator settings for clock settings compatible with USB 
        //operation.  Note: Proper settings depends on USB speed (full or low).
        #if(USB_SPEED_OPTION == USB_FULL_SPEED)
            OSCTUNE = 0x80; //3X PLL ratio mode selected
            OSCCON = 0x70;  //Switch to 16MHz HFINTOSC
            OSCCON2 = 0x10; //Enable PLL, SOSC, PRI OSC drivers turned off
            while(OSCCON2bits.PLLRDY != 1);   //Wait for PLL lock
            *((unsigned char*)0xFB5) = 0x90;  //Enable active clock tuning for USB operation
        #endif
        //Configure all I/O pins for digital mode (except RA0/AN0 which has POT on demo board)
        ANSELA = 0x01;
        ANSELB = 0x00;
        ANSELC = 0x00;
        ANSELD = 0x00;
        ANSELE = 0x00;
    #endif
    
    #if defined(__32MX460F512L__)|| defined(__32MX795F512L__)
    // Configure the PIC32 core for the best performance
    // at the operating frequency. The operating frequency is already set to 
    // 60MHz through Device Config Registers
    SYSTEMConfigPerformance(60000000);
	#endif

  #if defined(__dsPIC33EP512MU810__) || defined (__PIC24EP512GU810__)

    // Configure the device PLL to obtain 60 MIPS operation. The crystal
    // frequency is 8MHz. Divide 8MHz by 2, multiply by 60 and divide by
    // 2. This results in Fosc of 120MHz. The CPU clock frequency is
    // Fcy = Fosc/2 = 60MHz. Wait for the Primary PLL to lock and then
    // configure the auxiliary PLL to provide 48MHz needed for USB 
    // Operation.

	PLLFBD = 38;				/* M  = 60	*/
	CLKDIVbits.PLLPOST = 0;		/* N1 = 2	*/
	CLKDIVbits.PLLPRE = 0;		/* N2 = 2	*/
	OSCTUN = 0;			

    /*	Initiate Clock Switch to Primary
     *	Oscillator with PLL (NOSC= 0x3)*/
	
    __builtin_write_OSCCONH(0x03);		
	__builtin_write_OSCCONL(0x01);
	while (OSCCONbits.COSC != 0x3);       

    // Configuring the auxiliary PLL, since the primary
    // oscillator provides the source clock to the auxiliary
    // PLL, the auxiliary oscillator is disabled. Note that
    // the AUX PLL is enabled. The input 8MHz clock is divided
    // by 2, multiplied by 24 and then divided by 2. Wait till 
    // the AUX PLL locks.

    ACLKCON3 = 0x24C1;   
    ACLKDIV3 = 0x7;
    ACLKCON3bits.ENAPLL = 1;
    while(ACLKCON3bits.APLLCK != 1); 

    #endif

    #if defined(PIC18F46J50_PIM) || defined(PIC18F_STARTER_KIT_1) || defined(PIC18F47J53_PIM)
	//Configure all I/O pins to use digital input buffers.  The PIC18F87J50 Family devices
	//use the ANCONx registers to control this, which is different from other devices which
	//use the ADCON1 register for this purpose.
    ANCON0 = 0x7F;                  // All pins to digital (except AN7: temp sensor)
    ANCON1 = 0xBF;                  // Default all pins to digital.  Bandgap on.

    #endif
    
   #if defined(PIC24FJ64GB004_PIM) || defined(PIC24FJ256DA210_DEV_BOARD)
	//On the PIC24FJ64GB004 Family of USB microcontrollers, the PLL will not power up and be enabled
	//by default, even if a PLL enabled oscillator configuration is selected (such as HS+PLL).
	//This allows the device to power up at a lower initial operating frequency, which can be
	//advantageous when powered from a source which is not guarantee to be adequate for 32MHz
	//operation.  On these devices, user firmware needs to manually set the CLKDIV<PLLEN> bit to
	//power up the PLL.
    {
        unsigned int pll_startup_counter = 600;
        CLKDIVbits.PLLEN = 1;
        while(pll_startup_counter--);
    }

    //Device switches over automatically to PLL output after PLL is locked and ready.
    #endif


//	The USB specifications require that USB peripheral devices must never source
//	current onto the Vbus pin.  Additionally, USB peripherals should not source
//	current on D+ or D- when the host/hub is not actively powering the Vbus line.
//	When designing a self powered (as opposed to bus powered) USB peripheral
//	device, the firmware should make sure not to turn on the USB module and D+
//	or D- pull up resistor unless Vbus is actively powered.  Therefore, the
//	firmware needs some means to detect when Vbus is being powered by the host.
//	A 5V tolerant I/O pin can be connected to Vbus (through a resistor), and
// 	can be used to detect when Vbus is high (host actively powering), or low
//	(host is shut down or otherwise not supplying power).  The USB firmware
// 	can then periodically poll this I/O pin to know when it is okay to turn on
//	the USB module/D+/D- pull up resistor.  When designing a purely bus powered
//	peripheral device, it is not possible to source current on D+ or D- when the
//	host is not actively providing power on Vbus. Therefore, implementing this
//	bus sense feature is optional.  This firmware can be made to use this bus
//	sense feature by making sure "USE_USB_BUS_SENSE_IO" has been defined in the
//	HardwareProfile.h file.    
    #if defined(USE_USB_BUS_SENSE_IO)
    tris_usb_bus_sense = INPUT_PIN; // See HardwareProfile.h
    #endif
    
//	If the host PC sends a GetStatus (device) request, the firmware must respond
//	and let the host know if the USB peripheral device is currently bus powered
//	or self powered.  See chapter 9 in the official USB specifications for details
//	regarding this request.  If the peripheral device is capable of being both
//	self and bus powered, it should not return a hard coded value for this request.
//	Instead, firmware should check if it is currently self or bus powered, and
//	respond accordingly.  If the hardware has been configured like demonstrated
//	on the PICDEM FS USB Demo Board, an I/O pin can be polled to determine the
//	currently selected power source.  On the PICDEM FS USB Demo Board, "RA2" 
//	is used for	this purpose.  If using this feature, make sure "USE_SELF_POWER_SENSE_IO"
//	has been defined in HardwareProfile - (platform).h, and that an appropriate I/O pin 
//  has been mapped	to it.
    #if defined(USE_SELF_POWER_SENSE_IO)
    tris_self_power = INPUT_PIN;	// See HardwareProfile.h
    #endif

    UserInit();
    
    USBDeviceInit();	//usb_device.c.  Initializes USB module SFRs and firmware
    					//variables to known states.
}//end InitializeSystem



/******************************************************************************
 * Function:        void UserInit(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        This routine should take care of all of the demo code
 *                  initialization that is required.
 *
 * Note:            
 *
 *****************************************************************************/
void UserInit(void)
{
    
    //initialize the variable holding the handle for the last
    // transmission
    USBOutHandle = 0;
    USBInHandle = 0;

}//end UserInit

/********************************************************************
 * Function:        void ProcessIO(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        This function is a place holder for other user
 *                  routines. It is a mixture of both USB and
 *                  non-USB tasks.
 *
 * Note:            None
 *******************************************************************/

void ProcessIO(void)
{
	signed char byte_number_to_be_sent;	//"byte_number_to_be_sent" number smaller than 127  

    // User Application USB tasks
    if((USBDeviceState < CONFIGURED_STATE)||(USBSuspendControl==1)) return;
    
    //Check if we have received an OUT data packet from the host
    if(!HIDRxHandleBusy(USBOutHandle))				
    {   
		if(!HIDTxHandleBusy(USBInHandle))
        {
			ProcessModbusRTU(ReceivedDataBuffer, &byte_number_to_be_sent);	//process the data received by USB with the Modbus protocol
        	USBInHandle = HIDTxPacket(HID_EP,(BYTE*)&ToSendDataBuffer,SEND_DATA_BUFFER_SIZE);
        }        

        //Re-arm the OUT endpoint, so we can receive the next OUT data packet 
        //that the host may try to send us.
        USBOutHandle = HIDRxPacket(HID_EP, (BYTE*)&ReceivedDataBuffer, RECEIVE_DATA_BUFFER_SIZE);
    }

    
}//end ProcessIO




// FUNCTION: BoolExpressionEvaluator  -  Function to evaluate the logical expression, this version expects bool expressions that are not using commas ',' between operators
//			Return Value: Returns a bool value('0' or '1' represented on the char type) that is the result of evaluating the logical(bool) expression.
// BOOL EXPRESSION - LOGICAL OPERATORS:  '/' - NOT ;   '.' - AND ;  '+' - OR . PRIORITY: '(' - OPEN/START ;  ')' - CLOSE/END
// BOOL EXPRESSION - ARGUMENTS: '0' - FALSE ;  '1' - TRUE .
char BoolExpressionEvaluator(char *bool_expr_str_ptr, unsigned char str_max_size)
{
	int i;
    stack_array arr;
	init_stack(arr);

	bool_expr_str_ptr[str_max_size-1]='\0';	//make sure the 'bool_expr_str_ptr' has at least one '\0' string terminator
 
    // traversing string from the end.
    for (i = strlen(bool_expr_str_ptr) - 1; i >= 0; i--)
    {
        if (bool_expr_str_ptr[i] == '(')
        {
			char s_vector[bool_func_N_bytes];
			char s_vector_index=0;	//the index of the next free/empty location on 's_vector'
            while (top_stack(arr) != ')')
            {
				s_vector[s_vector_index] = top_stack(arr);
				s_vector_index++;
                pop_stack(&arr);
            }
            pop_stack(&arr);
 
			// for single bool value ('0' or '1') inside parenthesis
			if(s_vector_index == 1 && (s_vector[0]=='0' || s_vector[0]=='1') ) {
				push_stack(s_vector[0], &arr);
			}

            // for NOT operation
            if (s_vector_index == 2)
            {
                s_vector[1] == '1' ? push_stack('0', &arr) : push_stack('1', &arr);
            }
            // for AND and OR operation
            else if (s_vector_index == 3)
            {
				//method N1 to evaluate bool operations: arg1.arg2 and arg1+arg2. This method slower and takes excessive program memory space.
                //arg1 = s_vector[0] - 48;  arg2 = s_vector[2] - 48;	
                //result = (s_vector[1] == '.' ? (arg1 && arg2) : (arg1 || arg2) );
                //push_stack((char)result + 48, &arr);
				//----------

				//method N2 to evaluate bool operations: arg1.arg2 and arg1+arg2. This method much faster and less program memory space.
				push_stack( (s_vector[1] == '.' ? (s_vector[0] & s_vector[2]) : (s_vector[0] | s_vector[2]) ), &arr);	
            }
        }
        else
        {
            push_stack(bool_expr_str_ptr[i], &arr);
        }
    }
    return top_stack(arr);
}
 


// FUNCTION: ReplaceBoolVariablesByValue  -  Function replaces all boolean variables by their corresponding value at the current moment
//			Return Value: void  (The output of the function is to be saved on the string variable pointed by 'char *bool_expr_str_ptr')
// BOOL VARIABLES:  'R' - raw value (frequency or ADC) ;  'M' - sensor measurement (same units of the sensor calibration of correspondent sensor channel) ;  'C' - raw counter value ;  'K' - counter measurement (same units of the counter calibration of correspondent sensor channel)
void ReplaceBoolVariablesByValue(char *bool_func_str_ptr, char *bool_expr_str_ptr, unsigned char str_max_size) {
	char pos=0, bool_expr_pos=0;
	char sensor_logic_state=CONST_NEG_ONE;
	float sensor_val_for_logic=CONST_NEG_ONE;	//sensor value that will be used to calculate the sensor state 'sensor_logic_state' (boolean)
	char next_char_numeric_value, compare_is_greater_than_local;	//when reading a sensor logic variable, the 'next_char_numeric_value' is the variable that stores the char after the character indicating the type of sensor value used for the sensor logic variable
	char sensor_configs;	//byte with the sensor configurations, on each byte is saved a configuration for one sensor channel
	float sensor_trigger_value;	//saved value of the sensor trigger,

	for(pos=0; pos<str_max_size && bool_func_str_ptr[pos]!='\0'; pos++, sensor_val_for_logic=CONST_NEG_ONE) {

		next_char_numeric_value = bool_func_str_ptr[pos+1]-'0';    //converts the next character to a value (0 up to 9), in case it is a character of '0' to '9'.
		
		if(next_char_numeric_value>=0 && next_char_numeric_value<(N_LCR_CHANNELS+N_ADC_CHANNELS)) {
			switch(bool_func_str_ptr[pos]) {		//process the 1st character of the sensor logic variable that indicates the type of sensor value to use on the comparison with 'sensor_trigger_value'

				case 'R':	//case of raw value (frequency or ADC)
					if( next_char_numeric_value < N_LCR_CHANNELS)
						sensor_val_for_logic = (float) SENSORS_FREQ[next_char_numeric_value];	// next_char_numeric_value E [0;5]	
					else
						sensor_val_for_logic = ADC_measurements[next_char_numeric_value-N_LCR_CHANNELS];	// next_char_numeric_value E [6;9]
					break;

				case 'M':	//case of measurement
					sensor_val_for_logic = measurements[next_char_numeric_value];	// next_char_numeric_value E [0;9]
					break;

				case 'C':	//case of raw counter value
					sensor_val_for_logic = (float) counters_LCR_sensors[next_char_numeric_value];	// next_char_numeric_value E [0;5]
					break;

				case 'K':	//case of counter measurement
					sensor_val_for_logic = counters_measurements[next_char_numeric_value];	// next_char_numeric_value E [0;5]
					break;
			}

		}

		if(sensor_val_for_logic >= 0) {
				READ_SENSOR_CONFIG(EEPROM_ADDR_CHANNELS_CONFIGS, next_char_numeric_value, &sensor_configs, &sensor_trigger_value);
				//ReadSensorConfig(EEPROM_ADDR_CHANNELS_CONFIGS, next_char_numeric_value, &sensor_configs, &sensor_trigger_value);
				compare_is_greater_than_local = (sensor_configs&0x40)>0;

				//this is: compare_is_greater_than_local == 1 => sensor_val_for_logic > sensor_trigger_value
				//this is: compare_is_greater_than_local == 0 => sensor_val_for_logic <= sensor_trigger_value
				//this is XNOR(compare_is_greater_than_local , sensor_val_for_logic > sensor_trigger_value)
				//sensor_logic_state = NotFunc( compare_is_greater_than_local ^ (sensor_val_for_logic > sensor_trigger_value) );
				sensor_logic_state = 0x01&(~ ( compare_is_greater_than_local ^ (sensor_val_for_logic > sensor_trigger_value) ) );

				bool_expr_str_ptr[bool_expr_pos] = sensor_logic_state + '0';    //converts 'sensor_logic_state' from bool value (0 or 1), to a character of '0' or '1'.
				bool_expr_pos++;
				pos++;	//extra increment of 'pos' (the index of 'bool_func_str_ptr') since a sensor logic variable (that is 2 characters) was now processed/replaced.
			}
		else {
			//current character read from bool function is not of a sensor logic variable (ex: 'R3', 'M2', 'C5', 'K4'), so just copy it to the bool expression to evaluate
			bool_expr_str_ptr[bool_expr_pos]=bool_func_str_ptr[pos];
			bool_expr_pos++;
		}
		
	}

	bool_expr_str_ptr[bool_expr_pos]='\0';
}




//Calculate CRC16
#define POLY 0xA001
unsigned int crc16( unsigned char *p, unsigned char n )
{
	unsigned char  i;
	unsigned short crc = 0xFFFF;

	//unsigned short buffer;

	while (n--)
	{
		crc ^= *p++;
		for (i = 8; i != 0; i--)
		{
			if(crc & 1)
				crc = (crc >> 1) ^ POLY;
			else
				crc >>= 1;
		}
	}

	//switch byte order, from little-endian to big-endian
	//buffer = ( (crc << 8) & 0xFF00 );
	//crc = buffer | ((crc >> 8) & 0x00FF) ;

	return (crc);
}

void FillModbusRTU_8byte_reply(unsigned char * buffer_reply_ptr, unsigned char msg_dest_device_ID, unsigned char function_code, int reg_starting_address, int data_or_N_registers_written) {
	unsigned short CRC;

	buffer_reply_ptr[0]=msg_dest_device_ID;
	buffer_reply_ptr[1]=function_code;
	buffer_reply_ptr[2]=0;	//register_starting_address
	buffer_reply_ptr[3]=reg_starting_address;	//register_starting_address
	buffer_reply_ptr[4]=0;	//number of registers written
	buffer_reply_ptr[5]=data_or_N_registers_written;	//number of registers written
	//CRC = crc16( buffer_reply_ptr, 6 );	//commented because CRC insertion was passed to the end of Modbus processing, to avoid filling program memory with repeated identical code
	//buffer_reply_ptr[6] = CRC&0x00FF;
	//buffer_reply_ptr[7] = (CRC>>8)&0x00FF;
	//byte_number = 8;
}


// Fill_Reply_Data_Channels_ModbusRTU - Fill a char buffer with the bytes of a float array, requested_number_of_channels - is the number of float variables to be filled, N_CHANNELS - is the size of the float array "Data_Channels_Vector_ptr"
//vector_data_type: 'l'->long int, 'f'->float
void Fill_Reply_Data_Channels_ModbusRTU(unsigned char *buffer_reply_ptr, unsigned char msg_dest_device_ID, unsigned char function_code, int requested_number_of_channels, void *Data_Channels_Vector_ptr, char vector_data_type, int N_CHANNELS, signed char *byte_number_ptr) {
	unsigned char byte_number;	//"byte_number" number smaller than 255	

	unsigned char *data;	//pointer to the data to be sent by the USB interface
	unsigned short CRC;
	long int *long_int_data_ptr;
	float *float_data_ptr;
	long int long_int_data;
	float float_data;

	switch(vector_data_type) {
		case 'l':
			long_int_data_ptr = (long int *) Data_Channels_Vector_ptr;
			break;
		case 'f':
			float_data_ptr = (float *) Data_Channels_Vector_ptr;
			break;
	}

	buffer_reply_ptr[0] = msg_dest_device_ID;
	buffer_reply_ptr[1] = function_code;
	//buffer_reply_ptr[2] = length_or_data*2;	//each requested position by the ModbusRTU protocol is of an register with 16bits (2 bytes)

	for(ch_number_var2=0,byte_number=3; ch_number_var2<N_CHANNELS && ch_number_var2<requested_number_of_channels; ch_number_var2++, byte_number=byte_number+4) {
		
		if( vector_data_type == 'l' ) {
			long_int_data = long_int_data_ptr[ch_number_var2]; //this is a long integer of the value from the selected channel
			data = (unsigned char *) &long_int_data;	//save the address of the data to be sent
		}
		else {
			float_data = float_data_ptr[ch_number_var2]; //this is a float of the value from the selected channel
			data = (unsigned char *) &float_data;	//save the address of the data to be sent
		}

		buffer_reply_ptr[byte_number]=data[1];		//send byte #0 of calib_const
		buffer_reply_ptr[byte_number+1]=data[0];		//send byte #1 of calib_const
		buffer_reply_ptr[byte_number+2]=data[3];		//send byte #2 of calib_const
		buffer_reply_ptr[byte_number+3]=data[2];		//send byte #3 of calib_const
		}

	buffer_reply_ptr[2] = byte_number-3; //fill the return_byte_number field, number of bytes of the "data" of the reply
	//calculate the CRC of the reply
	//CRC = crc16( buffer_reply_ptr, byte_number );	//commented because CRC insertion was passed to the end of Modbus processing, to avoid filling program memory with repeated identical code
	//buffer_reply_ptr[byte_number] = CRC&0x00FF;
	//buffer_reply_ptr[byte_number+1] = (CRC>>8)&0x00FF;
	byte_number = byte_number + 2;	//2 more byte of the CRC that will be inserted later to the end of the modbus reply
	*byte_number_ptr = byte_number;	//number of bytes on the created message to be send by the USART (from Multiple-Sensor to Master(PC))
}

void ConvertByteArray_FloatModbus_To_FloatIEEE(unsigned char *byte_array) {

	unsigned char char_buffer, i;

	for(i=0; i<4; i=i+2) {
		char_buffer = byte_array[0+i];
		byte_array[0+i] = byte_array[1+i];
    	byte_array[1+i] = char_buffer;
	}

}


//#define DATA_BUFFER_SIZE 40
//unsigned char data_buffer[DATA_BUFFER_SIZE];	//OBSOLTE: now using the same buffer as USB for filling the replies to be sent to the  
//  *******************************************************************
// Function:        int ProcessModbusRTU(void)
// PreCondition:    None
// Input:           None
// Output:          Return Value: Returns '1' in case a command with a valid CRC was read, returns '0' in case wasn't read a valid command, returns '-1' in case a command was read but was not addressed to this device (device_ID is not same of this device and is not '255' for broadcast).
// Side Effects:    None
// Overview:        This function processes a ModbusRTU message that was received and fills the ToSendDataBuffer with the response to the ModbusRTU Master
// Note:            None
//  *******************************************************************
char ProcessModbusRTU(unsigned char *message, signed char *byte_number_to_be_sent_ptr) {
	char command;

	//unsigned char debug_buffer[35];	//DEBUG CODE
	long int freq_4byte_int=0;

	unsigned char msg_dest_device_ID;

	unsigned char ID, function_code;
	signed char start_of_RTU_command, last_start_of_RTU_command;	//numbers smaller than 127	
    int reg_starting_address, length_or_data;	//must be integer(int) type
	unsigned char byte_1, byte_2, received_CRC_byte1, received_CRC_byte2;
	signed char byte_number, start_data_byte_number;	//numbers smaller than 127
	unsigned char requested_number, calib_table_line_number;	//numbers smaller than 255	 
	char CRC_is_OK=0;
	unsigned char calib_channel_number;	//number between 0 and 255
	float trigger_value;
	unsigned char *trigger_value_data;

	float calib_table_line_vector[2];	//array with the values of a line of a calibration table ( RAW_value (4 bytes) | measurement (4 bytes) )

	unsigned short CRC;

	this_device_ID = Read_b_eep( EEPROM_ADDR_BOARD_ID );	//read the device_ID from the EEPROM
	
	//try to process all the start of commands inside the buffer in case the CRC is OK then they are valid commands
	//start_of_RTU_command=0;
	//CRC_is_OK=0;
	for(last_start_of_RTU_command = -1,  start_of_RTU_command=0, CRC_is_OK=0;  start_of_RTU_command<RECEIVE_DATA_BUFFER_SIZE && start_of_RTU_command>=0 && CRC_is_OK<1 ; start_of_RTU_command=start_of_RTU_command+4) {

		//search for the start of the command, search for ID and function_code, "byte_number" here is start_of_RTU_command
		byte_number = Find_Any_Start_Modbus_RTU(message+start_of_RTU_command, 64, this_device_ID );
		if( byte_number < 0 || byte_number < last_start_of_RTU_command || byte_number > 254) {
			//in case a start of the command wasn't found considering this device_ID, then try the 255(broadcast) device_ID
			 byte_number = Find_Any_Start_Modbus_RTU(message+start_of_RTU_command, 64, 255 );
			 if( byte_number >=0 && byte_number>last_start_of_RTU_command) {
				start_of_RTU_command = byte_number;
			 }
			 else {
				start_of_RTU_command=start_of_RTU_command+4;
			 }
		}
		else {
			start_of_RTU_command = byte_number;
		}
	
		//only process the command if is addressed to this device_ID or is addressed to any device (device_ID=255)
		if( start_of_RTU_command>=0 && last_start_of_RTU_command != start_of_RTU_command) {	//start_of_RTU_command>=0 means the start of command was found inside the message content

			//the msg_dest_device_ID must be device_ID or 255
			msg_dest_device_ID = (unsigned char) message[start_of_RTU_command];
	
			//after the device_ID is the function_code
			function_code = (unsigned char) message[start_of_RTU_command+1];
			//the next 2 bytes are the address of the 1st register to be read (starting address)
			byte_1 = (unsigned char) message[start_of_RTU_command+2];
			byte_2 = (unsigned char) message[start_of_RTU_command+3];
			reg_starting_address = ( (((int) byte_1) << 8 ) & 0xFF00) | ( byte_2 & 0x00FF ) ;
	
			//the next 2 bytes are the length of the data to be read (number of bytes to read)
			byte_1 = (unsigned char) message[start_of_RTU_command+4];
			byte_2 = (unsigned char) message[start_of_RTU_command+5];
			length_or_data = ( (((int) byte_1) << 8 ) & 0xFF00) | ( byte_2 & 0x00FF ) ;

			if(	function_code==0x03 || function_code==0x04 || function_code==0x06 ) {

				received_CRC_byte1 = (unsigned char) message[start_of_RTU_command+6];
				received_CRC_byte2 = (unsigned char) message[start_of_RTU_command+7];
				//check if the CRC of the message is OK
				CRC = crc16( message+start_of_RTU_command, 6);
				start_data_byte_number=start_of_RTU_command+5; //start to read data after: ID(1byte)|func_code(1byte)|address_first_reg(2bytes)|N_reg_to_write(2bytes)|N_following_data_bytes(1byte)

			}
			else {
				if( function_code==0x10 ) {
					//
					start_data_byte_number=start_of_RTU_command+7; //start to read data after: ID(1byte)|func_code(1byte)|address_first_reg(2bytes)|N_reg_to_write(2bytes)|N_following_data_bytes(1byte)
					//byte_number=start_of_RTU_command+7; //start to read data after: ID(1byte)|func_code(1byte)|address_first_reg(2bytes)|N_reg_to_write(2bytes)|N_following_data_bytes(1byte)
					//for(i=0; i < length_or_data*2 && i < DATA_BUFFER_SIZE; byte_number++, i++ ) {
					//	data_buffer[i]=message[byte_number];
					//}

					received_CRC_byte1 = (unsigned char) message[start_data_byte_number + length_or_data*2];
					received_CRC_byte2 = (unsigned char) message[start_data_byte_number + length_or_data*2+1];

					//check if the CRC of the message is OK
					CRC = crc16( message+start_of_RTU_command, 7+(length_or_data*2));
				}
			}
			
			if( received_CRC_byte1 == (CRC&0x00FF) && received_CRC_byte2 == (CRC>>8)&0x00FF) {
				CRC_is_OK=1;
			}
			else {
				CRC_is_OK=0;
			}

		}	

		//save the position of the search for the start of an ModbusRTU command, "last_start_of_RTU_command"
		if(start_of_RTU_command <= last_start_of_RTU_command) {
			start_of_RTU_command=last_start_of_RTU_command+4;
		}
		else {
			last_start_of_RTU_command = start_of_RTU_command;
		}

	}

		if( CRC_is_OK==1 ) {

			//only process the command if is addressed to this device_ID or is addressed to any device (device_ID=255)
			if( (this_device_ID != msg_dest_device_ID) && (msg_dest_device_ID != 255 ) ) {
				CRC_is_OK=-1;		//in case is not addressed to this device then set CRC_is_OK=-1, to indicate the receive message is not addressed to this device
			}
			else {

				wait_ticks(1000); //wait about 1ms before replying to the serial command, to assure no overlapping occurs on a RS485 bus
	
				//fill here the 1st 2 bytes of the reply because are the same for all Modbus commands, so no need to repeat identical code
				ToSendDataBuffer[0]=msg_dest_device_ID;	
				ToSendDataBuffer[1]=function_code;
	
				//fill here so no need to repeat identical code
				requested_number = (int) length_or_data / 2;	//each requested position is 16bit, each channel value is 32bit, so the number of requested channels is half of the received value
	
				//check if is a read_input_registers command (0x04) or read_holding_registers command (0x03)
				if( function_code == 0x04 || function_code == 0x03 ) {
		
						//REPLY FORMAT: ID | respond_code | returned_byte_num | return_temperature_data | CRC_checkcode
	
					if(length_or_data % 2 == 0) {	//check if is being requested a number of bytes correspondent to complete integers (each integer is 32bit)
						if( reg_starting_address == QUERY_FREQUENCY_MEASUREMENTS_CODE_ModbusRTU) {  //code: 0x10						
							//requested_number = (int) length_or_data / 2;	//commented to avoid repeated code, each requested position is 16bit, each channel value is 32bit, so the number of requested channels is half of the received value
							Fill_Reply_Data_Channels_ModbusRTU(ToSendDataBuffer, msg_dest_device_ID, function_code, requested_number, SENSORS_FREQ, 'l', N_LCR_CHANNELS, byte_number_to_be_sent_ptr);
						}
	
						if( reg_starting_address == QUERY_ADC_MEASUREMENTS_CODE_ModbusRTU) {
							//requested_number = (int) length_or_data / 2;	//commented to avoid repeated code, each requested position is 16bit, each channel value is 32bit, so the number of requested channels is half of the received value
							Fill_Reply_Data_Channels_ModbusRTU( ToSendDataBuffer, msg_dest_device_ID, function_code, requested_number, ADC_measurements, 'f', N_ADC_CHANNELS, byte_number_to_be_sent_ptr);
						}
	
						if( reg_starting_address == QUERY_MEASUREMENTS_CH_CODE_ModbusRTU) { 
							//requested_number = (int) length_or_data / 2;	//commented to avoid repeated code, each requested position is 16bit, each channel value is 32bit, so the number of requested channels is half of the received value
							Fill_Reply_Data_Channels_ModbusRTU(ToSendDataBuffer, msg_dest_device_ID, function_code, requested_number, measurements, 'f', N_LCR_CHANNELS+N_ADC_CHANNELS, byte_number_to_be_sent_ptr);
						}
	
						if( reg_starting_address == QUERY_COUNTERS_MEASUREMENTS_CH_CODE_ModbusRTU) { 
							//requested_number = (int) length_or_data / 2;	//commented to avoid repeated code, each requested position is 16bit, each channel value is 32bit, so the number of requested channels is half of the received value
							Fill_Reply_Data_Channels_ModbusRTU(ToSendDataBuffer, msg_dest_device_ID, function_code, requested_number, counters_measurements, 'f', N_LCR_CHANNELS, byte_number_to_be_sent_ptr);
						}
					}
	
					if( reg_starting_address >= SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU && reg_starting_address < SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU + N_LCR_CHANNELS+N_ADC_CHANNELS) { 
						calib_channel_number = reg_starting_address - SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU;
						calib_table_line_number = (int) length_or_data;	//each requested position is 16bit, each line of calibration table is 64bit (8 bytes), so the number of lines is 1/4 of the received value			
	
						ReadCalibTableLine( calib_channel_number, calib_table_line_number, &calib_table_line_vector[0], &calib_table_line_vector[1] );	//read from I2C EEPROM the requested line of the calibration table
						Fill_Reply_Data_Channels_ModbusRTU(ToSendDataBuffer, msg_dest_device_ID, function_code, 2, calib_table_line_vector, 'f', 2, byte_number_to_be_sent_ptr);	//float_array_size = SEND_DATA_BUFFER_SIZE/4;	//each line contains 2 float variables (RAW_value, measurement)
					}  
	
	
					if( reg_starting_address >= SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU && reg_starting_address < SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU + N_LCR_CHANNELS+N_ADC_CHANNELS) { 
						calib_channel_number = reg_starting_address - SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU;		
	
						//ToSendDataBuffer[0]=msg_dest_device_ID;
						//ToSendDataBuffer[1]=function_code;
						ToSendDataBuffer[2] = 6+(2*CALIB_UNITS_SIZE_STR); //fill the return_byte_number field (1byte), number of bytes of the "data" of the reply				
						//ToSendDataBuffer[3] = sensor_configs;
						//reply_buffer_content: ID(1byte)|func_code(1byte)| N_following_data_bytes(1byte) |
						//reply_buffer_content(data): | 0x00(1 byte), calib_table_is_active_bool (1byte) |  number_table_lines (2 bytes) | 0x00(1 byte), calibration_mode_multi_or_single_ch (1byte) | units_str ( CALIB_UNITS_SIZE_STR bytes)
						ReadOrWriteCalibTableHeader( 0, calib_channel_number, ToSendDataBuffer+3);	//read calib table header
	
						//ReadCalibTableHeader( calib_channel_number, &ToSendDataBuffer[4], &ToSendDataBuffer[5], &ToSendDataBuffer[6], &ToSendDataBuffer[8], &ToSendDataBuffer[9]);	//OBSOLTE
	
						//CRC = crc16( ToSendDataBuffer, 9 + CALIB_UNITS_SIZE_STR );	//commented because CRC insertion was passed to the end of Modbus processing, to avoid filling program memory with repeated identical code
						//ToSendDataBuffer[9 + CALIB_UNITS_SIZE_STR] = CRC&0x00FF;
						//ToSendDataBuffer[10 + CALIB_UNITS_SIZE_STR] = (CRC>>8)&0x00FF;
						*byte_number_to_be_sent_ptr = 11 + (2*CALIB_UNITS_SIZE_STR);
					}  
	
	
					if( reg_starting_address >= SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU && reg_starting_address < SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU+N_LCR_CHANNELS+N_ADC_CHANNELS ) {
						//send the sensors configurations, byte 2 has the channel number to be configured
						// byte 3 has the various operating configurations of the sensor (represented by various bits)
						// byte 4,5,6,7, is a float number with the trigger value of the sensor to be used on calculation of the outputs
						calib_channel_number = reg_starting_address - SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU;
					
						//the sensor_trigger_value is being saved on the calib_constant
						READ_SENSOR_CONFIG(EEPROM_ADDR_CHANNELS_CONFIGS, calib_channel_number, &sensor_configs, &trigger_value);
						//ReadSensorConfig(EEPROM_ADDR_CHANNELS_CONFIGS, calib_channel_number, &sensor_configs, &trigger_value);
		
						//ToSendDataBuffer[0]=msg_dest_device_ID;
						//ToSendDataBuffer[1]=function_code;
						ToSendDataBuffer[2] = 16; //fill the return_byte_number field (1byte), number of bytes of the "data" of the reply				
						//ToSendDataBuffer[3] = sensor_configs;
						//ToSendDataBuffer[3] = 0;	//data_buffer[3] is empty (0 - zero)
						//ToSendDataBuffer[4] = 0;  //ToSendDataBuffer[4] = (sensor_configs>>7)&0x01; //make_compare_w_calibrated_sensor_value, ( 0 - for sensor RAW value, 1 - for sensor calibrated value ) ,  INFO: NOT AVAILABLE / OBSOLETE
						//ToSendDataBuffer[5] = 0;	//data_buffer[5] is empty (0 - zero)
						//ToSendDataBuffer[6] = (sensor_configs>>6)&0x01; //compare_is_greater_than , Select if the sensor channel is active when its value is greater[>] than trigger, or less[<] than trigger.
						//ToSendDataBuffer[7] = 0;	//data_buffer[7] is empty (0 - zero)
						//ToSendDataBuffer[8] = 4; //EMPTY SPACE, (sensor_configs>>5)&0x01; //sensor_prescaler, NOT AVAILABLE / OBSOLETE
						//ToSendDataBuffer[9] = 0;	//data_buffer[7] is empty (0 - zero)
						//ToSendDataBuffer[10] = 0;	//ToSendDataBuffer[10] = (sensor_configs>>2)&0x01;	//sensor_used_for_OUT[0], (0-not used, 1-used) ,  INFO: NOT AVAILABLE / OBSOLETE
						//ToSendDataBuffer[11] = 0;	//data_buffer[7] is empty (0 - zero)
						//ToSendDataBuffer[12] = 0;	//ToSendDataBuffer[12] = (sensor_configs>>1)&0x01;	//sensor_used_for_OUT[1], (0-not used, 1-used) ,  INFO: NOT AVAILABLE / OBSOLETE
						//ToSendDataBuffer[13] = 0;	//data_buffer[7] is empty (0 - zero)
						//ToSendDataBuffer[14] = 0;	//ToSendDataBuffer[14] = sensor_configs&0x01;	//sensor_used_for_OUT[2], (0-not used, 1-used) ,  INFO: NOT AVAILABLE / OBSOLETE
	
						for(byte_1=3; byte_1<=14; byte_1++) {
							ToSendDataBuffer[byte_1] = 0;
						}
						ToSendDataBuffer[6] = (sensor_configs>>6)&0x01; //compare_is_greater_than , Select if the sensor channel is active when its value is greater[>] than trigger, or less[<] than trigger.
	
						// trigger_value is a float variable were is saved the trigger value used to activate the digital output
						trigger_value_data = (unsigned char *) &trigger_value;
						ToSendDataBuffer[15] = trigger_value_data[1];	//this is the correct byte fill order for floating point type in ModbusRTU
						ToSendDataBuffer[16] = trigger_value_data[0];
						ToSendDataBuffer[17] = trigger_value_data[3];
						ToSendDataBuffer[18] = trigger_value_data[2];
		
						// check if the data read from the EEPROM is valid, in case is invalid signal it
						if(trigger_value!=trigger_value) {	//this is the test for NaN
							//fill the data with 255 (0xFF) when the read data was invalid
							//ToSendDataBuffer[3]=255;ToSendDataBuffer[4]=255;ToSendDataBuffer[5]=255;ToSendDataBuffer[6]=255;
							//ToSendDataBuffer[15]=255;ToSendDataBuffer[16]=255;ToSendDataBuffer[17]=255;ToSendDataBuffer[18]=255;
							for(byte_1=0; byte_1<4; byte_1++) {
								ToSendDataBuffer[byte_1+3]=255;
								ToSendDataBuffer[byte_1+15]=255;
							}
	
						//	ToSendDataBuffer[0] = SERIAL_REPLY_WITH_INVALID_DATA;
							trigger_value=0;
						}				
							
						//CRC = crc16( ToSendDataBuffer, 19 );	//commented because CRC insertion was passed to the end of Modbus processing, to avoid filling program memory with repeated identical code
						//ToSendDataBuffer[19] = CRC&0x00FF;
						//ToSendDataBuffer[20] = (CRC>>8)&0x00FF;
						*byte_number_to_be_sent_ptr = 21;						
					}	
	
					if( reg_starting_address == QUERY_OUTPUTS_CURRENT_VALUE_CODE_ModbusRTU ) {
						//send the sensors board outputs ( INACTIVE=0/ACTIVE=1 )
						//each byte is an output, 0x00 = INACTIVE, 0x01 = ACTIVE
	
						//ToSendDataBuffer[0]=msg_dest_device_ID;
						//ToSendDataBuffer[1]=function_code;
						ToSendDataBuffer[2] = 6; //fill the return_byte_number field (1byte), number of bytes of the "data" of the reply				
						//ToSendDataBuffer[3] = sensor_configs;
						ToSendDataBuffer[3] = 0;	//data_buffer[3] is empty (0 - zero)
						ToSendDataBuffer[4] = 0x0001&OUT[0];	//send the OUTPUT N.0
						ToSendDataBuffer[5] = 0;	//data_buffer[5] is empty (0 - zero)
						ToSendDataBuffer[6] = 0x0001&OUT[1];	//send the OUTPUT N.1
						ToSendDataBuffer[7] = 0;	//data_buffer[7] is empty (0 - zero)
						ToSendDataBuffer[8] = 0x0001&OUT[2];	//send the OUTPUT N.1
						
						//CRC = crc16( ToSendDataBuffer, 9 );	//commented because CRC insertion was passed to the end of Modbus processing, to avoid filling program memory with repeated identical code
						//ToSendDataBuffer[9] = CRC&0x00FF;
						//ToSendDataBuffer[10] = (CRC>>8)&0x00FF;
						*byte_number_to_be_sent_ptr = 11;						
					}
	
					
					if( reg_starting_address >= SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU && reg_starting_address < SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU+N_DIGITAL_OUT  ) {
	
						calib_channel_number = reg_starting_address - SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU; 	//calib_channel_number <=> output_number
	
						//ToSendDataBuffer[0]=msg_dest_device_ID;
						//ToSendDataBuffer[1]=function_code;
						ToSendDataBuffer[2] = bool_func_N_bytes; //fill the return_byte_number field (1byte), number of bytes of the "data" of the reply
						ReadOrWriteOutputsConfig(0, calib_channel_number, (char *) ToSendDataBuffer+3);
						
						CRC = crc16( ToSendDataBuffer, 3+bool_func_N_bytes );	//commented because CRC insertion was passed to the end of Modbus processing, to avoid filling program memory with repeated identical code
						ToSendDataBuffer[3+bool_func_N_bytes] = CRC&0x00FF;
						ToSendDataBuffer[4+bool_func_N_bytes] = (CRC>>8)&0x00FF;
						*byte_number_to_be_sent_ptr = 5+bool_func_N_bytes;
													
					} 
	
					if( reg_starting_address == QUERY_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU ) {
						//requested_number = (int) length_or_data / 2; //commented to avoid repeated code, requested_number <=> requested_number_of_channels	//each requested position is 16bit, each channel value is 32bit, so the number of requested channels is half of the received value
						Fill_Reply_Data_Channels_ModbusRTU(ToSendDataBuffer, msg_dest_device_ID, function_code, requested_number, counters_LCR_sensors, 'l', N_LCR_CHANNELS, byte_number_to_be_sent_ptr);
					} 
	
					if( reg_starting_address == SAVE_OR_QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_CODE_ModbusRTU ) {
	
						ReadModeCH(EEPROM_ADDR_MODE_CHANNEL_MULTIPLE_OR_SINGLE, &mode_is_single_channel_bool, &selected_channel_number);	//save mode configuration to the EEPROM
						//ToSendDataBuffer[0] = msg_dest_device_ID;
						//ToSendDataBuffer[1] = function_code;
						ToSendDataBuffer[2] = 2;	//length_or_data
						ToSendDataBuffer[3] = mode_is_single_channel_bool;
						ToSendDataBuffer[4] = selected_channel_number;
						//CRC = crc16( ToSendDataBuffer, 5 );	//commented because CRC insertion was passed to the end of Modbus processing, to avoid filling program memory with repeated identical code
						//ToSendDataBuffer[5] = CRC&0x00FF;
						//ToSendDataBuffer[6] = (CRC>>8)&0x00FF;
						*byte_number_to_be_sent_ptr = 7;						
	
					}
					
	
					if( reg_starting_address == SET_OR_GET_BOARD_ID_CODE_ModbusRTU) {
							//ToSendDataBuffer[0] = msg_dest_device_ID;
							//ToSendDataBuffer[1] = function_code;
							ToSendDataBuffer[2] = 2;	//length_or_data
							ToSendDataBuffer[3] = 0;
							ToSendDataBuffer[4] = this_device_ID;
							//CRC = crc16( ToSendDataBuffer, 5 );	//commented because CRC insertion was passed to the end of Modbus processing, to avoid filling program memory with repeated identical code
							//ToSendDataBuffer[5] = CRC&0x00FF;
							//ToSendDataBuffer[6] = (CRC>>8)&0x00FF;
							*byte_number_to_be_sent_ptr = 7;						
					}
		
				}
		
				//check if is a write_input_registers command (0x06)
				if( function_code == 0x06 ) {
	
				//		if( reg_starting_address == SELECT_MEASUREMENT_TIME_CODE_ModbusRTU ) {
				//			tmr3_measurement_time_selection = length_or_data;  //content of the function write_input_registers
				//			ToSendDataBuffer[0] = msg_dest_device_ID;
				//			ToSendDataBuffer[1] = function_code;
				//			ToSendDataBuffer[2] = reg_starting_address;
				//			ToSendDataBuffer[3] = length_or_data;
				//			ToSendDataBuffer[4] = received_CRC_byte1;
				//			ToSendDataBuffer[5] = received_CRC_byte2;				
				//		} 
	
						if( reg_starting_address == RESET_MULTIPLE_SENSOR_COUNTER_VALUES_CODE_ModbusRTU ) {
							FillModbusRTU_8byte_reply(ToSendDataBuffer, msg_dest_device_ID, function_code, reg_starting_address, 1);
							*byte_number_to_be_sent_ptr = 8;
							//reset the counters_LCR_sensors
							for(i=0;i<N_LCR_CHANNELS;i++) {
								counters_LCR_sensors[i]=0;
							}
						}
	
						if( reg_starting_address == SAVE_OR_QUERY_MULTIPLE_SENSOR_MODE_MULTI_OR_SINGLE_CODE_ModbusRTU ) {
						
							mode_is_single_channel_bool = (length_or_data>>8)&0x0001;	//the channel_mode is on the 1st byte of length_or_data
							selected_channel_number = length_or_data&0x00FF;	//the selected_channel_number is on the 2nd byte of length_or_data
							if(	mode_is_single_channel_bool == 0 || mode_is_single_channel_bool==1) {					
				
								WriteModeCH(EEPROM_ADDR_MODE_CHANNEL_MULTIPLE_OR_SINGLE, mode_is_single_channel_bool, selected_channel_number);	//save mode configuration to the EEPROM		
	
								FillModbusRTU_8byte_reply(ToSendDataBuffer, msg_dest_device_ID, function_code, reg_starting_address, 1);
								*byte_number_to_be_sent_ptr = 8;								
							}
					
						}
	
						if( reg_starting_address == SET_OR_GET_BOARD_ID_CODE_ModbusRTU ) {
							//set the device ID to the new ID received by the command
							Write_b_eep( EEPROM_ADDR_BOARD_ID, length_or_data );  //writes a byte from the EEPROM
							Busy_eep ();     // Wait until write is complete  
							this_device_ID = Read_b_eep( EEPROM_ADDR_BOARD_ID );	//read the device_ID from the EEPROM
							
							FillModbusRTU_8byte_reply(ToSendDataBuffer, msg_dest_device_ID, function_code, reg_starting_address, this_device_ID);
							*byte_number_to_be_sent_ptr = 8;						
						}
	
				}
	
				//check if is a Preset_Multiple_Register (write) command (0x10)
				if( function_code == 0x10 ) {
	
						if( reg_starting_address >= SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU && reg_starting_address < SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU+N_LCR_CHANNELS+N_ADC_CHANNELS ) {
							//save the sensors configurations, byte 3 has the channel number to be configured
							// byte 4 has the various operating configurations of the sensor (represented by various bits)
							// byte 5,6,7,8 is a float number with the trigger value of the sensor to be used on calculation of the outputs
							calib_channel_number = reg_starting_address - SAVE_OR_QUERY_CONFIGURATIONS_SENSOR_CH_CODE_ModbusRTU; //lower byte of the length_or_data
							//sensor_configs = data_buffer[1];
	
							//sensor_configs=0x00;  //start with an empty config
							//data_buffer[0] is empty (0 - zero)
							//sensor_configs = 0x00 | (message[start_data_byte_number+1]<<7)&0x80; //make_compare_w_calibrated_sensor_value, ( 0 - for sensor RAW value, 1 - for sensor calibrated value ) , INFO: NOT AVAILABLE / OBSOLETE
							//data_buffer[2] is empty (0 - zero)
							sensor_configs = 0x00 | (message[start_data_byte_number+3]<<6)&0x40; //compare_is_greater_than , Select if the sensor channel is active when its value is greater[>] than trigger, or less[<] than trigger.
							//sensor_configs = sensor_configs | 0x20;  //sensor_prescaler uses 3 bits, bits_number: 3,4,5, sensor_configs: xx543xxx), default=4, NOT AVAILABLE / OBSOLETE
							//sensor_configs = sensor_configs & 0xE7;  //sensor_prescaler uses 3 bits, bits_number: 3,4,5, sensor_configs: xx543xxx), default=4, NOT AVAILABLE / OBSOLETE
							//data_buffer[4] is empty (0 - zero)
							//sensor_configs = sensor_configs | (message[start_data_byte_number+5]<<2)&0x04; //sensor_used_for_OUT[0], (0-not used, 1-used) ,  INFO: NOT AVAILABLE / OBSOLETE
							//data_buffer[6] is empty (0 - zero)
							//sensor_configs = sensor_configs | (message[start_data_byte_number+7]<<1)&0x02; //sensor_used_for_OUT[1], (0-not used, 1-used) ,  INFO: NOT AVAILABLE / OBSOLETE
							//data_buffer[8] is empty (0 - zero)
							//sensor_configs = sensor_configs | message[start_data_byte_number+9]&0x01; //sensor_used_for_OUT[2], (0-not used, 1-used) ,   INFO: NOT AVAILABLE / OBSOLETE
						
							// trigger_value is a float variable were is saved the trigger value used to activate the digital output
							trigger_value_data = (unsigned char *) &trigger_value;						
							trigger_value_data[0] = message[start_data_byte_number+11]; //appropriate byte order for receiving data from ModbusRTU application
							trigger_value_data[1] = message[start_data_byte_number+10];
							trigger_value_data[2] = message[start_data_byte_number+13];
							trigger_value_data[3] = message[start_data_byte_number+12];
				
							// to store sensor channel configurations, is being used the calibration variables just has a way of reusing variables already declared
							WRITE_SENSOR_CONFIG(EEPROM_ADDR_CHANNELS_CONFIGS, calib_channel_number, &sensor_configs, &trigger_value);
							//WriteSensorConfig(EEPROM_ADDR_CHANNELS_CONFIGS, calib_channel_number, sensor_configs, trigger_value);	
		
							FillModbusRTU_8byte_reply(ToSendDataBuffer, msg_dest_device_ID, function_code, reg_starting_address, 7);
							*byte_number_to_be_sent_ptr = 8;						
						}	
	
					
					if( reg_starting_address >= SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU && reg_starting_address < SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU+N_DIGITAL_OUT ) {
	
						calib_channel_number = reg_starting_address - SAVE_OR_QUERY_OUTPUT_CONFIG_CODE_ModbusRTU; 	//calib_channel_number <=> output_number
	
						ReadOrWriteOutputsConfig(1, calib_channel_number, (char *) message+start_data_byte_number);
	
						FillModbusRTU_8byte_reply(ToSendDataBuffer, msg_dest_device_ID, function_code, reg_starting_address, bool_func_N_bytes/2);
						*byte_number_to_be_sent_ptr = 8;
		
					}   
	
					if( reg_starting_address >= SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU && reg_starting_address < SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU+N_LCR_CHANNELS+N_ADC_CHANNELS ) {
						//save a line of calibration table to the I2C EEPROM , line (RAW_value, measurements)
						calib_channel_number = reg_starting_address - SAVE_OR_QUERY_LINE_CALIB_TABLE_CH_CODE_ModbusRTU; //lower byte of the length_or_data
						//sensor_configs = data_buffer[1];
	
						//DEBUG CODE
						//for(i=0; i< 32; i++) {
						//	debug_buffer[i]=message[i];
						//}	//DEBUG CODE
						
	
						calib_table_line_number = (int) ( (message[start_data_byte_number]<<8)&0xFF00) | ( message[start_data_byte_number+1]&0x00FF ) ;	//the data byte N.0, N.1 are an integer that is the line number to be read from the calib table	
	
						ConvertByteArray_FloatModbus_To_FloatIEEE(&message[start_data_byte_number+2]);
						ConvertByteArray_FloatModbus_To_FloatIEEE(&message[start_data_byte_number+6]);
						ReadOrWriteCalibTableLine_Bytes( 1, calib_channel_number, calib_table_line_number, &message[start_data_byte_number+2], &message[start_data_byte_number+6]);	//write to I2C EEPROM the correspondent line of the calibration table
						FillModbusRTU_8byte_reply(ToSendDataBuffer, msg_dest_device_ID, function_code, reg_starting_address, 5);
						*byte_number_to_be_sent_ptr = 8;
					}
	
					if( reg_starting_address >= SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU && reg_starting_address < SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU+N_LCR_CHANNELS+N_ADC_CHANNELS ) {
						calib_channel_number = reg_starting_address - SAVE_OR_QUERY_HEADER_CALIB_TABLE_CH_CODE_ModbusRTU; //lower byte of the length_or_data
				
						//buffer_content: ID(1byte)|func_code(1byte)| address_first_reg(2bytes) (includes operation code and the calib_channel_number) | N_reg_to_write(2bytes) | N_following_data_bytes(1byte)
						//buffer_content(data): | 0x00(1 byte), calib_table_is_active_bool (1byte) |  number_table_lines (2 bytes) | 0x00(1 byte), calibration_mode_multi_or_single_ch (1byte) | units_str ( CALIB_UNITS_SIZE_STR bytes)
						ReadOrWriteCalibTableHeader( 1, calib_channel_number, message+start_data_byte_number);	//write calib table header
						//WriteCalibTableHeader( calib_channel_number, message[start_data_byte_number+1], message[start_data_byte_number+2], message[start_data_byte_number+3], message[start_data_byte_number+5], &message[start_data_byte_number+6]);	//OBSOLETE
						FillModbusRTU_8byte_reply(ToSendDataBuffer, msg_dest_device_ID, function_code, reg_starting_address, length_or_data);
						*byte_number_to_be_sent_ptr = 8;
					}
					
				}
	
	
				//fill now the CRC bytes of the reply, done here because this part of the protocol is identical for all the Modbus commands and so no need to repeat code
				CRC = crc16( ToSendDataBuffer, *byte_number_to_be_sent_ptr - 2 );
				ToSendDataBuffer[*byte_number_to_be_sent_ptr - 2] = CRC&0x00FF;
				ToSendDataBuffer[*byte_number_to_be_sent_ptr - 1] = (CRC>>8)&0x00FF;

			}

		}

 return CRC_is_OK;
}


// *******************************************************************
// Function:        int USART_ProcessModbusRTU(void)
// PreCondition:    None
// Input:           None
// Output:          Return Value: 
// Side Effects:    None
// Overview:        This function requests to be processed the Modbus message and send the response by the RS485/USART interface
// Note:            None
// *******************************************************************
void USART_ProcessModbusRTU(unsigned char *message) {

	char CRC_is_OK;
	signed char byte_number_to_be_sent;	//"byte_number_to_be_sent" number smaller than 127

	CRC_is_OK = ProcessModbusRTU(message, &byte_number_to_be_sent); //request to process the received modbus message to fill the ToSendDataBuffer with the reply to the Modbus Master
	if( CRC_is_OK == 1 ) {
		PIN_DE_RE_MAX485=1; //RS485 protocol: before starting the transmission set the pin DE(/RE) on "driver output enable" (DE=1)	
		//write to the USART the reply to the command
		for(i=0;i<byte_number_to_be_sent;i++) {
			putcUSART(ToSendDataBuffer[i]);
			while(BusyUSART());  //wait for USART to process each character before next character
		}
		// putrsUSART("\r\n");	

	    wait_ticks(1000); //wait about 1ms before changing the RS485 driver back to listen mode, to assure the entire command is transmitted on the RS485 bus		
		PIN_DE_RE_MAX485=0; //RS485 protocol: after finishing the transmission set the pin DE(/RE) back to listen (DE=0)
	}

}




// ******************************************************************************************************
// ************** USB Callback Functions ****************************************************************
// ******************************************************************************************************
// The USB firmware stack will call the callback functions USBCBxxx() in response to certain USB related
// events.  For example, if the host PC is powering down, it will stop sending out Start of Frame (SOF)
// packets to your device.  In response to this, all USB devices are supposed to decrease their power
// consumption from the USB Vbus to <2.5mA* each.  The USB module detects this condition (which according
// to the USB specifications is 3+ms of no bus activity/SOF packets) and then calls the USBCBSuspend()
// function.  You should modify these callback functions to take appropriate actions for each of these
// conditions.  For example, in the USBCBSuspend(), you may wish to add code that will decrease power
// consumption from Vbus to <2.5mA (such as by clock switching, turning off LEDs, putting the
// microcontroller to sleep, etc.).  Then, in the USBCBWakeFromSuspend() function, you may then wish to
// add code that undoes the power saving things done in the USBCBSuspend() function.

// The USBCBSendResume() function is special, in that the USB stack will not automatically call this
// function.  This function is meant to be called from the application firmware instead.  See the
// additional comments near the function.

// Note *: The "usb_20.pdf" specs indicate 500uA or 2.5mA, depending upon device classification. However,
// the USB-IF has officially issued an ECN (engineering change notice) changing this to 2.5mA for all 
// devices.  Make sure to re-download the latest specifications to get all of the newest ECNs.

/******************************************************************************
 * Function:        void USBCBSuspend(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        Call back that is invoked when a USB suspend is detected
 *
 * Note:            None
 *****************************************************************************/
void USBCBSuspend(void)
{
	//Example power saving code.  Insert appropriate code here for the desired
	//application behavior.  If the microcontroller will be put to sleep, a
	//process similar to that shown below may be used:
	
	//ConfigureIOPinsForLowPower();
	//SaveStateOfAllInterruptEnableBits();
	//DisableAllInterruptEnableBits();
	//EnableOnlyTheInterruptsWhichWillBeUsedToWakeTheMicro();	//should enable at least USBActivityIF as a wake source
	//Sleep();
	//RestoreStateOfAllPreviouslySavedInterruptEnableBits();	//Preferably, this should be done in the USBCBWakeFromSuspend() function instead.
	//RestoreIOPinsToNormal();									//Preferably, this should be done in the USBCBWakeFromSuspend() function instead.

	//IMPORTANT NOTE: Do not clear the USBActivityIF (ACTVIF) bit here.  This bit is 
	//cleared inside the usb_device.c file.  Clearing USBActivityIF here will cause 
	//things to not work as intended.	
	

    #if defined(__C30__) || defined __XC16__
        //This function requires that the _IPL level be something other than 0.
        //  We can set it here to something other than 
        #ifndef DSPIC33E_USB_STARTER_KIT
        _IPL = 1;
        USBSleepOnSuspend();
        #endif
    #endif
}



/******************************************************************************
 * Function:        void USBCBWakeFromSuspend(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        The host may put USB peripheral devices in low power
 *					suspend mode (by "sending" 3+ms of idle).  Once in suspend
 *					mode, the host may wake the device back up by sending non-
 *					idle state signalling.
 *					
 *					This call back is invoked when a wakeup from USB suspend 
 *					is detected.
 *
 * Note:            None
 *****************************************************************************/
void USBCBWakeFromSuspend(void)
{
	// If clock switching or other power savings measures were taken when
	// executing the USBCBSuspend() function, now would be a good time to
	// switch back to normal full power run mode conditions.  The host allows
	// 10+ milliseconds of wakeup time, after which the device must be 
	// fully back to normal, and capable of receiving and processing USB
	// packets.  In order to do this, the USB module must receive proper
	// clocking (IE: 48MHz clock must be available to SIE for full speed USB
	// operation).  
	// Make sure the selected oscillator settings are consistent with USB 
    // operation before returning from this function.
}

/********************************************************************
 * Function:        void USBCB_SOF_Handler(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        The USB host sends out a SOF packet to full-speed
 *                  devices every 1 ms. This interrupt may be useful
 *                  for isochronous pipes. End designers should
 *                  implement callback routine as necessary.
 *
 * Note:            None
 *******************************************************************/
void USBCB_SOF_Handler(void)
{
    // No need to clear UIRbits.SOFIF to 0 here.
    // Callback caller is already doing that.
}

/*******************************************************************
 * Function:        void USBCBErrorHandler(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        The purpose of this callback is mainly for
 *                  debugging during development. Check UEIR to see
 *                  which error causes the interrupt.
 *
 * Note:            None
 *******************************************************************/
void USBCBErrorHandler(void)
{
    // No need to clear UEIR to 0 here.
    // Callback caller is already doing that.

	// Typically, user firmware does not need to do anything special
	// if a USB error occurs.  For example, if the host sends an OUT
	// packet to your device, but the packet gets corrupted (ex:
	// because of a bad connection, or the user unplugs the
	// USB cable during the transmission) this will typically set
	// one or more USB error interrupt flags.  Nothing specific
	// needs to be done however, since the SIE will automatically
	// send a "NAK" packet to the host.  In response to this, the
	// host will normally retry to send the packet again, and no
	// data loss occurs.  The system will typically recover
	// automatically, without the need for application firmware
	// intervention.
	
	// Nevertheless, this callback function is provided, such as
	// for debugging purposes.
}


/*******************************************************************
 * Function:        void USBCBCheckOtherReq(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        When SETUP packets arrive from the host, some
 * 					firmware must process the request and respond
 *					appropriately to fulfill the request.  Some of
 *					the SETUP packets will be for standard
 *					USB "chapter 9" (as in, fulfilling chapter 9 of
 *					the official USB specifications) requests, while
 *					others may be specific to the USB device class
 *					that is being implemented.  For example, a HID
 *					class device needs to be able to respond to
 *					"GET REPORT" type of requests.  This
 *					is not a standard USB chapter 9 request, and 
 *					therefore not handled by usb_device.c.  Instead
 *					this request should be handled by class specific 
 *					firmware, such as that contained in usb_function_hid.c.
 *
 * Note:            None
 *******************************************************************/
void USBCBCheckOtherReq(void)
{
    USBCheckHIDRequest();
}//end


/*******************************************************************
 * Function:        void USBCBStdSetDscHandler(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        The USBCBStdSetDscHandler() callback function is
 *					called when a SETUP, bRequest: SET_DESCRIPTOR request
 *					arrives.  Typically SET_DESCRIPTOR requests are
 *					not used in most applications, and it is
 *					optional to support this type of request.
 *
 * Note:            None
 *******************************************************************/
void USBCBStdSetDscHandler(void)
{
    // Must claim session ownership if supporting this request
}//end


/*******************************************************************
 * Function:        void USBCBInitEP(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        This function is called when the device becomes
 *                  initialized, which occurs after the host sends a
 * 					SET_CONFIGURATION (wValue not = 0) request.  This 
 *					callback function should initialize the endpoints 
 *					for the device's usage according to the current 
 *					configuration.
 *
 * Note:            None
 *******************************************************************/
void USBCBInitEP(void)
{
    //enable the HID endpoint
    USBEnableEndpoint(HID_EP,USB_IN_ENABLED|USB_OUT_ENABLED|USB_HANDSHAKE_ENABLED|USB_DISALLOW_SETUP);
    //Re-arm the OUT endpoint for the next packet
    USBOutHandle = HIDRxPacket(HID_EP,(BYTE*)&ReceivedDataBuffer,RECEIVE_DATA_BUFFER_SIZE);
}

/********************************************************************
 * Function:        void USBCBSendResume(void)
 *
 * PreCondition:    None
 *
 * Input:           None
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        The USB specifications allow some types of USB
 * 					peripheral devices to wake up a host PC (such
 *					as if it is in a low power suspend to RAM state).
 *					This can be a very useful feature in some
 *					USB applications, such as an Infrared remote
 *					control	receiver.  If a user presses the "power"
 *					button on a remote control, it is nice that the
 *					IR receiver can detect this signalling, and then
 *					send a USB "command" to the PC to wake up.
 *					
 *					The USBCBSendResume() "callback" function is used
 *					to send this special USB signalling which wakes 
 *					up the PC.  This function may be called by
 *					application firmware to wake up the PC.  This
 *					function will only be able to wake up the host if
 *                  all of the below are true:
 *					
 *					1.  The USB driver used on the host PC supports
 *						the remote wakeup capability.
 *					2.  The USB configuration descriptor indicates
 *						the device is remote wakeup capable in the
 *						bmAttributes field.
 *					3.  The USB host PC is currently sleeping,
 *						and has previously sent your device a SET 
 *						FEATURE setup packet which "armed" the
 *						remote wakeup capability.   
 *
 *                  If the host has not armed the device to perform remote wakeup,
 *                  then this function will return without actually performing a
 *                  remote wakeup sequence.  This is the required behavior, 
 *                  as a USB device that has not been armed to perform remote 
 *                  wakeup must not drive remote wakeup signalling onto the bus;
 *                  doing so will cause USB compliance testing failure.
 *                  
 *					This callback should send a RESUME signal that
 *                  has the period of 1-15ms.
 *
 * Note:            This function does nothing and returns quickly, if the USB
 *                  bus and host are not in a suspended condition, or are 
 *                  otherwise not in a remote wakeup ready state.  Therefore, it
 *                  is safe to optionally call this function regularly, ex: 
 *                  anytime application stimulus occurs, as the function will
 *                  have no effect, until the bus really is in a state ready
 *                  to accept remote wakeup. 
 *
 *                  When this function executes, it may perform clock switching,
 *                  depending upon the application specific code in 
 *                  USBCBWakeFromSuspend().  This is needed, since the USB
 *                  bus will no longer be suspended by the time this function
 *                  returns.  Therefore, the USB module will need to be ready
 *                  to receive traffic from the host.
 *
 *                  The modifiable section in this routine may be changed
 *                  to meet the application needs. Current implementation
 *                  temporary blocks other functions from executing for a
 *                  period of ~3-15 ms depending on the core frequency.
 *
 *                  According to USB 2.0 specification section 7.1.7.7,
 *                  "The remote wakeup device must hold the resume signaling
 *                  for at least 1 ms but for no more than 15 ms."
 *                  The idea here is to use a delay counter loop, using a
 *                  common value that would work over a wide range of core
 *                  frequencies.
 *                  That value selected is 1800. See table below:
 *                  ==========================================================
 *                  Core Freq(MHz)      MIP         RESUME Signal Period (ms)
 *                  ==========================================================
 *                      48              12          1.05
 *                       4              1           12.6
 *                  ==========================================================
 *                  * These timing could be incorrect when using code
 *                    optimization or extended instruction mode,
 *                    or when having other interrupts enabled.
 *                    Make sure to verify using the MPLAB SIM's Stopwatch
 *                    and verify the actual signal on an oscilloscope.
 *******************************************************************/
void USBCBSendResume(void)
{
    static WORD delay_count;
    
    //First verify that the host has armed us to perform remote wakeup.
    //It does this by sending a SET_FEATURE request to enable remote wakeup,
    //usually just before the host goes to standby mode (note: it will only
    //send this SET_FEATURE request if the configuration descriptor declares
    //the device as remote wakeup capable, AND, if the feature is enabled
    //on the host (ex: on Windows based hosts, in the device manager 
    //properties page for the USB device, power management tab, the 
    //"Allow this device to bring the computer out of standby." checkbox 
    //should be checked).
    if(USBGetRemoteWakeupStatus() == TRUE) 
    {
        //Verify that the USB bus is in fact suspended, before we send
        //remote wakeup signalling.
        if(USBIsBusSuspended() == TRUE)
        {
            USBMaskInterrupts();
            
            //Clock switch to settings consistent with normal USB operation.
            USBCBWakeFromSuspend();
            USBSuspendControl = 0; 
            USBBusIsSuspended = FALSE;  //So we don't execute this code again, 
                                        //until a new suspend condition is detected.

            //Section 7.1.7.7 of the USB 2.0 specifications indicates a USB
            //device must continuously see 5ms+ of idle on the bus, before it sends
            //remote wakeup signalling.  One way to be certain that this parameter
            //gets met, is to add a 2ms+ blocking delay here (2ms plus at 
            //least 3ms from bus idle to USBIsBusSuspended() == TRUE, yeilds
            //5ms+ total delay since start of idle).
            delay_count = 3600U;        
            do
            {
                delay_count--;
            }while(delay_count);
            
            //Now drive the resume K-state signalling onto the USB bus.
            USBResumeControl = 1;       // Start RESUME signaling
            delay_count = 1800U;        // Set RESUME line for 1-13 ms
            do
            {
                delay_count--;
            }while(delay_count);
            USBResumeControl = 0;       //Finished driving resume signalling

            USBUnmaskInterrupts();
        }
    }
}


/*******************************************************************
 * Function:        BOOL USER_USB_CALLBACK_EVENT_HANDLER(
 *                        USB_EVENT event, void *pdata, WORD size)
 *
 * PreCondition:    None
 *
 * Input:           USB_EVENT event - the type of event
 *                  void *pdata - pointer to the event data
 *                  WORD size - size of the event data
 *
 * Output:          None
 *
 * Side Effects:    None
 *
 * Overview:        This function is called from the USB stack to
 *                  notify a user application that a USB event
 *                  occured.  This callback is in interrupt context
 *                  when the USB_INTERRUPT option is selected.
 *
 * Note:            None
 *******************************************************************/
BOOL USER_USB_CALLBACK_EVENT_HANDLER(int event, void *pdata, WORD size)
{
    switch(event)
    {
        case EVENT_TRANSFER:
            //Add application specific callback task or callback function here if desired.
            break;
        case EVENT_SOF:
            USBCB_SOF_Handler();
            break;
        case EVENT_SUSPEND:
            USBCBSuspend();
            break;
        case EVENT_RESUME:
            USBCBWakeFromSuspend();
            break;
        case EVENT_CONFIGURED: 
            USBCBInitEP();
            break;
        case EVENT_SET_DESCRIPTOR:
            USBCBStdSetDscHandler();
            break;
        case EVENT_EP0_REQUEST:
            USBCBCheckOtherReq();
            break;
        case EVENT_BUS_ERROR:
            USBCBErrorHandler();
            break;
        case EVENT_TRANSFER_TERMINATED:
            //Add application specific callback task or callback function here if desired.
            //The EVENT_TRANSFER_TERMINATED event occurs when the host performs a CLEAR
            //FEATURE (endpoint halt) request on an application endpoint which was 
            //previously armed (UOWN was = 1).  Here would be a good place to:
            //1.  Determine which endpoint the transaction that just got terminated was 
            //      on, by checking the handle value in the *pdata.
            //2.  Re-arm the endpoint if desired (typically would be the case for OUT 
            //      endpoints).
            break;
        default:
            break;
    }      
    return TRUE; 
}

/** EOF main.c *************************************************/
#endif
