/***** Includes *****/
/* Standard C Libraries */
#include <stdlib.h>

/* TI Drivers */
#include <ti/drivers/rf/RF.h>
#include <ti/drivers/PIN.h>

/* Driverlib Header files */
#include DeviceFamily_constructPath(driverlib/rf_ble_mailbox.h)

/* Board Header files */
#include "Board.h"

/* Application Header files */
#include "smartrf_settings/smartrf_settings_ble.h"

/***** Defines *****/
/*
 * BLE PACKET
 * ________________________________________________________________________________________________________________________
 *|                                 |               |                                 |          |           |             |
 *| PDU HEADER FIELD             2B | ADV ADDR      | PACKET DATA                 27B | CRC   3B | RSSI   1B | STATUS   2B |
 *| ADV TYPE   1B | PACKET LEN   1B |  - BLE4    6B | SERIAL NUMBER   2B | DATA   25B |          |           |             |
 *|                                 |  - BLE5   10B |                                 |          |           |             |
 *|_________________________________|_______________|_________________________________|__________|___________|_____________|
 *
 * PROPRIETARY PACKET
 * _____________________________________________________________
 *|                |                                 |          |
 *| LEN FIELD   1B | PACKET DATA                 27B | CRC   2B |
 *|                | SERIAL NUMBER   2B | DATA   25B |          |
 *|________________|_________________________________|__________|
 *
 */

/* Packet TX Configuration */
#define PACKET_DATA_LENGTH               	  27 /* Packet for BLE5 cannot exceed 37 with header (10) */
#define PACKET_INTERVAL      RF_convertMsToRatTicks(500) /* Set packet interval to 500ms */

/***** Variable declarations *****/
static RF_Object rfBleObject;
static RF_Handle rfBleHandle;

/* Pin driver handle */
static PIN_Handle pinHandle;
static PIN_State pinState;

static uint8_t packet[PACKET_DATA_LENGTH];
static uint16_t seqNumber;

/*
 * Application LED pin configuration table:
 *   - All LEDs board LEDs are off.
 */
PIN_Config pinTable[] =
{
    Board_PIN_LED1 | PIN_GPIO_OUTPUT_EN | PIN_GPIO_LOW | PIN_PUSHPULL | PIN_DRVSTR_MAX,
    PIN_TERMINATE
};

/***** Function definitions *****/
static void txDoneCallback(RF_Handle h, RF_CmdHandle ch, RF_EventMask e)
{
    if (e & RF_EventLastCmdDone)
    {
        /* Successful TX */
        /* Toggle LED1 */
        PIN_setOutputValue(pinHandle, Board_PIN_LED1,!PIN_getOutputValue(Board_PIN_LED1));
    }
    else
    {
        /* Error Condition: switch off LED1 */
        PIN_setOutputValue(pinHandle, Board_PIN_LED1, 0);
    }
}

void *mainThread(void *arg0)
{
    uint32_t curtime;
    uint32_t cmdStatus;
    RF_EventMask terminationReason;

    pinHandle = PIN_open(&pinState, pinTable);
    if (pinHandle == NULL)
    {
        while(1);
    }

    /* Initialize multimode scheduling params
     * - Params shared between rf drivers since commands are synchronous
     * - Ignore end time
     * - Priority should not affect transmission
     */
    //RF_ScheduleCmdParams schParams;
    //schParams.endTime = 0;
    //schParams.priority = RF_PriorityNormal;
    //schParams.allowDelay = RF_AllowDelayAny;

    /* Initialize BLE RF Driver */
    RF_Params rfBleParams;
    RF_Params_init(&rfBleParams);

    /* Set mode for multiple clients */
    //RF_modeBle.rfMode = RF_modeBle;

    /* Request access to the ble radio */
    RF_ble_cmdBleAdvNc.pParams->advLen = PACKET_DATA_LENGTH;
    RF_ble_cmdBleAdvNc.pParams->pAdvData = packet;
    RF_ble_cmdBleAdvNc.startTrigger.triggerType = TRIG_ABSTIME;
    RF_ble_cmdBleAdvNc.startTrigger.pastTrig = 1;
    RF_ble_cmdBleAdvNc.startTime = 0;

    /* Request access to the bleradio and
     * - RF_ble_cmdFs does not need to run unless no channel is specified (0xFF)
     * - Channel 17 (0x8C) is used by default
     */
    rfBleHandle = RF_open(&rfBleObject, &RF_modeBle,
                          (RF_RadioSetup*)&RF_ble_cmdRadioSetup, &rfBleParams);
    //(void)RF_scheduleCmd(rfPropHandle, (RF_Op*) &RF_cmdFs, &schParams, NULL, 0);

    /* Get current time */
    curtime = RF_getCurrentTime();

    while (1)
    {
        /* Create packet with incrementing sequence number and random payload */
        packet[0] = (uint8_t) (seqNumber >> 8);
        packet[1] = (uint8_t) (seqNumber++);
        uint8_t i;
        for (i = 2; i < PACKET_DATA_LENGTH; i++)
        {
            packet[i] = rand();
        }

        /* Transmit BLE packet */
        /* Set absolute TX time to utilize automatic power management */
        curtime += PACKET_INTERVAL;

        RF_ble_cmdBleAdvNc.startTime = curtime;
        terminationReason = RF_runCmd(rfBleHandle, (RF_Op*)&RF_ble_cmdBleAdvNc,
                                              RF_PriorityNormal, txDoneCallback, 0);

        if(terminationReason & RF_EventCmdPreempted)
        {
            // It is possible for a scheduled command to be preempted by another
            // higher priority command. In this case the RF driver will either
            // cancel/abort/stop the preempted command and return the appropriate
            // event flag. Additionally, the command preempted event flag is also set.

        }

        // Mask off the RF_EventCmdPreempted bit to allow further processing
        // in the switch-case block
        switch (terminationReason & ~(RF_EventCmdPreempted))
        {
            case RF_EventLastCmdDone:
                // A stand-alone radio operation command or the last radio
                // operation command in a chain finished.
                break;
            case RF_EventCmdCancelled:
                // Command cancelled before it was started; it can be caused
                // by RF_cancelCmd() or RF_flushCmd().
                break;
            case RF_EventCmdAborted:
                // Abrupt command termination caused by RF_cancelCmd() or
                // RF_flushCmd().
                break;
            case RF_EventCmdStopped:
                // Graceful command termination caused by RF_cancelCmd() or
                // RF_flushCmd().
                break;
            default:
                // Uncaught error event
                while (1);
        }

        cmdStatus = RF_ble_cmdBleAdvNc.status;

        switch (cmdStatus)
        {
            case BLE_DONE_OK:
                // Packet transmitted successfully
                break;
            case BLE_DONE_STOPPED:
                // received CMD_STOP while transmitting packet and finished
                // transmitting packetoi
                break;
            case BLE_DONE_ABORT:
                // Received CMD_ABORT while transmitting packet
                break;
            case BLE_ERROR_PAR:
                // Observed illegal parameter
                break;
            case BLE_ERROR_NO_SETUP:
                // Command sent without setting up the radio in a supported
                // mode using CMD_BLE_RADIO_SETUP or CMD_RADIO_SETUP
                break;
            case BLE_ERROR_NO_FS:
                // Command sent without the synthesizer being programmed
                break;
            case BLE_ERROR_TXUNF:
                // TX underflow observed during operation
                break;
            default:
                // Uncaught error event - these could come from the
                // pool of states defined in rf_mailbox.h
                while (1);
        }
    }
}
