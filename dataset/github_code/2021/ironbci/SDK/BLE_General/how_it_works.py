import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "EAREEG"
FIRST_NAME_ID = '0000fe42-8e22-4541-9d4c-21edae82ed19'

async def find_eareeg_device():
    devices = await BleakScanner.discover()
    for device in devices:
        if device.name == DEVICE_NAME:
            return device
    return None

async def main():
    print(f"Scanning for {DEVICE_NAME}...")
    eareeg_device = await find_eareeg_device()
    
    if eareeg_device is None:
        print(f"{DEVICE_NAME} not found.")
        return

    print(f"Found {DEVICE_NAME} at address: {eareeg_device.address}")
    
    async with BleakClient(eareeg_device.address) as client:
        print(f"Connected to {DEVICE_NAME}")

        def callback(sender, data):
            print ("data lenght", len(data))
            print(data)
            voltage_1 = (data[0] << 16) | (data[1] << 8) | data[2]
            voltage_2 = (data[3] << 16) | (data[4] << 8) | data[5]
            voltage_3 = (data[6] << 16) | (data[7] << 8) | data[8]
            voltage_4 = (data[9] << 16) | (data[10] << 8) | data[11] 
            voltage_5 = (data[12] << 16) | (data[13] << 8) | data[14]
            voltage_6 = (data[15] << 16) | (data[16] << 8) | data[17]
            voltage_7 = (data[18] << 16) | (data[19] << 8) | data[20]
            voltage_8 = (data[21] << 16) | (data[22] << 8) | data[23] 

            voltage_1_1 = (data[0] << 16) | (data[1] << 8) | data[2]
            voltage_2_1 = (data[3] << 16) | (data[4] << 8) | data[5]
            voltage_3_1 = (data[6] << 16) | (data[7] << 8) | data[8]
            voltage_4_1 = (data[9] << 16) | (data[10] << 8) | data[11] 
            voltage_5_1 = (data[12] << 16) | (data[13] << 8) | data[14]
            voltage_6_1 = (data[15] << 16) | (data[16] << 8) | data[17]
            voltage_7_1 = (data[18] << 16) | (data[19] << 8) | data[20]
            voltage_8_1 = (data[21] << 16) | (data[22] << 8) | data[23] 

            voltage_1_2 = (data[24] << 16) | (data[25] << 8) | data[26]
            voltage_2_2 = (data[27] << 16) | (data[28] << 8) | data[29]
            voltage_3_2 = (data[30] << 16) | (data[31] << 8) | data[32]
            voltage_4_2 = (data[33] << 16) | (data[34] << 8) | data[35] 
            voltage_5_2 = (data[36] << 16) | (data[37] << 8) | data[38]
            voltage_6_2 = (data[39] << 16) | (data[40] << 8) | data[41]
            voltage_7_2 = (data[42] << 16) | (data[43] << 8) | data[44]
            voltage_8_2 = (data[45] << 16) | (data[46] << 8) | data[47] 

            voltage_1_3 = (data[48] << 16) | (data[49] << 8) | data[50]
            voltage_2_3 = (data[51] << 16) | (data[52] << 8) | data[53]
            voltage_3_3 = (data[54] << 16) | (data[55] << 8) | data[56]
            voltage_4_3 = (data[57] << 16) | (data[58] << 8) | data[59] 
            voltage_5_3 = (data[60] << 16) | (data[61] << 8) | data[62]
            voltage_6_3 = (data[63] << 16) | (data[64] << 8) | data[65]
            voltage_7_3 = (data[66] << 16) | (data[67] << 8) | data[68]
            voltage_8_3 = (data[69] << 16) | (data[70] << 8) | data[71]
            
            voltage_1_4 = (data[72] << 16) | (data[73] << 8) | data[74]
            voltage_2_4 = (data[75] << 16) | (data[76] << 8) | data[77]
            voltage_3_4 = (data[78] << 16) | (data[79] << 8) | data[80]
            voltage_4_4 = (data[81] << 16) | (data[82] << 8) | data[83] 
            voltage_5_4 = (data[84] << 16) | (data[85] << 8) | data[86]
            voltage_6_4 = (data[87] << 16) | (data[88] << 8) | data[89]
            voltage_7_4 = (data[90] << 16) | (data[91] << 8) | data[92]
            voltage_8_4 = (data[93] << 16) | (data[94] << 8) | data[95]
            

            voltage_1_5 = (data[96] << 16) | (data[97] << 8) | data[98]
            voltage_2_5 = (data[99] << 16) | (data[100] << 8) | data[101]
            voltage_3_5 = (data[102] << 16) | (data[103] << 8) | data[104]
            voltage_4_5 = (data[105] << 16) | (data[106] << 8) | data[107] 
            voltage_5_5 = (data[108] << 16) | (data[109] << 8) | data[110]
            voltage_6_5 = (data[111] << 16) | (data[112] << 8) | data[113]
            voltage_7_5 = (data[114] << 16) | (data[115] << 8) | data[116]
            voltage_8_5 = (data[117] << 16) | (data[118] << 8) | data[119]


            voltage_1_final = (voltage_1, voltage_1_1, voltage_1_2, voltage_1_3,voltage_1_4, voltage_1_5)
            voltage_2_final = (voltage_2, voltage_2_1, voltage_2_2, voltage_2_3,voltage_2_4, voltage_2_5)
            voltage_3_final = (voltage_3, voltage_3_1, voltage_3_2, voltage_3_3,voltage_3_4, voltage_3_5)
            voltage_4_final = (voltage_4, voltage_4_1, voltage_4_2, voltage_4_3,voltage_4_4, voltage_4_5)

            voltage_5_final = (voltage_5, voltage_5_1, voltage_5_2, voltage_5_3,voltage_5_4, voltage_5_5)
            voltage_6_final = (voltage_6, voltage_6_1, voltage_6_2, voltage_6_3,voltage_6_4, voltage_6_5)
            voltage_7_final = (voltage_7, voltage_7_1, voltage_7_2, voltage_7_3,voltage_7_4, voltage_7_5)
            voltage_8_final = (voltage_8, voltage_8_1, voltage_8_2, voltage_8_3,voltage_8_4, voltage_8_5)
            #voltage_2_final = (voltage_2, voltage_2_1)

            
            data_test = 0x7FFFFF
            data_check = 0xFFFFFF


            data_to_graph = []

            for data_to_voltage in voltage_1_final:
                
                convert_voltage = data_to_voltage | data_test  
                if convert_voltage == data_check:
                    voltage_1_after_convert = (data_to_voltage - 16777214)
                else:
                    voltage_1_after_convert = data_to_voltage
                result = round(1000000 * 4.5 * (voltage_1_after_convert / 16777215), 2)
            
                data_to_graph.append(result)
                
            print("result", data_to_graph)

        await client.start_notify(FIRST_NAME_ID, callback)
        print("Notification started. Press Ctrl+C to stop.")
        
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await client.stop_notify(FIRST_NAME_ID)
            print("Notification stopped.")

asyncio.run(main())
