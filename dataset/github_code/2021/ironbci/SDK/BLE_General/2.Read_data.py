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
            voltage_1 = (data[0] << 16) | (data[1] << 8) | data[2]
            data_test = 0x7FFFFF
            data_check = 0xFFFFFF
            
            convert_voltage = voltage_1 | data_test
            
            if convert_voltage == data_check:
                voltage_1_after_convert = (voltage_1 - 16777214)
            else:
                voltage_1_after_convert = voltage_1
            result = round(1000000 * 4.5 * (voltage_1_after_convert / 16777215), 2)
            print(f"Result: {result}")

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
