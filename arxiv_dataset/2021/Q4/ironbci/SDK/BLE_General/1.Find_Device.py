import asyncio
from bleak import BleakScanner
import sys
from scipy import signal
#from matplotlib import pyplot as plt

address = None
FIRST_NAME_ID = '0000fe42-8e22-4541-9d4c-21edae82ed19'

async def find_eareeg():
    devices = await BleakScanner.discover()
    for device in devices:
        if device.name == "EAREEG":
            print(f"Found EAREEG device:")
            print(f"Name: {device.name}")
            print(f"Address: {device.address}")
            address = device.address
            
            return address
    print("EAREEG device not found")
    return None

async def main():
    ironbci_address = await find_eareeg()
    
    if ironbci_address:
        print ("ironbci_address", ironbci_address)    
     
asyncio.run(main())
