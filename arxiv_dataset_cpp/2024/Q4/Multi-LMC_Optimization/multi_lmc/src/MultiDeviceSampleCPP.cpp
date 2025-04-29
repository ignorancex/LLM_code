#include <LeapC.h>
#include <string>
#include <iostream>

#if defined(_MSC_VER)
  #include <Windows.h>
  #include <process.h>
  #define LockType CRITICAL_SECTION
  #define ThreadType HANDLE
  #define ThreadReturnType void
  #define LockMutex EnterCriticalSection
  #define UnlockMutex LeaveCriticalSection
#else
  #include <unistd.h>
  #include <pthread.h>
  #define LockType pthread_mutex_t
  #define ThreadType pthread_t
  #define ThreadReturnType void*
  #define LockMutex pthread_mutex_lock
  #define UnlockMutex pthread_mutex_unlock
#endif

// To stop the service loop.
int stop;

static ThreadType pollingThread;
static LockType dataLock;

#define LEAPC_CHECK(func)                  \
  do                                       \
  {                                        \
    eLeapRS result = func;                 \
    if (result != eLeapRS_Success)         \
    {                                      \
      std::cerr << "Fatal error in calling function: " << #func << ": " << std::hex << result << std::endl; \
      abort();                             \
    }                                      \
  } while (0)

static const char* devicePIDToString(eLeapDevicePID p)
{
  if (p == eLeapDevicePID_3Di)
  {
    return "3Di";
  }
  if (p == eLeapDevicePID_Peripheral)
  {
    return "Leap Motion Controller";
  }
  if (p == eLeapDevicePID_SIR170)
  {
    return "Stereo SIR170";
  }
  return "Unknown Tracking Device";
}

struct DeviceState
{
  LEAP_DEVICE device;
  uint32_t id;
};

static ThreadReturnType pollingServiceLoop(void* p)
{
  LEAP_CONNECTION* connection = (LEAP_CONNECTION*)p;

  // LEAP_DEVICE's can be passed to -Ex forms of the LeapC API
  DeviceState* devices = nullptr;
  uint32_t deviceCount = 0;

  while (true)
  {
    LockMutex(&dataLock);
    if (stop)
    {
      UnlockMutex(&dataLock);
      break;
    }
    UnlockMutex(&dataLock);

    const uint32_t timeoutMilliseconds = 10;
    LEAP_CONNECTION_MESSAGE msg;
    eLeapRS result = LeapPollConnection(*connection, timeoutMilliseconds, &msg);
    if (result == eLeapRS_Timeout)
    {
      continue;
    }

    if (msg.type == eLeapEventType_Device)
    {
      DeviceState* newDevices = reinterpret_cast<DeviceState*>(realloc(devices, sizeof(DeviceState) * (deviceCount + 1)));
      if (!newDevices)
      {
        std::cerr << "Failed to allocate memory to accommodate " << deviceCount + 1 << " devices" << std::endl;
        abort();
      }
      else
      {
        devices = newDevices;
        LEAPC_CHECK(LeapOpenDevice(msg.device_event->device, &devices[deviceCount].device));

        devices[deviceCount].id = msg.device_event->device.id;

        LEAP_DEVICE_INFO deviceInfo;
        memset(&deviceInfo, 0, sizeof(deviceInfo));

        // Use stack memory to allocate for the serial number field.
        char serial[1000];
        memset(serial, 0, sizeof(serial));
        deviceInfo.serial = serial;
        deviceInfo.serial_length = sizeof(serial) - 1;
        deviceInfo.size = sizeof(deviceInfo);

        LEAPC_CHECK(LeapGetDeviceInfo(devices[deviceCount].device, &deviceInfo));

        std::cout << "Found device with ID: " << msg.device_event->device.id << ", type: "
                  << devicePIDToString(deviceInfo.pid) << ", serial number: " << deviceInfo.serial << std::endl;

        // Unconditionally subscribe to the device:
        LEAPC_CHECK(LeapSubscribeEvents(*connection, devices[deviceCount].device));
        ++deviceCount;
      }
    }

    if (msg.type == eLeapEventType_DeviceLost)
    {
      LEAP_DEVICE* deviceLost = nullptr;
      for (int i = 0; i < deviceCount; ++i)
      {
        if (devices[i].id == msg.device_event->device.id)
        {
          deviceLost = &devices[i].device;
        }
      }

      if (deviceLost)
      {
        std::cout << "Unsubscribing from device: " << msg.device_event->device.id << std::endl;
        LEAPC_CHECK(LeapUnsubscribeEvents(*connection, *deviceLost));
        LeapCloseDevice(*deviceLost);
      }
    }

    if (msg.type == eLeapEventType_Tracking)
    {
      if (msg.tracking_event->info.frame_id % 100 == 0)
      {
        std::cout << "Got tracking event for device ID: " << msg.device_id
                  << ", Tracking Frame ID: " << msg.tracking_event->info.frame_id
                  << ", Hand Count: " << msg.tracking_event->nHands << std::endl;
      }
    }
  }

  free(devices);
}

int main(void)
{
  LEAP_CONNECTION connection;
  LEAP_CONNECTION_CONFIG config;

  // Set connection to multi-device aware
  memset(&config, 0, sizeof(config));
  config.size = sizeof(config);
  config.flags = eLeapConnectionConfig_MultiDeviceAware;

  // Open LeapC connection:
  LEAPC_CHECK(LeapCreateConnection(&config, &connection));
  LEAPC_CHECK(LeapOpenConnection(connection));

  std::cout << "Press Enter or Control-C to exit, tracking messages will follow:" << std::endl;

#if defined(_MSC_VER)
  InitializeCriticalSection(&dataLock);
  pollingThread = (ThreadType)_beginthread(pollingServiceLoop, 0, &connection);
#else
  pthread_mutex_init(&dataLock, nullptr);
  pthread_create(&pollingThread, nullptr, pollingServiceLoop, &connection);
#endif

  (void)getchar();

  LockMutex(&dataLock);
  stop = 1;
  UnlockMutex(&dataLock);

#if defined(_MSC_VER)
  WaitForSingleObject(pollingThread, INFINITE);
  CloseHandle(pollingThread);
#else
  pthread_join(pollingThread, nullptr);
  pthread_mutex_destroy(&dataLock);
#endif

  LeapCloseConnection(connection);
  LeapDestroyConnection(connection);

  return 0;
}
