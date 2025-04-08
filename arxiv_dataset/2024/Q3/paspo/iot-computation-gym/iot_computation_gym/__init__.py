from gym.envs.registration import register

register(
    id="iot-computation-env-v0",
    entry_point="iot_computation_gym.envs:IOTComputationEnv",
)
