TENSORFLOW_IMPORTS = """
import os
import tensorflow
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '"-1"'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
"""


def titanfuzz_to_compile(py_code: str) -> str:
    code = ""
    flag = False
    for line in py_code.split("\n"):
        if line.startswith("output"):
            code += f"@tf.function(jit_compile=True)\ndef tf_func():\n    {line}\n"
            flag = True
        else:
            code += f"{line}\n"
    if flag:
        code += "tf_func()\n"
    return code


def titanfuzz_to_cuda(py_code: str) -> str:
    """
    There is no need to implement!
    """
    raise NotImplementedError


if __name__ == "__main__":
    py_code = """
input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
conv1d_transpose = tf.keras.layers.Conv1DTranspose(filters=2, kernel_size=2, strides=2, padding='valid', activation='relu')
output = conv1d_transpose(input_data)
"""
    py_code = TENSORFLOW_IMPORTS + py_code
    code = titanfuzz_to_compile(py_code)
    print(code)
    exec(code)
