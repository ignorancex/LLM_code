# %% libraries
import os
from collections import Counter
import importlib
import re
import pickle
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding

# %% body
def recursive_lowercase(data):
    if isinstance(data, str):
        return data.lower()
    elif isinstance(data, list):
        return [recursive_lowercase(el) for el in data]
    elif isinstance(data, set):
        return {recursive_lowercase(el) for el in data}
    elif isinstance(data, dict):
        return {key: recursive_lowercase(value) for key, value in data.items()}
    else:
        return data

def recursive_convert(data, converter):
    if isinstance(data, str):
        return converter[data]
    elif isinstance(data, list):
        return [recursive_convert(el) for el in data]
    elif isinstance(data, set):
        return {recursive_convert(el) for el in data}
    elif isinstance(data, dict):
        return {key: recursive_convert(value) for key, value in data.items()}
    else:
        return data


def unique_dicts(dict_list):
    unique = []
    for el in dict_list:
        if el not in dict_list:
            unique.append(el)

    return unique


def save_json(data, filename):
    with open(filename, "w") as outfile:
        json.dump(data, outfile)


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def unlist(nested_list):
    unlisted = [subel for el in nested_list for subel in el]
    return unlisted


def pickle_save(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def pickle_load(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created at: {dir_path}")
    else:
        print(f"Directory already exists at: {dir_path}")

def pickle_save_encrypted(data, filename, public_key_path):
    # Serialize dataset (can be large)
    serialized_data = pickle.dumps(data)

    # Generate random AES key and IV
    aes_key = os.urandom(32)  # AES-256
    iv = os.urandom(16)

    # Encrypt the serialized data with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    encryptor = cipher.encryptor()

    # PKCS7 padding
    padding_len = 16 - (len(serialized_data) % 16)
    padded_data = serialized_data + bytes([padding_len] * padding_len)

    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Load RSA public key
    with open(public_key_path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read())

    # Encrypt AES key with RSA
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        rsa_padding.OAEP(
            mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Save to file: [2-byte key length][encrypted AES key][IV][encrypted data]
    with open(filename, "wb") as f:
        f.write(len(encrypted_aes_key).to_bytes(2, "big"))
        f.write(encrypted_aes_key)
        f.write(iv)
        f.write(encrypted_data)

def pickle_load_encrypted(filename, private_key_path, password=None):

    # Load private RSA key
    with open(private_key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=password.encode() if password else None,
        )

    with open(filename, "rb") as f:
        key_len = int.from_bytes(f.read(2), "big")
        encrypted_key = f.read(key_len)
        iv = f.read(16)
        encrypted_data = f.read()

    # Decrypt AES key
    aes_key = private_key.decrypt(
        encrypted_key,
        rsa_padding.OAEP(
            mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Decrypt data
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove PKCS7 padding
    padding_len = padded_data[-1]
    data = padded_data[:-padding_len]

    return pickle.loads(data)
