import platform

# Set the appropriate fft according to system
operatingSystem = platform.uname().system
if operatingSystem == "Darwin":
    import numpy.fft as _fft
elif operatingSystem == "Linux":
    import numpy.fft as _fft

# assign the interfaces for import from other modules
fft, ifft = _fft.fft, _fft.ifft
