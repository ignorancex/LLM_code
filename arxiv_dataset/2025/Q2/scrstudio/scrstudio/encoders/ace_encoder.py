import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent / "../../../third_party"))
from glace.ace_encoder import ACEEncoderConfig, ACEEncoder  # noqa E402
