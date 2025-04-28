import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from open_biomed.core.pipeline import TrainValPipeline

if __name__ == "__main__":
    pipeline = TrainValPipeline()
    pipeline.run()