from utils import *
from engine import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":

    # Parsing Options
    options = main_parser(sys.argv[1:])
    if len(sys.argv) == 1:
        custom_print("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()

    # Loading Configuration
    config = config_reader(options.model)

    if options.modelSum:
        Engine(config).model_summary()
    if options.train:
        Engine(config).model_training(resume_training=options.resume, checkpoint=options.checkpoint)
    if options.test and options.checkpoint != None:
        Engine(config).model_test(options.checkpoint)







