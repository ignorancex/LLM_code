######################################################################################
# Output logging operations.
# Used these resources:
# https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
# https://docs.python.org/3/howto/logging.html, https://docs.python.org/3/howto/logging-cookbook.html
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
# https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
# https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile
#####################################################################################

import os
import sys
sys.path.append(os.getcwd())

import logging


def get_module_list():
    """ Specify the list of modules to exclude while logging. """
    return ['matplotlib', 'numba']


def set_logger(filename, console_level=logging.INFO):  # console_level. Use either INFO or DEBUG!
    """ This function configures basics of the logging. """
    # Set up logging to file
    logging.basicConfig(filename=filename, filemode='w', level=logging.DEBUG,
                        format='%(asctime)-s %(name)-5s %(levelname)-5s %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    # Don't print basic messages from imported modules. # TODO This can be possibly done more efficiently.
    # See https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
    for module_name in get_module_list():
        logging.getLogger(module_name).setLevel(logging.WARNING)

    # Define a Handler which writes INFO messages or higher to the sys.stdout
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(console_level)
    # Set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-10s: %(levelname)-5s %(message)s')
    # Tell the handler to use this format
    console.setFormatter(formatter)
    # Add the handler to the root logger
    logging.getLogger().addHandler(console)

    # Now, we can log to the root logger, or any other logger. First the root...
    logging.info("Logger is created. %s" % filename)


def close_logger():
    """ This function releases the handlers and closes the logger. """
    logging.info("Logger is closed.")
    handlers = logging.getLogger().handlers[:]
    for handler in handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)


def example_usage():
    """ This function shows an example usage. """
    set_logger(filename='example.log')

    # Now, we can log to the root logger, or any other logger. First the root...
    logging.info("This is an info message!")
    logging.warning("This is a warning message!")
    logging.debug("This is a debug message!")
    logging.error("This is an error message!")
    logging.critical("This is a critical message!")

    # Now, define a couple of other loggers which might represent areas in your application:
    logger1 = logging.getLogger('func1')
    logger2 = logging.getLogger('func2')

    logger1.debug('Debug message inside logger func1().')
    logger1.info('Info message inside logger func1()')
    logger2.warning('Warning message inside logger func2()')
    logger2.error('Error message inside logger func2()')

    close_logger()
