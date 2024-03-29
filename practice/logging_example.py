import logging, os, pathlib
from datetime import datetime as dt

# script_start_time = dt.today().strftime('%Y-%m-%d_%H-%M-%S')

# logging.basicConfig(filename='example_log_%s.log' % script_start_time, encoding='utf-8', level=logging.DEBUG)
# logging.info('Started')
# input('press any key to continue')
# logging.debug('Some extra info that will only be logged if the logging level is set to the highest degree (DEBUG)')
# logging.info('Finished')


def create_log(folder):  # Create log file for storing error details
    init_time = dt.today().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(folder):
            os.makedirs(folder)
    log_file_path = pathlib.PurePath(folder, 'flat_field_correction_%s.log' % init_time)
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s    %(message)s', datefmt='%Y-%m-%d_%H-%M-%S')
    print('\nLogging to: %s' % str(log_file_path))
    
create_log('logs')
logging.info('test')