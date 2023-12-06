from sympy import im
import os
from tools import *

class MyLogger():
    log_file = ''
    logs = []

    def __init__(self, log_file):
        time_stamp = datetime2str()
        self.log_file = log_file + '.' + time_stamp + '.txt'
        self.logs = []

        # check dir
        make_dir_if_needed(self.log_file)

    def log(self, msg='\n'):
        self.logs.append(msg)
    
    def save(self):
        save_file(self.log_file, self.logs)
        print(f'logs have been saved to {self.log_file}')