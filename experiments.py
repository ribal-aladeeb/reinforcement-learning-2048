
"""
1) Create experiement folder 'exp_#_somehash'
2) a pickle folder, and a text folder
3)
5)
"""

import os
import numpy as np

EXPERIMENTS_DIRECTORY = 'experiments/'


def ensure_exists(dir: str):
    if not os.path.isdir(dir):
        os.mkdir(dir)


class Experiment:

    def __init__(self, resumed=False, folder_name=None):
        ensure_exists(EXPERIMENTS_DIRECTORY)

        if resumed:
            # TODO implement the loading of an existing experiment

            # if resumed:
            #     # make sure folder name arg is the same as the one being resumed
            #     assert folder_name != None, 'You did not provide an experiment name to resume from'
            #     assert folder_name in exp_folders, f'Experiment you wish to resume {folder_name} does not exist'
            #     # leave this usecase for later
            pass
            return

        self.folder = os.path.join(EXPERIMENTS_DIRECTORY, self.create_exp_folder(folder_name))
        ensure_exists(self.folder)
        os.mkdir(os.path.join(self.folder, 'text/'))
        os.mkdir(os.path.join(self.folder, 'binary/'))

    def create_exp_folder(self, folder_name=None):

        exp_folders: list = os.listdir(EXPERIMENTS_DIRECTORY)

        if folder_name != None:
            try:
                os.mkdir(os.path.join(EXPERIMENTS_DIRECTORY, folder_name))
                return os.path.join(EXPERIMENTS_DIRECTORY, folder_name)
            except FileExistsError:
                print(f'File {folder_name} already exists')

        latest = 0 if len(exp_folders) == 0 else max([int(filename[4:filename.find('_', 4)]) for filename in exp_folders])

        return f'exp_{latest+1}_{hash(np.random.rand())}'

