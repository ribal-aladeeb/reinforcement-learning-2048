import os
from shutil import copy
import numpy as np
import json
import pickle
import time
import sys


EXPERIMENTS_DIRECTORY = 'experiments/'


def ensure_exists(dir: str):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def get_project_root_dir() -> str:
    # because the root of the project contains the .git/ repo
    while not os.path.isdir('.git/'):
        if os.getcwd() == '/':
            print('\nYou are trying to get the root folder of the big data project')
            print('but you are running this script outside of the project.')
            print('Navigate to the project directory and try again.')
            exit(1)
        else:
            os.chdir('..')
    if sys.platform == "linux":
        return f'{os.getcwd()}/'
    elif sys.platform == "win32":
        return f'{os.getcwd()}/'
    else:
        raise OSError("Not implemented for Mac OS")
        return f'file://{os.getcwd()}/' # change for mac


class Experiment:

    def __init__(self, resumed=False, folder_name=None, python_file_name=None):
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
        self.folder_name = folder_name
        self.folder = os.path.join(get_project_root_dir(), self.create_exp_folder(folder_name))

        ensure_exists(self.folder)
        os.mkdir(os.path.join(self.folder, 'text/'))
        os.mkdir(os.path.join(self.folder, 'binary/'))

        self.hyperparameters = {}
        self.episodes = []
        self.runtime = time.time()

        if python_file_name:
            self._save_python_file(python_file_name)

    def _save_python_file(self, python_file_name):
        ugly_path = os.path.join(os.path.dirname(__file__), python_file_name)
        ugly_src = os.path.join(self.folder, f'{python_file_name}.txt')
        from shutil import copyfile
        copyfile(ugly_path, ugly_src)

    def create_exp_folder(self, folder_name=None):

        exp_folders: list = os.listdir(EXPERIMENTS_DIRECTORY)

        if folder_name != None:
            try:
                os.mkdir(os.path.join(EXPERIMENTS_DIRECTORY, folder_name))
                return os.path.join(EXPERIMENTS_DIRECTORY, folder_name)
            except FileExistsError:
                print(f'File {folder_name} already exists. Different folder name will be used.')

        latest = 0 if len([f for f in exp_folders if 'exp_' in f]) == 0 else max([int(filename[4:filename.find('_', 4)]) for filename in exp_folders if 'exp_' in filename])

        return os.path.join(EXPERIMENTS_DIRECTORY, f'exp_{latest+1}_{hash(np.random.rand())}')

    def add_hyperparameter(self, mapping):
        '''
        Mapping should be a dict with single key value pair that has the name of
        the hyperparameter and its value.
        '''
        assert type(mapping) == dict
        self.hyperparameters.update(mapping)

    def add_episode(self, board, epsilon, number, reward):
        episode = {
            'max_tile': np.max(board.state.flatten()),
            'merge_score': board.merge_score(),
            'number': number,
            'reward': reward,
            'epsilon': epsilon,
            'number_moves': len(board._action_history)
        }
        self.episodes.append(episode)

    def save(self):
        elapsed = time.time()-self.runtime

        self.runtime = time.strftime('%H:%M:%S', time.gmtime(elapsed))

        with open(os.path.join(self.folder, 'text/hyperparams.json'), mode='w') as f:
            json.dump(self.hyperparameters, f, indent=4)

        with open(os.path.join(self.folder, 'text/runtime.txt'), mode='w') as f:
            f.write(self.runtime)

        with open(os.path.join(self.folder, 'binary/hyperparameters.p'), mode='wb') as f:
            pickle.dump(self.hyperparameters, f)

        with open(os.path.join(self.folder, 'binary/runtime.p'), mode='wb') as f:
            pickle.dump(round(elapsed, 2), f)

        with open(os.path.join(self.folder, 'binary/episodes.p'), mode='wb') as f:
            pickle.dump(self.episodes, f)



