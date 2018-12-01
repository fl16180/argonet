''' This script removes all generated results from the state results directory,
but maintains pre-processed input data files. '''

import os
import sys
import shutil
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib import config


KEEP_FILES = ['*_merged_data.csv', 'GFT_scaled.csv',
              ]


def refresh():
	for state in config.STATES:
		print 'Cleaning {0}'.format(state)

		try:
			path = config.STATES_DIR + '/{0}/'.format(state)
			os.chdir(path)
			all_files = glob.glob('*')

			keep = [glob.glob(x) for x in KEEP_FILES]

			for f in keep:
				all_files.remove(f[0])

			try:
				all_files.remove('pyplots')
				shutil.rmtree('pyplots')
			except ValueError:
				print '/pyplots/ already not present'

			for f in all_files:
				os.remove(f)


		except Exception as e:
			print e

	os.chdir(config.STATES_DIR + '/_overview/')
	files = glob.glob('*')
	for f in files:
		os.remove(f)


def rename():

    OLD_NAME = 'merged2'
    NEW_NAME = 'gt'

    for state in config.STATES:
        print 'Renaming files in {0}'.format(state)

        try:
            path = config.STATES_DIR + '/{0}/'.format(state)
            os.chdir(path)

            os.rename(OLD_NAME + '_preds.csv', NEW_NAME + '_preds.csv')
            os.rename(OLD_NAME + '_table.csv', NEW_NAME + '_table.csv')

        except Exception as e:
            print e

def move():

    for state in config.STATES:
        print 'Copying file in {0}'.format(state)

        try:
            path = config.STATES_DIR + '/{0}/'.format(state)
            os.chdir(path)

            shutil.copyfile('top_preds.csv', 'C:/Users/fredl/Desktop/{0}_top_preds.csv'.format(state))

        except Exception as e:
            print e

if __name__ == '__main__':

	# move()
    refresh()
