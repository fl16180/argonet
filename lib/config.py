'''config.py

	Sets global locations for state-flu project allowing all scripts and functions to access
	central directories such as data and output.

	Also defines global constants such as state abbreviations.

'''

# define project top-level directory.
import os
HOME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# define data directories
DATA_DIR = HOME_DIR + '/data'
GT_DIR = DATA_DIR + '/GT'
ATH_LOC = DATA_DIR + '/athena/ATHdata.csv'
STATE_ILI_LOC = DATA_DIR + '/ILI/States_ILI.csv'

# define output directory
STATES_DIR = HOME_DIR + '/results'

STATES = ("AL,AK,AZ,AR,CA,CO,CT,DE,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,"
          "MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY"
          ).split(',')

USE_DATA_FROM = '2009-10-04'
END_DATE = '2017-05-14'
