# tugasstatistikpacmann
````python
    import json
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set_style('whitegrid')
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 100)
# source - https://fantasy.premierleague.com/drf/bootstrap-static
with open('/Users/avntrr/Documents/Pacmann/FPL/data.json') as data_file:    
    data = json.load(data_file)
````
