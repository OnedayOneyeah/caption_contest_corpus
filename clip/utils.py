import os
import json
import zipfile
import numpy as np
import pickle
from collections import OrderedDict, Counter
import pandas as pd

def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()

def mkdirp(p):
    # if not os.path.exists(p):
    os.makedirs(p, exist_ok=True)