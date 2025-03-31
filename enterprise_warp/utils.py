"""
Utility functions for enterprise_warp
"""

import glob
import json

def get_noise_dict(psrlist,noisefiles):
    """
    Reads in list of pulsar names and returns dictionary
    of {parameter_name: value} for all noise parameters.
    By default the input list is None and we use the 34 pulsars used in
    the stochastic background analysis.
    """

    params = {}
    json_files = sorted(glob.glob(noisefiles + '*.json'))
    for ff in json_files:
        if any([pp in ff for pp in psrlist]):
            with open(ff, 'r') as fin:
                params.update(json.load(fin))
    return params

def get_noise_dict_psr(psrname,noisefiles):
    """
    get_noise_dict for only one pulsar
    """
    params = dict()
    with open(noisefiles+psrname+'_noise.json', 'r') as fin:
        params.update(json.load(fin))
    return params
