"""
Custom selection file: add your own Enterprise selections.
Make sure to apply enterprise.signals.selections.Selection() on your function.
"""

from enterprise.signals.selections import *

def custom_selection(flags,custom_flag_bor,flagval_bor):
    """
    Arguments "custom_flag_bor" and "flagval_bor" are variables
    inside Enterprise Pulsar object - they can be added there manually.
    """
    custom_flag = custom_flag_bor
    flagval = flagval_bor
    if flagval==None or custom_flag==None:
        print('Kwargs flagval and custom_flag must be specified!')
        raise ValueError
    seldict = dict()
    seldict[flagval] = flags[custom_flag_bor]==flagval_bor
    return seldict
