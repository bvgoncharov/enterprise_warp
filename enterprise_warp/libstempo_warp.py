import numpy as np

import enterprise.constants as const
import libstempo.toasim as LT

def red_psd(ff,AA,gamma):
    norm = AA**2 * const.yr**3 / (12 * np.pi**2)
    return norm * (ff*const.yr)**(-gamma)

def red_v1_psd(ff,AA,gamma,fc):
    norm = AA**2 * const.yr**3 / (12 * np.pi**2)
    return norm * ((ff+fc)*const.yr)**(-gamma)

def dm_psd(ff,AA,gamma,max_rad_freq_hz,DMk=4.15e3):
    return red_psd(ff,AA,gamma)*DMk/max_rad_freq_hz**2

def lorenzian_red_psd(ff,PP,fc,alpha):
    return PP / (1 + (ff/fc)**2)**(alpha/2)

def plot_noise_psd_from_dict(psr, psd_params, backends, ff):
   for backend in backends:
       label = 'RMS white noise in '+backend
       wpsd = psd_params[backend]['rms_toaerr']*1e-6
       wpsd = np.repeat(wpsd,len(ff))
       plt.loglog(ff,wpsd,label=label)

   if 'red' in psd_params.keys():
       if 'A' in psd_params['red']:
           label = 'Red noise, lgA='+\
               str(round(np.log10(psd_params['red']['A']),2))+\
               ', gamma='+str(round(psd_params['red']['gamma'],2))
           plt.loglog(ff, red_psd(ff,psd_params['red']['A'],\
               psd_params['red']['gamma']),label=label)
       elif 'P' in psd_params['red']:
           label = 'Red noise, lgP='+\
               str(round(np.log10(psd_params['red']['P']),2))+\
               ', alpha='+str(round(psd_params['red']['alpha'],2))+\
               ', lg(fc)='+str(round(np.log10(psd_params['red']['fc']),2))
           plt.loglog(ff, lorenzian_red_psd(ff,psd_params['red']['P'],\
               psd_params['red']['fc'],psd_params['red']['alpha']),label=label)

   if 'dm' in psd_params.keys():
       if 'A' in psd_params['dm']:
           label = 'DM noise at max freq, lgA='+\
               str(round(np.log10(psd_params['dm']['A']),2))+\
               ', gamma='+str(round(psd_params['dm']['gamma'],2))
           #plt.loglog(ff, dm_psd(ff,psd_params['dm']['A'],\
           #    psd_params['dm']['gamma'],max(psr.freqs)*1e9),label=label)
           print('Function dm_psd in core.py: not clear how to include DM \
                  constant and radio frequencies')
           print('Not plotting DM noise PSD')

def add_noise(t2pulsar, noise_dict, sim_dm=True, sim_white=True, sim_red=True,
              seed=None):
   """
   Recognize noise from noise parameter name, and add to t2pulsar.
   """

   added_noise_psd_params = dict()
   flagid=list()
   if 'f' in t2pulsar.flags():
      backends = np.unique(t2pulsar.flagvals('f'))
      flagid.append('f')
   if 'g' in t2pulsar.flags(): #PPTA
      backends = np.append(backends, np.unique(t2pulsar.flagvals('g')) )
      flagid.append('g')
   if 'sys' in t2pulsar.flags() and not 'group' in t2pulsar.flags():
      backends = np.unique(t2pulsar.flagvals('sys'))
      flagid.append('sys')
   if 'sys' in t2pulsar.flags() and 'group' in t2pulsar.flags():
      backends = np.unique(t2pulsar.flagvals('group'))
      flagid.append('group')
   if flagid==[]:
      backends = []
      raise Exception('Backend convention is not recognized')

   used_backends=list()
   for noise_param, noise_val in noise_dict.items():
      if not np.isscalar(noise_val):
         noise_val = noise_val[0]
         noise_dict[noise_param] = noise_dict[noise_param][0]
      backend_name = ''
      param_name = ''
      for bcknd in backends:
         if bcknd in noise_param:
            used_backends.append(bcknd)
            backend_name = bcknd
            for fid in flagid: # for 2 flags in PPTA
              flagid_bcknd = ''
              if backend_name in t2pulsar.flagvals(fid):
                flagid_bcknd = fid
            added_noise_psd_params.setdefault(backend_name,dict())
            toaerr_bcknd = t2pulsar.toaerrs[ np.where(\
                           t2pulsar.flagvals(flagid_bcknd)==backend_name)[0] ]
            added_noise_psd_params[backend_name]['rms_toaerr'] = \
                              (np.sum(toaerr_bcknd**2)/len(toaerr_bcknd))**(0.5)
            added_noise_psd_params[backend_name]['mean_toaerr'] = \
                               np.mean(toaerr_bcknd)

      if 'efac' in noise_param.lower() and sim_white:
         param_name = 'efac'
         if not backend_name == '':
            #LT.add_efac(t2pulsar, efac=noise_val, seed=seed,\
            #            flags=backend_name, flagid=flagid_bcknd)
            added_noise_psd_params[backend_name]['efac'] = noise_val
            print('Added efac: ',noise_val,backend_name)
         elif noise_param==t2pulsar.name+'_efac':
            #LT.add_efac(t2pulsar, efac=noise_val, seed=seed)
            added_noise_psd_params['efac'] = noise_val
            print('Added efac: ',noise_val)
         else:
            raise Exception('Efac is not recognized. Neither signle, nor per \
                             backend. Parameter name from noise file: ', \
                             noise_param,'. Backends: ',backends)
   
      elif 'log10_equad' in noise_param.lower() and sim_white:
         param_name = 'log10_equad'
         if not backend_name == '':
            #LT.add_equad(t2pulsar, equad=10**noise_val, seed=seed, \
            #             flags=backend_name, flagid=flagid_bcknd)
            added_noise_psd_params[backend_name]['equad'] = 10**noise_val
            print('Added equad: ',10**noise_val,backend_name)
         elif noise_param==t2pulsar.name+'_log10_equad':
            #LT.add_equad(t2pulsar, equad=10**noise_val, seed=seed)
            added_noise_psd_params['equad'] = 10**noise_val
            print('Added efac: ',10**noise_val)
         else:
            raise Exception('Equad is not recognized. Neither signle, nor per \
                             backend. Parameter name from noise file: ', \
                             noise_param,'. Backends: ',backends)
   
      elif 'log10_ecorr' in noise_param.lower() and sim_white:
         param_name = 'log10_ecorr'
         if not backend_name == '':
            #LT.add_jitter(t2pulsar, ecorr=10**noise_val, seed=seed, \
            #              flags=backend_name, flagid=flagid_bcknd)
            added_noise_psd_params[backend_name]['ecorr'] = 10**noise_val
            print('Added ecorr: ',10**noise_val,backend_name)
         elif noise_param==t2pulsar.name+'_log10_ecorr':
            #LT.add_jitter(t2pulsar, ecorr=10**noise_val, seed=seed)
            added_noise_psd_params['ecorr'] = 10**noise_val
            print('Added efac: ',noise_val)
         else:
            raise Exception('Ecorr is not recognized. Neither signle, nor per backend. Parameter name from noise file: ',noise_param,'. Backends: ',backends)
   
      elif 'dm_gp_log10_A' in noise_param and sim_dm:
         param_name = 'dm_gp_log10_A'
         #LT.add_dm(t2pulsar,A=10**noise_val,\
         #          gamma=noise_dict[t2pulsar.name+'_dm_gp_gamma'],\
         #          seed=seed,components=30)
         added_noise_psd_params.setdefault('dm',dict())
         added_noise_psd_params['dm']['A'] = 10**noise_val
         print('Added DM noise, A=',10**noise_val,' gamma=',noise_dict[t2pulsar.name+'_dm_gp_gamma'])
      elif 'dm_gp_gamma' in noise_param and sim_dm:
         added_noise_psd_params.setdefault('dm',dict())
         added_noise_psd_params['dm']['gamma'] = noise_val
         param_name = 'dm_gp_gamma'
         pass

      elif noise_param==t2pulsar.name+'_log10_A' and sim_red:
         param_name = 'log10_A'
         #LT.add_rednoise(t2pulsar,A=10**noise_val,\
         #                gamma=noise_dict[t2pulsar.name+'_gamma'],\
         #                seed=seed,components=30)
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['A'] = 10**noise_val
         print('Added red noise, A=',10**noise_val,' gamma=',noise_dict[t2pulsar.name+'_gamma'])
      elif noise_param==t2pulsar.name+'_gamma':
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['gamma'] = noise_val
         param_name = 'gamma'
         pass

      elif 'log10_P0' in noise_param:
         param_name = 'log10_P0'
         #LT_custom.add_lorenzianrednoise(t2pulsar,P=10**noise_val,\
         #                            fc=10**noise_dict[t2pulsar.name+'_fc'],\
         #                            alpha=noise_dict[t2pulsar.name+'_alpha'],\
         #                            seed=seed)
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['P'] = 10**noise_val
         print('Added red noise, P=',10**noise_val,' fc=',noise_dict[t2pulsar.name+'_fc'], ' alpha=',noise_dict[t2pulsar.name+'_alpha'])
      elif 'alpha' in noise_param:
         param_name = 'alpha'
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['alpha'] = noise_val
         pass
      elif 'fc' in noise_param:
         param_name = 'fc'
         added_noise_psd_params.setdefault('red',dict())
         added_noise_psd_params['red']['fc'] = 10**noise_val
         pass

      else:
         print('Warning: parameter ',noise_param,' is not recognized or \
                switched off. It was not used to simulate any data.')

   if sim_white:
      vector_vals, vector_bckd = added_noise_psd_to_vector( \
              added_noise_psd_params,param='efac')
      LT.add_efac(t2pulsar, efac=vector_vals, seed=seed, flags=vector_bckd, 
                  flagid=flagid_bcknd)
      vector_vals, vector_bckd = added_noise_psd_to_vector( \
              added_noise_psd_params,param='equad')
      LT.add_equad(t2pulsar, equad=vector_vals, seed=seed, flags=vector_bckd, 
                   flagid=flagid_bcknd)

   if sim_red:
      Ared = added_noise_psd_params['red']['A']
      gred = added_noise_psd_params['red']['gamma']
      LT.add_rednoise(t2pulsar,A=Ared,gamma=gred,seed=seed,components=30)

   if sim_dm:
      Adm = added_noise_psd_params['dm']['A']
      gdm = added_noise_psd_params['dm']['gamma']
      LT.add_dm(t2pulsar,A=Adm,gamma=gdm,seed=seed,components=30)

   used_backends = np.unique(used_backends)
   backends = np.unique(backends)

   if len(backends)!=len(used_backends):
      print('[!] Warning, number of not used backends: ',\
            len(backends)-len(used_backends))

   return t2pulsar, used_backends, added_noise_psd_params

def added_noise_psd_to_vector(added_noise_psd_params,param='efac'):
    """
    Conversion of dict to vector to simulate efac/equad in libstempo
    """
    vector_vals = list()
    vector_bckd = list()
    for key, val in added_noise_psd_params.items():
        if param in val:
            vector_vals.append( val[param] )
            vector_bckd.append( key )
    return vector_vals, vector_bckd
