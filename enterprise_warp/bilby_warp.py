import bilby

class PTABilbyLikelihood(bilby.Likelihood):
    def __init__(self, pta, parameters):
        self.pta = pta
        self.parameters = parameters
        
    def log_likelihood(self):
        return self.pta.get_lnlikelihood(self.parameters)

    def get_one_sample(self):
        return {par.name: par.sample() for par in pta[0].params}

def get_bilby_prior_dict(pta):
    ''' Get Bilby parameter dict from Enterprise PTA object '''
    priors = dict()
    for param in pta.params:

      if param.size==None:
        if param.type=='uniform':
          #priors[param.name] = bilby.core.prior.Uniform( \
          #    param._pmin, param._pmax, param.name)
          priors[param.name] = bilby.core.prior.Uniform( \
              # param._pmin
              param.prior._defaults['pmin'], param.prior._defaults['pmax'], \
              param.name)
        elif param.type=='normal':
          #priors[param.name] = bilby.core.prior.Normal( \
          #    param._mu, param._sigma, param.name)
          priors[param.name] = bilby.core.prior.Normal( \
              param.prior._defaults['mu'], param.prior._defaults['sigma'], \
              param.name)
      else:
        if param.name=='jup_orb_elements' and param.type=='uniform':
          for ii in range(param.size):
            priors[param.name+'_'+str(ii)] = bilby.core.prior.Uniform( \
                -0.05, 0.05, param.name+'_'+str(ii))

    # Consistency check
    for key, val in priors.items():
        if key not in pta.param_names:
          print('[!] Warning: Bilby\'s ',key,' is not in PTA params:',\
              pta.param_names)

    return priors 
