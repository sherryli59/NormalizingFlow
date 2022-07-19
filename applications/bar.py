import numpy as np

def logsum(a_n):
    
  # Compute the maximum argument.
  max_log_term = np.max(a_n)

  # Compute the reduced terms.
  terms = np.exp(a_n - max_log_term)

  # Compute the log sum.
  log_sum = np.log(sum(terms)) + max_log_term
        
  return log_sum

def BARzero(w_F,w_R,DeltaF):
    """
    ARGUMENTS
      w_F (np.array) - w_F[t] is the forward work value from snapshot t.
                        t = 0...(T_F-1)  Length T_F is deduced from vector.
      w_R (np.array) - w_R[t] is the reverse work value from snapshot t.
                        t = 0...(T_R-1)  Length T_R is deduced from vector.
      DeltaF (float) - Our current guess
    RETURNS
      fzero - a variable that is zeroed when DeltaF satisfies BAR.
    """

    # Recommended stable implementation of BAR.

    # Determine number of forward and reverse work values provided.
    T_F = float(w_F.size) # number of forward work values
    T_R = float(w_R.size) # number of reverse work values

    # Compute log ratio of forward and reverse counts.
    M = np.log(T_F / T_R)
    
    # Compute log numerator.
    # log f(W) = - log [1 + exp((M + W - DeltaF))]
    #          = - log ( exp[+maxarg] [exp[-maxarg] + exp[(M + W - DeltaF) - maxarg]] )
    #          = - maxarg - log[exp[-maxarg] + (T_F/T_R) exp[(M + W - DeltaF) - maxarg]]
    # where maxarg = max( (M + W - DeltaF) )
    exp_arg_F = (M + w_F - DeltaF)
    max_arg_F = np.choose(np.greater(0.0, exp_arg_F), (0.0, exp_arg_F))
    log_f_F = - max_arg_F - np.log( np.exp(-max_arg_F) + np.exp(exp_arg_F - max_arg_F) )
    log_numer = logsum(log_f_F) - np.log(T_F)
    
    # Compute log_denominator.
    # log_denom = log < f(-W) exp[-W] >_R
    # NOTE: log [f(-W) exp(-W)] = log f(-W) - W
    exp_arg_R = (M - w_R - DeltaF)
    max_arg_R = np.choose(np.greater(0.0, exp_arg_R), (0.0, exp_arg_R))
    log_f_R = - max_arg_R - np.log( np.exp(-max_arg_R) + np.exp(exp_arg_R - max_arg_R) ) - w_R 
    log_denom = logsum(log_f_R) - np.log(T_R)

    # This function must be zeroed to find a root
    fzero  = DeltaF - (log_denom - log_numer)

    return fzero
    
def BAR(w_F,w_R, DeltaF=0.0, maximum_iterations=1000,relative_tolerance=1.0e-5):
    for iteration in range(maximum_iterations):
        DeltaF_old = DeltaF
        DeltaF = -BARzero(w_F,w_R,DeltaF) + DeltaF
        relative_change = abs((DeltaF - DeltaF_old)/DeltaF)      
        if ((iteration > 0) and (relative_change < relative_tolerance)):
            #print("tolerance reached at iteration%d"%iteration)
            break
    return DeltaF