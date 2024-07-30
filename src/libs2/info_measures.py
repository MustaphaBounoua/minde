import numpy as np
import torch
from src.libs.util import concat_vect, deconcat  ,pop_elem_i
from src.libs.importance import get_normalizing_constant



def mi_cond_sigma(sde, s_marg ,s_cond, x_t, g, mean ,std,sigma, ,importance_sampling):
    
    """
    Calculates the total correlation (tc) between two variables using the score functions of the joint and marginal distributions.
    See : [1] Proposition 2 and Algorithm 2.
    
    [1] Bounoua, M., Franzese, G., & Michiardi, P. (2024). 
    S $\Omega $ I: Score-based O-INFORMATION Estimation. arXiv preprint arXiv:2402.05667.

    Parameters:
    sde (object): The SDE object.
    s_joint (list): The joint denoising score.
    s_marg (list): The marginal score .
    g (numpy.ndarray): The diffusion coefficient.
    importance_sampling (bool): Flag indicating whether to use importance sampling.

    Returns:
        The computed TC.
    """
    M = g.shape[0] 
    
    chi_t_x = mean ** 2 * sigma ** 2 + std**2
    ref_score_x = (x_t)/chi_t_x 
    const = get_normalizing_constant((1,)).to(sde.device)
    if importance_sampling:
        e_marg = -const * 0.5 * ((s_marg + std * ref_score_x)**2).sum() / M
        e_cond = -const * 0.5 * ((s_cond + std * ref_score_x)**2).sum() / M
    else:
        e_marg = -(g**2* ((s_marg + std * ref_score_x)**2) ).sum() / M
        e_cond = -(g**2* ((s_cond + std * ref_score_x)**2 ) ).sum() / M
    return e_marg - e_cond



def mi_cond(sde, s_marg ,s_cond, g, importance_sampling):
    
    """
    Calculates the total correlation (tc) between two variables using the score functions of the joint and marginal distributions.
    See : [1] Proposition 2 and Algorithm 2.
    
    [1] Bounoua, M., Franzese, G., & Michiardi, P. (2024). 
    S $\Omega $ I: Score-based O-INFORMATION Estimation. arXiv preprint arXiv:2402.05667.

    Parameters:
    sde (object): The SDE object.
    s_joint (list): The joint denoising score.
    s_marg (list): The marginal score .
    g (numpy.ndarray): The diffusion coefficient.
    importance_sampling (bool): Flag indicating whether to use importance sampling.

    Returns:
        The computed TC.
    """
    
    M = g.shape[0] 
    const = get_normalizing_constant((1,)).to(sde.device)
    if importance_sampling:
        mi = const *0.5* (s_marg - s_cond  )**2).sum()/ M
    else:
        mi = 0.5* (g**2*(s_marg - s_cond )**2).sum()/ M
        
    return mi.item()



def mi_joint(sde, s_marg ,s_cond, g ,importance_sampling):
    
    """
    Calculates the total correlation (tc) between two variables using the score functions of the joint and marginal distributions.
    See : [1] Proposition 2 and Algorithm 2.
    
    [1] Bounoua, M., Franzese, G., & Michiardi, P. (2024). 
    S $\Omega $ I: Score-based O-INFORMATION Estimation. arXiv preprint arXiv:2402.05667.

    Parameters:
    sde (object): The SDE object.
    s_joint (list): The joint denoising score.
    s_marg (list): The marginal score .
    g (numpy.ndarray): The diffusion coefficient.
    importance_sampling (bool): Flag indicating whether to use importance sampling.

    Returns:
        The computed TC.
    """
    
    M = g.shape[0] 

    if importance_sampling:
        const = get_normalizing_constant((1,)).to(sde.device)
        mi = const *0.5* ((s_marg - s_cond  )**2).sum()/ M
        
        
    else:
        tc = 0.5* (g**2*(concat_vect(s_joint) - concat_vect(s_marg)  )**2).sum()/ M
    return 



def compute_entropy(sde, score , x_t,x_0,  std, mean, g ,importance_sampling,sigma =1.0 ):
    """
    Compute the entropy of a given distribution using the technique in [1] equation 15.
    
    [1]"Franzese, G., Bounoua, M., & Michiardi, P. (2023). 
    MINDE: Mutual Information Neural Diffusion Estimation. arXiv preprint arXiv:2310.09031."
    
    Args:
        sde: The SDE class object .
        score: The score function of the distribution, if importance_sampling==False the score is not scaled by STD.
        x_t: The value of x at time t.
        x_0: The initial value of x.
        mean,std: The mean and std of the diffusion process P_t(x_t).
        g: The g value diffusion coefficient.
        importance_sampling: A boolean indicating whether to use importance sampling.
        sigma: The sigma value to use to compute the reference score (default is 1.0).

    Returns:
        The computed entropy.
    """
    
    khi_t =   mean **2 * sigma **2 + std**2 

    M = x_0.shape[0]
    N = x_0.shape[1]  

    term = N*0.5*np.log(2 *np.pi ) + 0.5* torch.sum(x_0**2)/M - 0.5 * N * torch.sum( torch.log(khi_t) -1 +  1 / khi_t ) 

    if importance_sampling:
        const = get_normalizing_constant((1,)).to(sde.device)
        h = term - const *0.5* ((score + std * x_t  /  khi_t  )**2).sum()/ M   
    else:
        h = term - 0.5* (g**2*(score + x_t  /  khi_t  )**2).sum()/ M  
    return h.item()






def get_tc(sde, s_joint ,s_marg, g ,importance_sampling):
    
    """
    Calculates the total correlation (tc) between two variables using the score functions of the joint and marginal distributions.
    See : [1] Proposition 2 and Algorithm 2.
    
    [1] Bounoua, M., Franzese, G., & Michiardi, P. (2024). 
    S $\Omega $ I: Score-based O-INFORMATION Estimation. arXiv preprint arXiv:2402.05667.

    Parameters:
    sde (object): The SDE object.
    s_joint (list): The joint denoising score.
    s_marg (list): The marginal score .
    g (numpy.ndarray): The diffusion coefficient.
    importance_sampling (bool): Flag indicating whether to use importance sampling.

    Returns:
        The computed TC.
    """
    
    M = g.shape[0] 

    if importance_sampling:

        const = get_normalizing_constant((1,)).to(sde.device)
        tc = const *0.5* ((concat_vect(s_joint) - concat_vect(s_marg)  )**2).sum()/ M
    else:
        tc = 0.5* (g**2*(concat_vect(s_joint) - concat_vect(s_marg)  )**2).sum()/ M
    return tc.item()



def get_dtc(sde, s_joint ,s_cond, g ,importance_sampling):
    
    """
    Calculates the total correlation (tc) between two variables using the score functions of the joint and conditional distributions.
    See : [1] Proposition 2 and Algorithm 2.
    
    [1] Bounoua, M., Franzese, G., & Michiardi, P. (2024). 
    S $\Omega $ I: Score-based O-INFORMATION Estimation. arXiv preprint arXiv:2402.05667.

    Parameters:
    sde (object): The SDE object.
    s_joint (list): The joint denoising score.
    s_marg (list): The marginal score .
    g (numpy.ndarray): The diffusion coefficient.
    importance_sampling (bool): Flag indicating whether to use importance sampling.

    Returns:
        The computed DTC.
    """
    
    M = g.shape[0] 

    if importance_sampling:
        const = get_normalizing_constant((1,)).to(sde.device)
        dtc = const *0.5* ((concat_vect(s_joint) - concat_vect(s_cond)  )**2).sum()/ M
    else:
       
        dtc = 0.5* (g**2*(concat_vect(s_joint) - concat_vect(s_cond)  )**2).sum()/ M

    return dtc.item()  


def compute_all_measures(tc,dtc):
    """Given TC and DTC, compute the other measures.
    """
    return {
                "tc": tc,
                "dtc":dtc,
                "o_inf": tc- dtc,
                "s_inf" :tc + dtc,
             } 
    
    

    
def compute_grad_o_inf(sde, s_marg ,s_con_i ,s_cond_ij, grad_var, g ,importance_sampling):
    """
    Calculate the gradient of the O-information directly using Equation 10-11.(See[1])
    
    [1] Bounoua, M., Franzese, G., & Michiardi, P. (2024). 
    S $\Omega $ I: Score-based O-INFORMATION Estimation. arXiv preprint arXiv:2402.05667.

    Args:
        sde: The SDE Class object.
        s_marg: Marginal scores.
        s_con_i: Conditional scores.
        s_cond_ij: Condtional scores ij .
        grad_var: The variable for which the gradient is calculated.
        g: The diffusion coefficient.
        importance_sampling: Boolean indicating whether importance sampling is used.

    Returns:
        The gradient of the output information measure.

    """
    M = g.shape[0] 

    if importance_sampling:
        const = get_normalizing_constant((1,)).to(sde.device)
        i_x_i = const *0.5* (( s_marg [grad_var] - s_con_i [grad_var]  )**2).sum()/ M
        i_xij =  torch.stack([const *0.5* (( s_marg[grad_var] - s_cond_ij[grad_var][var_j] )**2).sum()/ M
                                for var_j in s_marg.keys() if var_j != grad_var ]).sum()
        o_inf=  (2-len(s_marg.keys())  ) * i_x_i + i_xij    
    else:

        i_x_i = 0.5* (g**2*( s_marg [grad_var] - s_con_i [grad_var]  )**2).sum()/ M
        i_xij = 0.5* (g**2*(concat_vect( grad_var(s_marg,[grad_var])) - concat_vect( pop_elem_i(s_cond_ij[grad_var],[grad_var]) ) )**2).sum()/ M
        o_inf=  (2-len(s_marg.keys())  ) * i_x_i + i_xij
    return o_inf.item()
    
    





    

    
       

    