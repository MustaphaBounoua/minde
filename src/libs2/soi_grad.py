import numpy as np
import torch
from ..libs.util import concat_vect, deconcat, marginalize_data, cond_x_data, pop_elem_i, marginalize_one_var, get_samples
from ..libs.info_measures import get_tc, get_dtc, compute_all_measures, compute_grad_o_inf,compute_entropy

from .soi import SOI


class SOI_grad(SOI):
    def __init__(self, args, nb_var=None, gt=None):
        args.o_inf_order = 2
        super(SOI_grad, self).__init__(
            args, nb_var=nb_var, gt=gt)

    def infer_scores_for_o_inf_grad(self, x_t,t, data, std_w, marg_masks, cond_mask,do_s_joint_marg = True):
        """_summary_

        Args:
            x_t : the noisy data.
            t : the time t
            data :the clean data
            std_w : rescale the score
            marg_masks : the masks for the marginals
            cond_mask : the masks for the conditionals
            s_joint_marg (bool): return S(X\i) or not. (default: {True}

        Returns:
            s_joint, s_marg, s_cond_x, s_joint_marg, s_cond_ij
        """
        X_t = deconcat(x_t, self.var_list, self.sizes)

        # Prepare the marginals input by replacing the non marginals with zeros or gaussian noise
        # X_t^i = [Noise, Noise, Noise, Noise,X_t^{i}, Noise, Noise , Noise]
        marginals = {var: marginalize_data(X_t, var, fill_zeros=(
            self.marginals == -1))  for var in self.var_list}

        # X_t^\i = [X_t^1, X_t^2, ... , X_t^{i-1},X_t^{i} = Noise, X_t^{i+1}, ... , X_t^N]
        joint_marginals = {var: marginalize_one_var(
            X_t, var, fill_zeros=(self.marginals == -1)) for var in self.var_list}

        # X_t^i|X_t^\i = [X^1, X^2, ... , X^{i-1}, X_t^{i} , X^{i+1}, ... , X_t^N]
        cond_x = {var: cond_x_data(X_t, data, var) for var in self.var_list}

        # X_t^i|X_t^\ij = [X^1, X^2, ... , X^{i-1}, X_t^{i} , .., X_t^{j} = Noise , ... , X_t^N]
        cond_ij = {var_i: {var_j: marginalize_one_var(cond_x[var_i], var_j, fill_zeros=(self.marginals == -1)) 
                           for var_j in self.var_list
                           } for var_i in self.var_list}

        with torch.no_grad():
           
            s_joint = - self.score_inference(x_t, t=t, mask=torch.ones_like(
                marg_masks[self.var_list[0]]), std=std_w).detach()
            s_marg = {}
            s_cond_x = {}
            s_cond_ij = {}
            s_joint_marg = {}
            for var in self.var_list:
             
                s_marg[var] = - self.score_inference(
                    concat_vect(marginals[var]), t=t, mask=marg_masks[var],
                    std=std_w).detach()
                
                if do_s_joint_marg:
                    s_joint_marg[var] = - self.score_inference(
                        concat_vect(joint_marginals[var]), t=t, mask= - marg_masks[var], std=std_w).detach()
                else:
                    s_joint_marg[var]=None
                
            
                s_cond_x[var] = - self.score_inference(
                    concat_vect(cond_x[var]), t=t, 
                    mask=cond_mask[var], std=std_w).detach()
             
                s_cond_ij[var] = {}

                for var_j in self.var_list:
                    if var_j != var:
                        s_cond_ij[var][var_j] = - self.score_inference(
                            concat_vect(cond_ij[var][var_j]), t=t, mask=cond_mask[var] - (marg_masks[var_j] > 0).int(), std=std_w).detach()
                    else:
                        s_cond_ij[var][var_j] = torch.zeros_like(
                            s_cond_x[var])

        s_joint = deconcat(s_joint, self.var_list, self.sizes)

        # deconcat each marginal score and put them in the same dict
        # S^i = [S(X^1), S(X^2), ... , ... , S(X^N)]
        s_marg = {var: deconcat(s_marg[var], self.var_list, self.sizes)[
            var] for var in self.var_list}

        # {"var_i": "S(X^i|X^\i)"}
        s_cond_x = {var: deconcat(s_cond_x[var], self.var_list, self.sizes)[
            var] for var in self.var_list}

        # {"var_i": "S(X^\i)"}
        if do_s_joint_marg:
            s_joint_marg = {var: deconcat(s_joint_marg[var], self.var_list, self.sizes)
                        for var in self.var_list}

        # {"var_i": {"var_j": "S(X^i|X^\ij)" }}
        s_cond_ij = {
            var_i: {var_j: deconcat(s_cond_ij[var_i][var_j], self.var_list, self.sizes)[var_i] 
                    for var_j in self.var_list}
            for var_i in self.var_list
        }

        return s_joint, s_marg, s_cond_x, s_joint_marg, s_cond_ij


    def compute_o_inf_with_grad(self, data=None, eps=1e-5):
        """
        Compute the gradient of o_inf b ycomputeing \Omega(X)- \Omega(X^\i) for each variable i.

        Args:
            data (dict): A dictionary containing the input data.{x0:  , x1: , x2: , ...}
            importance_sampling (bool, optional): Flag indicating whether to use importance sampling. Defaults to False.
            eps (float, optional): A small value used for sampling if importance sampling is set to false. Defaults to 1e-5.
            nb_iter (int, optional): Number of iterations for Monte Carlo estimation. Defaults to 10.

        Returns:
            tuple: A tuple containing the computed o_inf measures.
        """
        self.eval()
        self.to("cuda" if self.args.accelerator == "gpu" else "cpu")
        if data ==None:
            data= self.test_samples
        if isinstance(data, dict)==False:
            data = get_samples(data, device=self.device)
        self.sde.device = self.device
        var_list = list(data.keys())
        data_0 = {x_i: data[x_i].to(self.device) for x_i in var_list}
        x_0 = concat_vect(data_0)

        N = len(self.sizes)
        M = x_0.shape[0]

        marg_masks, cond_mask = self.get_masks(var_list)

        tc_mc = []
        dtc_mc = []

        minus_tc_mc = {var: [] for var in data.keys()}
        minus_dtc_mc = {var: [] for var in data.keys()}

        for i in range(self.args.mc_iter):
            if self.args.importance_sampling:
                t = (self.sde.sample_importance_sampling_t(
                    shape=(M, 1))).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
            _, g = self.sde.sde(t)

            x_t, _, _, std = self.sde.sample(x_0=x_0, t=t)
         
            std_w = None if self.args.importance_sampling else std

            s_joint, s_marg, s_cond_x, s_joint_marg, s_cond_x_ij = self.infer_scores_for_o_inf_grad(
                x_t,t, data, std_w, marg_masks, cond_mask)

            tc_mc.append(get_tc(self.sde, s_joint=s_joint, s_marg=s_marg,
                        g=g,  importance_sampling=self.args.importance_sampling))
            dtc_mc.append(get_dtc(self.sde, s_joint=s_joint,  s_cond=s_cond_x,
                          g=g, importance_sampling=self.args.importance_sampling))

            for grad_var in var_list:

                minus_tc_mc[grad_var].append(get_tc(self.sde, s_joint=pop_elem_i(s_joint_marg[grad_var],  [grad_var]),
                                                    s_marg=pop_elem_i(s_marg, [grad_var]), g=g,  importance_sampling=self.args.importance_sampling))
                ## {var :  S(X^i|X^\ij) } , j = grad_var, i != grad_var
                s_cond_i_grad_var =   {var : s_cond_x_ij[var][grad_var]  for var in var_list if var != grad_var}
                
                minus_dtc_mc[grad_var].append(get_dtc(self.sde, s_joint=pop_elem_i(s_joint_marg[grad_var], [grad_var]),
                                                      s_cond= s_cond_i_grad_var, g=g,
                                                      importance_sampling=self.args.importance_sampling))



        tc, dtc = np.mean(tc_mc), np.mean(dtc_mc)
        out = compute_all_measures(tc, dtc)
        tc_minus, dtc_minus = {var: np.mean(minus_tc_mc[var]) for var in var_list}, {var: np.mean(minus_dtc_mc[var]) for var in var_list}
        
        g_o_inf = {var: out["o_inf"] - (tc_minus[var] - dtc_minus[var]) for var in var_list}

        out["g_o_inf"] = g_o_inf
        out["tc_minus"] = tc_minus
        out["dtc_minus"] = dtc_minus

        return out

    def compute_o_inf_with_grad_2(self, data=None, eps=1e-5):
        """
        Compute the gradient of o_inf using equation 11 from the paper which does not require S(X^\i).

        Args:
            data (dict): A dictionary containing the input data.{x0:  , x1: , x2: , ...}
            importance_sampling (bool, optional): Flag indicating whether to use importance sampling. Defaults to False.
            eps (float, optional): A small value used for sampling if importance sampling is set to false. Defaults to 1e-5.
            nb_iter (int, optional): Number of iterations for Monte Carlo estimation. Defaults to 10.

        Returns:
            tuple: A tuple containing the computed o_inf measures.
        """
        self.eval()
        self.to("cuda" if self.args.accelerator == "gpu" else "cpu")
        if data ==None:
                data= self.test_samples
        if isinstance(data, dict)==False:
            ## ToDo: Rewrite a batched version of the function
            data=get_samples(data, device=self.device, N=10000)
            
        self.sde.device = self.device
        var_list = list(data.keys())
        data_0 = {x_i: data[x_i].to(self.device) for x_i in var_list}
        x_0 = concat_vect(data_0)

        N = len(self.sizes)
        M = x_0.shape[0]

        marg_masks, cond_mask = self.get_masks(var_list)

        tc_mc = []
        dtc_mc = []

        g_o_inf = {var: [] for var in data.keys()}

        for i in range(self.args.mc_iter):
            if self.args.importance_sampling:
                t = (self.sde.sample_importance_sampling_t(
                    shape=(M, 1))).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
            _, g = self.sde.sde(t)

            x_t, _, _, std = self.sde.sample(x_0=x_0, t=t)

            std_w = None if self.args.importance_sampling else std

            s_joint, s_marg, s_cond_x, _ , s_cond_x_ij = self.infer_scores_for_o_inf_grad(
                x_t,t, data, std_w, marg_masks, cond_mask ,do_s_joint_marg = False )

            tc_mc.append(get_tc(self.sde, s_joint=s_joint, s_marg=s_marg,
                                g=g,  importance_sampling=self.args.importance_sampling))
            dtc_mc.append(get_dtc(self.sde, s_joint=s_joint,  s_cond=s_cond_x,
                                  g=g, importance_sampling=self.args.importance_sampling))

            for grad_var in var_list:

                g_o_inf[grad_var].append(compute_grad_o_inf(self.sde, s_marg=s_marg,
                                                        s_con_i=s_cond_x,
                                                        s_cond_ij=s_cond_x_ij,
                                                        grad_var=grad_var, g=g,
                                                        importance_sampling=self.args.importance_sampling))

          

        tc, dtc = np.mean(tc_mc), np.mean(dtc_mc)
        out = compute_all_measures(tc, dtc)
        g_o_inf = {var: np.mean(g_o_inf[var]) for var in var_list}

        out["g_o_inf"] = g_o_inf
        return out

    def compute_entropies(self, data,eps=1e-5):
        """
        Compute the gradient of o_inf.

        Args:
            data (dict): A dictionary containing the input data.{x0:  , x1: , x2: , ...}
            importance_sampling (bool, optional): Flag indicating whether to use importance sampling. Defaults to False.
            eps (float, optional): A small value used for sampling if importance sampling is set to false. Defaults to 1e-5.
            nb_iter (int, optional): Number of iterations for Monte Carlo estimation. Defaults to 10.

        Returns:
            tuple: A tuple containing the computed o_inf measures.
        """
        
        self.sde.device = self.device
        var_list = list(data.keys())
        data_0 = {x_i: data[x_i].to(self.device) for x_i in var_list}
        x_0 = concat_vect(data_0)

       
        M = x_0.shape[0]

        marg_masks, cond_mask = self.get_masks(var_list)

        e_joint_mc = []
        e_marg_i_mc = {var: [] for var in data.keys()}
        e_i_cond_slash = {var: [] for var in data.keys()}
        e_cond_ij_mc = {var: [] for var in data.keys()}
        e_joint_i_mc = {var: [] for var in data.keys()}
        for i in range(self.args.mc_iter):
            if self.args.importance_sampling:
                t = (self.sde.sample_importance_sampling_t(
                    shape=(M, 1))).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
            f, g = self.sde.sde(t)
            x_t, z_m, mean, std = self.sde.sample(x_0=x_0, t=t)
            X_t = deconcat(x_t, var_list, self.sizes)

            std_w = None if self.args.importance_sampling else std

            s_joint, s_marg, s_cond_x, s_joint_marg, s_cond_x_ij = self.infer_scores_for_o_inf_grad(
                x_t,t, data, std_w, marg_masks, cond_mask)
            
            e_joint_mc.append(compute_entropy(self.sde, score=concat_vect(s_joint),
                                            x_t=x_t, std=std, g=g,
                                            importance_sampling=self.args.importance_sampling,
                                            x_0=x_0,
                                            mean=mean)) ## Compute H(X)

            for var_minus_i in var_list:
                
                ## Compute H(X^i|X^{\i})
                e_i_cond_slash[var_minus_i].append(compute_entropy(self.sde, score=s_cond_x[var_minus_i],
                                                                       x_t=X_t[var_minus_i],
                                                                       x_0=data[var_minus_i],
                                                                       std=std, g=g,  importance_sampling=self.args.importance_sampling,
                                                                       mean=mean))
                ## Compute H(X_i)
                e_marg_i_mc[var_minus_i].append(compute_entropy(self.sde, score=s_marg[var_minus_i],
                                                              x_t=X_t[var_minus_i],
                                                              x_0=data[var_minus_i],
                                                              std=std, g=g,
                                                              importance_sampling=self.args.importance_sampling,
                                                              mean=mean))
                
                ## Compute H(X_{\i})
                e_joint_i_mc[var_minus_i].append(compute_entropy(self.sde, score=concat_vect(pop_elem_i( s_joint_marg[var_minus_i],[var_minus_i])),
                                                              x_t=concat_vect(pop_elem_i( X_t,[var_minus_i]) ),
                                                              x_0=concat_vect(pop_elem_i( data_0,[var_minus_i])),
                                                              std=std, g=g,
                                                              importance_sampling=self.args.importance_sampling,
                                                              mean=mean))

                if var_minus_i != var_list[0]:
                ## compute H(X_i|X_{-ij}) for only i=0
                
                    e_cond_ij_mc[var_minus_i].append(compute_entropy(self.sde, score=s_cond_x_ij[var_list[0]][var_minus_i],
                                                              x_t = X_t[var_list[0]],
                                                              x_0 = data[var_list[0]],
                                                              std = std, g=g,
                                                              importance_sampling=self.args.importance_sampling,
                                                              mean=mean))
                
                
            
        e_joint = np.mean(e_joint_mc)
        e_marg_i = [np.mean(e_marg_i_mc[key]) for key in var_list]
        
        e_i_cond_slash = [np.mean(e_i_cond_slash[key]) for key in var_list]
        e_joint_i = [ np.mean(e_joint_i_mc[var]) for var in var_list ]
        e_cond_ij = [ np.mean(e_cond_ij_mc[var]) for var in var_list[1:] ]

        return {"e_joint": e_joint, "e_marg_i": e_marg_i,"e_joint_i":e_joint_i, "e_i_cond_slash": e_i_cond_slash,"e_cond_ij":e_cond_ij}



    

    def calculate_hidden_dim(self):
        # return dimensions for the hidden layers
        if self.args.arch == "mlp":
            return super().calculate_hidden_dim() *2
        else:
            dim_m = np.max(self.sizes)
            if dim_m <= 5:
                htx = 72
            elif dim_m <= 10:
                htx = 96
            else:
                htx = 120
            return htx

        

    def logger_estimates(self):
        # Log the estimates of the o_inf measures durring training

        if self.current_epoch % self.args.test_epoch == 0 and self.current_epoch != 0:
            r = self.compute_o_inf_with_grad(data=self.test_samples)
         
            print("O_inf - GT: ", np.round( self.gt["o_inf"],decimals=3 ) , "O_inf - SOI_new: ", np.round(  r["o_inf"],decimals=3 ))
            
            print("Gradient O_inf - GT: ", " ".join([ "x{}: {},".format(i,np.round(x,decimals=3) ) for i,x in enumerate(self.gt["g_o_inf"]) ]) )
            print("Gradient O_inf - SOI:" , " ".join([ "x{}: {},".format(i,np.round(x,decimals=3) ) for i,x in enumerate( list(r["g_o_inf"].values() )) ] ))
            
            for met in ["tc", "o_inf", "dtc"]:
                self.logger.experiment.add_scalars('Measures/{}'.format(met),
                                                   {'gt': self.gt[met],
                                                    'e': r[met],
                                                    }, global_step=self.global_step)

    
            for index, var in enumerate( self.var_list) :
                for met in ["g_o_inf"]: #,"tc_minus","dtc_minus"]:
                    self.logger.experiment.add_scalars('{}/i_{}'.format(met,var) ,
                                                    {'gt': self.gt[met][index] , 
                                                        'e': r[met] [var] ,
                                                        }, global_step=self.global_step)
                    
            if self.args.debug:
                entropies = self.compute_entropies(data=self.test_samples)
                for i in range(len(self.var_list)):

                    self.logger.experiment.add_scalars('Debbug/e_marg_i{}'.format(i),
                                                       {'gt': self.gt["e_marg_i"][i],
                                                        'e': entropies["e_marg_i"][i],
                                                        }, global_step=self.global_step)
                    
                    self.logger.experiment.add_scalars('Debbug/e_joint_i{}'.format(i),
                                                       {'gt': self.gt["e_minus_i"][i],
                                                        'e': entropies["e_joint_i"][i],
                                                        }, global_step=self.global_step)

                    self.logger.experiment.add_scalars('Debbug/e_i_cond_slash{}'.format(i),
                                                       {'gt': self.gt["e_joint"] - self.gt["e_minus_i"][i],
                                                        'e': entropies["e_i_cond_slash"][i],
                                                        }, global_step=self.global_step)
                    if i!=len(self.var_list)-1:
                        self.logger.experiment.add_scalars('Debbug/e_cond_ij_{}'.format(i),
                                                       {'gt': self.gt["e_cond_ij"][i],
                                                        'e': entropies["e_cond_ij"][i],
                                                        }, global_step=self.global_step)
                    

                self.logger.experiment.add_scalars('Debbug/e_joint',
                                                   {'gt': self.gt["e_joint"],
                                                    'e': entropies["e_joint"],
                                                    }, global_step=self.global_step)
                
                
