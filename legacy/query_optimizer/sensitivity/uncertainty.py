import numpy as np
import math
import torch

'''
modified uncertainty calculations from https://github.com/plahoti-lgtm/RiskAdvisor

Uses renyi entropy instead of shannon entropy
'''

def renyi_entropy(prob,alpha):
    entropy=-1*math.log2(np.sum(prob**2))
    return entropy

def compute_renyi_uncertainty(p_y_x,alpha):
    return np.apply_along_axis(renyi_entropy,axis=1,arr=p_y_x,alpha=alpha)

def compute_total_uncertainity(p_y_x_marginal_h,alpha):
    """computes total uncertainitity given marginal p(y|x) over all h \in H.
    
    Args:
        pred_prob: np.ndarray of p(y|x) 
    Returns:
        a 1D array of per item uncertainities:  
    """
    return compute_renyi_uncertainty(p_y_x_marginal_h,alpha)

    
def compute_aleatoric_uncertainity(all_hyp_p_y_h_x,alpha):
    """computes total uncertainitity given p(y|h,x) for all h \in H.
    
    Args:
        pred_prob: np.ndarray of p(y|h,x) 
            for all x \in X 
                    h \in H
                    y in Y
    Returns:
        a 1D array of per item uncertainities:  
    """
    all_hyp_uncertainity = []
    for p_y_h_x in all_hyp_p_y_h_x:
        all_hyp_uncertainity.append(compute_renyi_uncertainty(p_y_h_x,alpha))
    return np.array(all_hyp_uncertainity).mean(axis=0)


def get_aleatoric_epistemic_uncertainities(ensemble_clf,X,alpha):
    """Given an trained ensemble auditor, compute predicted prob of error, aleatoric and epistemic uncertainities
    Args:
        ensemble_clf: Fitted ensemble model
        X: data records x \in X
    Returns:
        pred_prob_error
        aleatoric_uncertainity: np.array of aleatoric_uncertainity for each x \in X
        epistemic_uncertainities: np.array of aleatoric_uncertainity for each x \in X
    """
    # Fetch pred_prob P(z|h,x) of each SGBT in the E-SGBT (auditor ensemble model)
    # aud_estimators = ensemble_clf.estimators_
    aud_estimators = ensemble_clf
    all_hyp_p_y_h_x = []
    for aud_est in aud_estimators:
        all_hyp_p_y_h_x.append(aud_est.predict_proba(X))
    all_hyp_p_y_h_x = np.array(all_hyp_p_y_h_x)
    
    # Compute various uncertainities given p_z_h_x for all h \in H
    p_y_given_x = np.array(all_hyp_p_y_h_x).mean(axis=0)
    total_uncertainity = compute_total_uncertainity(p_y_given_x,alpha)
    aleatoric_uncertainity = compute_aleatoric_uncertainity(all_hyp_p_y_h_x,alpha)
    epistemic_uncertainity = total_uncertainity - aleatoric_uncertainity
    
    return p_y_given_x, aleatoric_uncertainity, epistemic_uncertainity, total_uncertainity


def entropy(prob):
    return -1 * torch.log2(torch.sum(prob**2, dim=-1))

# def entropy(self, prob):
#     return -1 * torch.sum(prob * torch.log2(prob + self.eps), dim=-1)

def expected_entropy(mc_preds):
    return torch.mean(entropy(mc_preds), dim=1)

def model_uncertainty(mc_preds):
    return entropy(torch.mean(mc_preds, dim=1)) - expected_entropy(mc_preds)



def entropy_shannon(prob):
    return -1 * torch.sum(prob*torch.log2(prob), dim=-1)

# def entropy(self, prob):
#     return -1 * torch.sum(prob * torch.log2(prob + self.eps), dim=-1)

def expected_entropy_shannon(mc_preds):
    return torch.mean(entropy_shannon(mc_preds), dim=1)

def model_uncertainty_shannon(mc_preds):
    return entropy_shannon(torch.mean(mc_preds, dim=1)) - expected_entropy_shannon(mc_preds)

