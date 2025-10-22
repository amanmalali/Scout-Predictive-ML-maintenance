import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import copy

class linear_model(nn.Module):
    """
    Square-link feature temperature scaler with hard bounds.

        raw(x) = w⋅x + b
        T(x)   = clip( raw(x)**2 , T_min , T_max )
    """
    def __init__(self, in_features: int, K: int = 5,
                 T_min: float = 0.5, T_max: float = 20.0):
        super().__init__()
        assert 0. < T_min < T_max
        self.T_min, self.T_max = T_min, T_max
        self.lin = nn.Linear(in_features, K)
        # nn.init.zeros_(self.lin.weight)
        # nn.init.constant_(self.lin.bias, 0.0)
        self.lin.bias.data.fill_(0.5)

    def forward(self, x):
        t_sq  = self.lin(x).pow(2)                    # square-link
        temps = torch.clamp(t_sq, self.T_min, self.T_max)
        return temps

# class linear_model(nn.Module):                      # 1-to-1 replacement
#     """
#     Feature-based temperature head that outputs one temperature per ensemble
#     member (default 5).  Temperatures are constrained by a sigmoid mapping:

#         T_k(x) = T_min + (T_max - T_min) * sigmoid(w_k^T x + b_k)

#     This keeps optimisation smooth yet never lets T leave [T_min, T_max].
#     """
#     def __init__(self, in_features: int,
#                  K: int = 5,
#                  T_min: float = 0.5,
#                  T_max: float = 7.0):
#         super().__init__()
#         assert 0. < T_min < T_max, "Require 0 < T_min < T_max"
#         self.T_min, self.T_max = T_min, T_max

#         self.lin1 = nn.Linear(in_features, K)

#         # Mild initialisation: sigmoid(0) = 0.5 → T ≈ (T_min + T_max)/2
#         nn.init.zeros_(self.lin1.weight)
#         nn.init.constant_(self.lin1.bias, 0.0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         raw    = self.lin1(x)                                        # (B, K)
#         temps  = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(raw)
#         return temps 


class uncer_loss(nn.Module):

    def __init__(self, beta=1):
        super(uncer_loss, self).__init__()
        self.diff_loss=nn.MSELoss()
        self.eps=1e-10

    def entropy(self,prob):
        return -1 * torch.log2(torch.sum(prob**2, dim=-1))

    # def entropy(self, prob):
    #     return -1 * torch.sum(prob * torch.log2(prob + self.eps), dim=-1)

    def expected_entropy(self,mc_preds):
        return torch.mean(self.entropy(mc_preds), dim=1)

    def model_uncertainty(self,mc_preds):
        return self.entropy(torch.mean(mc_preds, dim=1)) - self.expected_entropy(mc_preds)
    
    def forward(self,logits):
        probs=F.softmax(logits,dim=-1)
        en=self.entropy(torch.mean(probs, dim=1))
        # scaled_en=F.tanh(en)
        # print(scaled_en,labels)
        return en
    
    def true_f1(self, prediction, true_label, uncertainty,
                                optimal_threshold):
        

        #CLASS 0
        n_ac_0 = torch.zeros(
            1, device=true_label.device)  # number of samples accurate and certain
        n_ic_0 = torch.zeros(
            1,
            device=true_label.device)  # number of samples inaccurate and certain
        n_au_0 = torch.zeros(
            1,
            device=true_label.device)  # number of samples accurate and uncertain
        n_iu_0 = torch.zeros(
            1,
            device=true_label.device)  # number of samples inaccurate and uncertain
        
        #CLASS 1
        n_ac_1 = torch.zeros(
            1, device=true_label.device)  # number of samples accurate and certain
        n_ic_1 = torch.zeros(
            1,
            device=true_label.device)  # number of samples inaccurate and certain
        n_au_1 = torch.zeros(
            1,
            device=true_label.device)  # number of samples accurate and uncertain
        n_iu_1 = torch.zeros(
            1,
            device=true_label.device)  # number of samples inaccurate and uncertain
        
        #CLASS 2
        n_ac_2 = torch.zeros(
            1, device=true_label.device)  # number of samples accurate and certain
        n_ic_2 = torch.zeros(
            1,
            device=true_label.device)  # number of samples inaccurate and certain
        n_au_2 = torch.zeros(
            1,
            device=true_label.device)  # number of samples accurate and uncertain
        n_iu_2 = torch.zeros(
            1,
            device=true_label.device)  # number of samples inaccurate and uncertain
        

        #CLASS 3
        n_ac_3 = torch.zeros(
            1, device=true_label.device)  # number of samples accurate and certain
        n_ic_3 = torch.zeros(
            1,
            device=true_label.device)  # number of samples inaccurate and certain
        n_au_3 = torch.zeros(
            1,
            device=true_label.device)  # number of samples accurate and uncertain
        n_iu_3 = torch.zeros(
            1,
            device=true_label.device)  # number of samples inaccurate and uncertain

        avu = torch.ones(1, device=true_label.device)
        avu.requires_grad_(True)
        print(len(true_label))
        for i in range(len(true_label)):
            if (true_label[i].item()==0):
                if ((true_label[i].item() == prediction[i].item())
                        and uncertainty[i].item() <= optimal_threshold):
                    """ accurate and certain """
                    n_ac_0 += 1
                elif ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() > optimal_threshold):
                    """ accurate and uncertain """
                    n_au_0 += 1
                elif ((true_label[i].item() != prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                    """ inaccurate and certain """
                    n_ic_0 += 1
                elif ((true_label[i].item() != prediction[i].item())
                    and uncertainty[i].item() > optimal_threshold):
                    """ inaccurate and uncertain """
                    n_iu_0 += 1

            elif (true_label[i].item()==1):
                if ((true_label[i].item() == prediction[i].item())
                        and uncertainty[i].item() <= optimal_threshold):
                    """ accurate and certain """
                    n_ac_1 += 1
                elif ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() > optimal_threshold):
                    """ accurate and uncertain """
                    n_au_1 += 1
                elif ((true_label[i].item() != prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                    """ inaccurate and certain """
                    n_ic_1 += 1
                elif ((true_label[i].item() != prediction[i].item())
                    and uncertainty[i].item() > optimal_threshold):
                    """ inaccurate and uncertain """
                    n_iu_1 += 1

            elif (true_label[i].item()==2):
                if ((true_label[i].item() == prediction[i].item())
                        and uncertainty[i].item() <= optimal_threshold):
                    """ accurate and certain """
                    n_ac_2 += 1
                elif ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() > optimal_threshold):
                    """ accurate and uncertain """
                    n_au_2 += 1
                elif ((true_label[i].item() != prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                    """ inaccurate and certain """
                    n_ic_2 += 1
                elif ((true_label[i].item() != prediction[i].item())
                    and uncertainty[i].item() > optimal_threshold):
                    """ inaccurate and uncertain """
                    n_iu_2 += 1

            elif (true_label[i].item()==3):
                if ((true_label[i].item() == prediction[i].item())
                        and uncertainty[i].item() <= optimal_threshold):
                    """ accurate and certain """
                    n_ac_3 += 1
                elif ((true_label[i].item() == prediction[i].item())
                    and uncertainty[i].item() > optimal_threshold):
                    """ accurate and uncertain """
                    n_au_3 += 1
                elif ((true_label[i].item() != prediction[i].item())
                    and uncertainty[i].item() <= optimal_threshold):
                    """ inaccurate and certain """
                    n_ic_3 += 1
                elif ((true_label[i].item() != prediction[i].item())
                    and uncertainty[i].item() > optimal_threshold):
                    """ inaccurate and uncertain """
                    n_iu_3 += 1


        prec_0 = (n_ac_0) / (n_ac_0 + n_au_0+self.eps)
        recall_0=(n_ac_0)/ (n_ac_0 + n_ic_0+self.eps)
        f1_0=(2*prec_0*recall_0)/(prec_0+recall_0)
        # avu_0 = (n_ac_0 + n_au_0) / (n_ac_0 + n_au_0 + n_ic_0 + n_iu_0 + self.eps)
        # print("F1 SCORE 0:",f1_0)
        # print("Accuracy 0:",avu_0)

        prec_1 = (n_ac_1) / (n_ac_1 + n_au_1+self.eps)
        recall_1=(n_ac_1)/ (n_ac_1+n_ic_1+self.eps)
        f1_1=(2*prec_1*recall_1)/(prec_1+recall_1+self.eps)
        # avu_1 = (n_ac_1 + n_au_1) / (n_ac_1 + n_au_1 + n_ic_1 + n_iu_1 + self.eps)
        # print("F1 SCORE 1:",f1_1)
        # print("Accuracy 1:",avu_1)

        prec_2 = (n_ac_2) / (n_ac_2 + n_au_2+self.eps)
        recall_2=(n_ac_2)/ (n_ac_2+n_ic_2+self.eps)
        f1_2=(2*prec_2*recall_2)/(prec_2+recall_2+self.eps)
        # avu_2 = (n_ac_2 + n_au_2) / (n_ac_2 + n_au_2 + n_ic_2 + n_iu_2 + self.eps)
        # print("F1 SCORE 2:",f1_2)
        # print("Accuracy 2:",avu_2)

        prec_3 = (n_ac_3) / (n_ac_3 + n_au_3+self.eps)
        recall_3=(n_ac_3)/ (n_ac_3+n_ic_3+self.eps)
        f1_3=(2*prec_3*recall_3)/(prec_3+recall_3+self.eps)

        print("TRUE F1 SCORE 0: ",f1_0)
        print("TRUE F1 SCORE 1: ",f1_1)
        print("TRUE F1 SCORE 2: ",f1_2)
        print("TRUE F1 SCORE 3: ",f1_3)
        print("PREC 3 :",prec_3)
        print("RECALL 3 :",recall_3)

    
    def precision_uncer(self,logits,labels,uncer_threshold):
        
        probs=F.softmax(logits,dim=-1)
        mean_probs=torch.mean(probs,dim=1)

        confidences, predictions = torch.max(mean_probs, 1)

        unc=self.entropy(mean_probs)

        unc_th = torch.tensor(uncer_threshold,device=logits.device)


        #CLASS 0
        n_ac_0 = torch.zeros(
            1, device=logits.device)  # number of samples accurate and certain
        n_ic_0 = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and certain
        n_au_0 = torch.zeros(
            1,
            device=logits.device)  # number of samples accurate and uncertain
        n_iu_0 = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and uncertain
        
        #CLASS 1
        n_ac_1 = torch.zeros(
            1, device=logits.device)  # number of samples accurate and certain
        n_ic_1 = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and certain
        n_au_1 = torch.zeros(
            1,
            device=logits.device)  # number of samples accurate and uncertain
        n_iu_1 = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and uncertain
        
        #CLASS 2
        n_ac_2 = torch.zeros(
            1, device=logits.device)  # number of samples accurate and certain
        n_ic_2 = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and certain
        n_au_2 = torch.zeros(
            1,
            device=logits.device)  # number of samples accurate and uncertain
        n_iu_2 = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and uncertain
        

        #CLASS 3
        n_ac_3 = torch.zeros(
            1, device=logits.device)  # number of samples accurate and certain
        n_ic_3 = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and certain
        n_au_3 = torch.zeros(
            1,
            device=logits.device)  # number of samples accurate and uncertain
        n_iu_3 = torch.zeros(
            1,
            device=logits.device)  # number of samples inaccurate and uncertain

        # avu = torch.ones(1, device=logits.device)
        avu_loss = torch.zeros(1, device=logits.device)




        for i in range(len(labels)):
            if (labels[i].item()==0):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac_0 += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                    and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au_0 += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic_0 += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                    and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu_0 += (1 - confidences[i]) * torch.tanh(unc[i])

            elif (labels[i].item()==1):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac_1 += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                    and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au_1 += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic_1 += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                    and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu_1 += (1 - confidences[i]) * torch.tanh(unc[i])

            elif (labels[i].item()==2):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    
                    n_ac_2 += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                    and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    
                    n_au_2 += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic_2 += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                    and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu_2 += (1 - confidences[i]) * torch.tanh(unc[i])

            elif (labels[i].item()==3):
                if ((labels[i].item() == predictions[i].item())
                        and unc[i].item() <= unc_th.item()):
                    """ accurate and certain """
                    n_ac_3 += confidences[i] * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() == predictions[i].item())
                    and unc[i].item() > unc_th.item()):
                    """ accurate and uncertain """
                    n_au_3 += confidences[i] * torch.tanh(unc[i])
                elif ((labels[i].item() != predictions[i].item())
                    and unc[i].item() <= unc_th.item()):
                    """ inaccurate and certain """
                    n_ic_3 += (1 - confidences[i]) * (1 - torch.tanh(unc[i]))
                elif ((labels[i].item() != predictions[i].item())
                    and unc[i].item() > unc_th.item()):
                    """ inaccurate and uncertain """
                    n_iu_3 += (1 - confidences[i]) * torch.tanh(unc[i])

        prec_0 = (n_ac_0) / (n_ac_0 + n_au_0+self.eps)
        recall_0=(n_ac_0)/ (n_ac_0 + n_ic_0+self.eps)
        f1_0=(2*prec_0*recall_0)/(prec_0+recall_0)
        # avu_0 = (n_ac_0 + n_au_0) / (n_ac_0 + n_au_0 + n_ic_0 + n_iu_0 + self.eps)
        # print("F1 SCORE 0:",f1_0)
        # print("Accuracy 0:",avu_0)

        prec_1 = (n_ac_1) / (n_ac_1 + n_au_1+self.eps)
        recall_1=(n_ac_1)/ (n_ac_1+n_ic_1+self.eps)
        f1_1=(2*prec_1*recall_1)/(prec_1+recall_1+self.eps)
        # avu_1 = (n_ac_1 + n_au_1) / (n_ac_1 + n_au_1 + n_ic_1 + n_iu_1 + self.eps)
        # print("F1 SCORE 1:",f1_1)
        # print("Accuracy 1:",avu_1)

        prec_2 = (n_ac_2) / (n_ac_2 + n_au_2+self.eps)
        recall_2=(n_ac_2)/ (n_ac_2+n_ic_2+self.eps)
        f1_2=(2*prec_2*recall_2)/(prec_2+recall_2+self.eps)
        # avu_2 = (n_ac_2 + n_au_2) / (n_ac_2 + n_au_2 + n_ic_2 + n_iu_2 + self.eps)
        # print("F1 SCORE 2:",f1_2)
        # print("Accuracy 2:",avu_2)

        prec_3 = (n_ac_3) / (n_ac_3 + n_au_3+self.eps)
        recall_3=(n_ac_3)/ (n_ac_3+n_ic_3+self.eps)
        f1_3=(2*prec_3*recall_3)/(prec_3+recall_3+self.eps)
        # avu_3 = (n_ac_3 + n_au_3) / (n_ac_3 + n_au_3 + n_ic_3 + n_iu_3 + self.eps)
        # print("F1 SCORE 3:",f1_3)
        # print("Accuracy 3:",avu_3)
        # p_ac = (n_ac) / (n_ac + n_ic)
        # p_ui = (n_iu) / (n_iu + n_ic)
        #print('Actual AvU: ', self.accuracy_vs_uncertainty(predictions, labels, uncertainty, optimal_threshold))
        # avu_loss = -1 * torch.log((f1_0+f1_1+f1_2+f1_3)/4+self.eps)
        avu_loss = -1 * torch.log((f1_0+f1_1+f1_2+f1_3)/4+self.eps)
        print("F1 SCORE 0: ",f1_0)
        print("F1 SCORE 1: ",f1_1)
        print("F1 SCORE 2: ",f1_2)
        print("F1 SCORE 3: ",f1_3)
        self.true_f1(predictions,labels,unc,unc_th)
        return avu_loss
    

    def uceloss(self,logits, labels, n_bins=10):
        probs=F.softmax(logits,dim=-1)
        mean_probs=torch.mean(probs,dim=1)
        labels=labels.to(torch.long)
        confidences, predictions = torch.max(mean_probs, 1)
        
        uncertainties=self.entropy(mean_probs)
        # uncertainties=self.model_uncertainty(probs)

        d = logits.device
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        _, predictions = torch.max(mean_probs, 1)
        errors = predictions.ne(labels)
        # uncertainties = nentr(softmaxes, base=softmaxes.size(1))
        errors_in_bin_list = []
        avg_entropy_in_bin_list = []

        uce = torch.zeros(1, device=d)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculate |uncert - err| in each bin
            in_bin = uncertainties.gt(bin_lower.item()) * uncertainties.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()  # |Bm| / n
            if prop_in_bin.item() > 0.0:
                errors_in_bin = errors[in_bin].float().mean()  # err()
                avg_entropy_in_bin = uncertainties[in_bin].mean()  # uncert()
                uce += torch.abs(avg_entropy_in_bin - errors_in_bin) * prop_in_bin

                errors_in_bin_list.append(errors_in_bin)
                avg_entropy_in_bin_list.append(avg_entropy_in_bin)

        err_in_bin = torch.tensor(errors_in_bin_list, device=d)
        avg_entropy_in_bin = torch.tensor(avg_entropy_in_bin_list, device=d)

        return uce, err_in_bin, avg_entropy_in_bin


class scaler_model(nn.Module):
    def __init__(self,models=5):
        super(scaler_model,self).__init__()
        self.temp=nn.Parameter(torch.ones(models)*1.5)

    def forward(self,logits):
        temps=self.temp.unsqueeze(1).expand(logits.size(0),logits.size(1),logits.size(2))
        return logits/temps

    # def forward_ext_temp(self,logits,temp):
    #     # print("TEMPS :",logits)
    #     # temps=temp.unsqueeze(2).expand(logits.size(0),logits.size(1),logits.size(2))
    #     # print(logits.shape)
    #     # print(temp.shape)
    #     return logits/temp
    
    def forward_ext_temp(self,logits,temp):
        # print("TEMPS :",logits)
        temps=temp.unsqueeze(2).expand(logits.size(0),logits.size(1),logits.size(2))
        # print(logits.shape)
        # print(temp.shape)
        return logits/temps

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _inv_sigmoid_T(T_eff: float, T_min: float, T_max: float) -> float:
    """Return z s.t.  T_eff = T_min + (T_max-T_min)*sigmoid(z)"""
    # clip to open interval to avoid inf logits at the ends
    frac = (T_eff - T_min) / (T_max - T_min)
    frac = min(max(frac, 1e-6), 1 - 1e-6)
    return math.log(frac / (1.0 - frac))


# --------------------------------------------------------------------------- #
class GlobalScaler(nn.Module):
    """Single scalar temperature for all ensemble members, smooth & bounded."""
    def __init__(self,
                 init_temp: float = 1.5,
                 T_min: float = 0.5,
                 T_max: float = 15.0):
        super().__init__()
        assert 0 < T_min < T_max
        self.T_min, self.T_max = T_min, T_max

        # store *raw* parameter so optimiser works in ℝ
        raw_init = _inv_sigmoid_T(init_temp, T_min, T_max)
        self.raw_temp = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T_eff = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.raw_temp)
        return logits / T_eff


class EnsembleScaler(nn.Module):
    """Static per-member temperatures (default 5) with smooth bounds."""
    def __init__(self,
                 n_models: int = 5,
                 init_temp: float = 1.5,
                 T_min: float = 0.5,
                 T_max: float = 15.0):
        super().__init__()
        assert 0 < T_min < T_max
        self.T_min, self.T_max = T_min, T_max

        raw_init = _inv_sigmoid_T(init_temp, T_min, T_max)
        self.raw_temp = nn.Parameter(
            torch.full((n_models,), raw_init, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits shape: (B, n_models, n_classes)
        T_eff = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(self.raw_temp)
        t = T_eff.view(1, -1, 1).expand_as(logits)  # broadcast over batch & classes
        return logits / t

def train_scaler(data, epochs: int = 100, feat: bool = True, lr: float = 1e-3):
    """
    Trains either
      • a feature-based temperature head (`linear_model`) + frozen scaler  OR
      • a global / per-member static scaler

    Returns
    -------
    (lin_model, scaler)        # lin_model is None if feat == False
    """
    dataloader = DataLoader(data, batch_size=64, shuffle=True)
    scaler     = scaler_model()                     # GlobalScaler / EnsembleScaler
    loss_f     = BrierEntropyLoss()                       # your UCE wrapper

    # ------------------------------------------------------------------ #
    #  1)  FEATURE-BASED head  (FBTS)
    # ------------------------------------------------------------------ #
    if feat:
        lin_model = linear_model(data.X.shape[1])   # ← already bounded via sigmoid
        optimizer = optim.Adam(lin_model.parameters(), lr=lr)

        best_state, best_loss = None, float("inf")

        scaler.eval()                               # frozen during FBTS training
        for e in range(epochs):
            lin_model.train()
            epoch_loss = 0.0

            for x, logits, y in dataloader:
                optimizer.zero_grad()

                temps   = lin_model(x)                              # (B, K)
                preds   = scaler.forward_ext_temp(logits, temps)    # user API
                loss= loss_f(preds, y)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if 0 < epoch_loss < best_loss:
                best_loss  = epoch_loss
                best_state = copy.deepcopy(lin_model.state_dict())

            print(f"EPOCH {e:03d} – total loss {epoch_loss:.4f}")

        if best_state is None:        # training diverged or all-zero loss
            return None, None

        lin_model.load_state_dict(best_state)
        print("Best FBTS loss :", best_loss)
        return lin_model, scaler

    # ------------------------------------------------------------------ #
    #  2)  STATIC global / per-member scaler
    # ------------------------------------------------------------------ #
    else:
        # optimiser must see the *raw* unconstrained parameter
        optimizer = optim.Adam([scaler.raw_temp], lr=lr)

        best_state, best_loss = None, float("inf")

        for e in range(epochs):
            scaler.train()
            for _, logits, y in dataloader:
                optimizer.zero_grad()
                loss = loss_f(scaler(logits), y)
                loss.backward()
                optimizer.step()

            # full-set evaluation
            scaler.eval()
            with torch.no_grad():
                logits_all = torch.tensor(data.logits, dtype=torch.float32)
                y_all      = torch.tensor(data.y)
                full_loss= loss_f(scaler(logits_all), y_all)

            if full_loss.item() < best_loss:
                best_loss  = full_loss.item()
                best_state = copy.deepcopy(scaler.state_dict())

            print(f"EPOCH {e:03d} – loss {full_loss.item():.4f}")

        scaler.load_state_dict(best_state)

        # pretty-print effective temperature(s)
        with torch.no_grad():
            T_eff = scaler.T_min + (scaler.T_max - scaler.T_min) * \
                    torch.sigmoid(scaler.raw_temp.data)
        print("Optimal effective temperature(s):", T_eff)

        return None, scaler



def eval_temp_scaler(data,scaler,lin_model=None,feat=True,uncer_thresh=0.4):
    if feat and lin_model is None:
        raise Exception("Feat cannot be true without linear model for evaluating the temp scaler")

    loss_func=uncer_loss()
    scaler.eval()
    if lin_model is not None and feat:
        print("Performance with feature based temperature scaling")
        lin_model.eval()
        new_temps=lin_model(torch.tensor(data.X,dtype=torch.float32))
        _=loss_func.precision_uncer(scaler.forward_ext_temp(torch.tensor(data.logits,dtype=torch.float32),new_temps),torch.tensor(data.y),uncer_thresh)
    else:
        print("Performance with temperature scaling")
        _=loss_func.precision_uncer(scaler(torch.tensor(data.logits)),torch.tensor(data.y),uncer_thresh)

    print("Performance without any scaling")
    _=loss_func.precision_uncer(torch.tensor(data.logits,dtype=torch.float32),torch.tensor(data.y),uncer_thresh)



def _train_static_scaler(
    data,                    # custom_torch_dataset
    scaler: nn.Module,       # new GlobalScaler or EnsembleScaler
    epochs: int = 1000,
    lr: float = 1e-3,
):
    loader   = DataLoader(data, batch_size=512, shuffle=True)
    # ---------------- optimiser on the *raw* parameter --------------------
    opt      = optim.Adam([scaler.raw_temp], lr=lr)
    loss_f   = BrierEntropyLoss()                        # your UCE wrapper
    best_state, best_loss = None, float("inf")

    for epoch in range(epochs):
        scaler.train()
        for _, logits, labels in loader:
            opt.zero_grad()
            loss = loss_f(scaler(logits), labels)
            loss.backward()
            opt.step()                # <- no clamp needed

        # ---------- full-set UCE on hold-out / full dataset --------------
        scaler.eval()
        with torch.no_grad():
            full_loss = loss_f(
                scaler(torch.tensor(data.logits, dtype=torch.float32)),
                torch.tensor(data.y),
            )
        if full_loss.item() < best_loss:
            best_loss  = full_loss.item()
            best_state = copy.deepcopy(scaler.state_dict())

    scaler.load_state_dict(best_state)

    # ------------ pretty print effective temperatures --------------------
    with torch.no_grad():
        T_eff = scaler.T_min + (scaler.T_max - scaler.T_min) * \
                torch.sigmoid(scaler.raw_temp.data)
    print("Optimal effective temperature(s):", T_eff)

    return scaler


# def train_temp_scaler(data,eval_data,temp_scale="feat"):
#     scaler=None
#     lin_model=None
#     if temp_scale=="feat":
#         lin_model,scaler=train_scaler(data)

#     elif temp_scale=="temp":
#         _,scaler=train_scaler(data,feat=False)

#     if eval_data is not None:
#         eval_temp_scaler(eval_data,scaler,lin_model)
    
#     return lin_model,scaler


def train_temp_scaler(data, eval_data=None, mode: str = "feat"):
    """
    mode ∈ {"feat", "global", "ensemble"}
    """
    print("Training temperature scaler")
    if mode == "feat":
        lin_model, scaler = train_scaler(data)              # original path
    elif mode == "global":
        lin_model = None
        scaler    = _train_static_scaler(data, GlobalScaler())
    elif mode == "ensemble":
        lin_model = None
        scaler    = _train_static_scaler(data, EnsembleScaler())
    else:
        raise ValueError("mode must be 'feat', 'global', or 'ensemble'")

    if eval_data is not None and lin_model is not None and scaler is not None:
        eval_temp_scaler(eval_data, scaler, lin_model, feat=(mode == "feat"))
    return lin_model, scaler


class BrierEntropyLoss(nn.Module):
    """
    Brier score loss pairing collision entropy with the true-error indicator.
    """
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def entropy(self, prob: torch.Tensor) -> torch.Tensor:
        # Collision entropy H2(p) = -log2(sum_i p_i^2)
        return -torch.log2(torch.sum(prob**2, dim=-1) + self.eps)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (B, K, C) after temperature scaling
            labels: LongTensor of shape (B,)
        Returns:
            Brier score loss: MSE between normalized entropy and 0/1 error
        """
        # 1) Ensemble mean probabilities across K members
        probs = F.softmax(logits, dim=-1)             # (B, K, C)
        mean_probs = probs.mean(dim=1)                # (B, C)

        # 2) Collision entropy, normalized to [0,1] for binary (log2(2)=1)
        H2 = self.entropy(mean_probs)                 # (B,)
        H2_norm = H2 / math.log2(mean_probs.size(-1))

        # 3) Predicted error probability = normalized entropy
        p_err = H2_norm                               # (B,)

        # 4) True error indicator: 1 if misclassified, else 0
        preds = torch.argmax(mean_probs, dim=1)
        true_err = (preds != labels).float()          # (B,)

        # 5) Brier score (mean squared error)
        loss = F.mse_loss(p_err, true_err)
        return loss