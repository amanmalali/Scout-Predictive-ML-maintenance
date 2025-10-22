import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import copy



class linear_model(nn.Module):
    def __init__(self,in_features):
        super(linear_model,self).__init__()
        self.lin1=nn.Linear(in_features,5)
        self.eps=1e-10
        self.lin1.bias.data.fill_(0.5)

    def forward(self,x):
        temps=self.lin1(x)
        temps=temps**2
        return temps


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

class GlobalScaler(nn.Module):
    """Single scalar temperature for all ensemble members."""
    def __init__(self, init_temp: float = 1.5):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(init_temp))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temp


class EnsembleScaler(nn.Module):
    """Static temperature per ensemble member (5 by default)."""
    def __init__(self, n_models: int = 5, init_temp: float = 1.5):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(n_models) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits expected shape: (B, n_models, n_classes)
        t = self.temp.view(1, -1, 1).expand_as(logits)
        return logits / t

def train_scaler(data,epochs=100,feat=True,lr=0.001):
    dataloader=DataLoader(data,batch_size=500,shuffle=True)
    scaler=scaler_model()

    if feat:
        lin_model=linear_model(data.X.shape[1])
        optimizer=optim.Adam(lin_model.parameters(),lr=lr)
        loss_func=uncer_loss()
        scaler.eval()
        bad_train=False

        best_model=None
        best_loss=100
        scaler.eval()
        for e in range(epochs):
            total_loss=0
            print("EPOCH : ",e)
            for x,logits,y in dataloader:
                lin_model.train()
                new_temps=lin_model(x)
                optimizer.zero_grad()
                loss,_,_=loss_func.uceloss(scaler.forward_ext_temp(logits,new_temps),y)
                total_loss+=loss.item()
                if not loss.requires_grad or loss.item()==0:
                    continue
                loss.backward()
                optimizer.step()
            if total_loss<best_loss and total_loss>0:
                best_loss=total_loss
                best_model=copy.deepcopy(lin_model.state_dict())
            if total_loss==0:
                bad_train=True
                continue
            print("Total Loss :",total_loss)
        
        print("Best Loss :",best_loss)
        lin_model.load_state_dict(best_model)
        if bad_train:
            lin_model=None
            scaler=None
        return lin_model,scaler

    else:
        optimizer=optim.Adam([scaler.temp],lr=lr)
        loss_func=uncer_loss()

        epochs=100
        best_loss=100
        best_model=None
        for e in range(epochs):
            print("EPOCH : ",e)
            for x,y in dataloader:
                scaler.train()
                optimizer.zero_grad()
                uce_loss,_,_=loss_func.uceloss(scaler(x),y)
                uce_loss.backward()
                optimizer.step()
            scaler.eval()
            uce_loss,_,_=loss_func.uceloss(scaler(torch.tensor(data.logits)),torch.tensor(data.y))
            if uce_loss.item()<best_loss:
                best_loss=uce_loss.item()
                best_model=copy.deepcopy(scaler.state_dict())
            print("Total Loss :",uce_loss.item())

        print("Best Loss :",best_loss)
        scaler.load_state_dict(best_model)
        return None,scaler


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
    data,                             # custom_torch_dataset
    scaler: nn.Module,                # GlobalScaler or EnsembleScaler
    epochs: int = 1000,
    lr: float = 1e-3):
    loader = DataLoader(data, batch_size=512, shuffle=True)
    opt    = optim.Adam([scaler.temp], lr=lr)
    loss_f = uncer_loss()
    best_state, best_loss = None, float("inf")

    for epoch in range(epochs):
        scaler.train()
        for _, logits, labels in loader:
            opt.zero_grad()
            loss, *_ = loss_f.uceloss(scaler(logits), labels)
            loss.backward()
            opt.step()

        # track full-set UCE
        scaler.eval()
        with torch.no_grad():
            full_loss, *_ = loss_f.uceloss(
                scaler(torch.tensor(data.logits, dtype=torch.float32)),
                torch.tensor(data.y),
            )
        if full_loss.item() < best_loss:
            best_loss = full_loss.item()
            best_state = copy.deepcopy(scaler.state_dict())
    print("Found optimal temperature :",scaler.temp)
    scaler.load_state_dict(best_state)
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




# logits=np.load("/Users/aragh/project/predictive_maintenance/temp_data/logits.npy")
# labels=np.load("/Users/aragh/project/predictive_maintenance/temp_data/calib_y.npy")

# train_x=np.load('/Users/aragh/project/predictive_maintenance/temp_data/calib_x.npy')


# val_x=np.load('/Users/aragh/project/predictive_maintenance/data/sense_census_val_x.npy')

# val_logits=np.load("/Users/aragh/project/predictive_maintenance/temp_data/val_logits.npy")
# val_labels=np.load("/Users/aragh/project/predictive_maintenance/temp_data/val_y.npy")



# data=custom_torch_dataset(train_x,logits,labels)
# dataloader=DataLoader(data,batch_size=500,shuffle=True)
# scaler=scaler_model()
# lin_model=linear_model(train_x.shape[1])
# optimizer=optim.Adam(lin_model.parameters(),lr=0.001)
# loss_func=uncer_loss()


# epochs=100
# scaler.eval()

# best_model=None
# best_loss=100
# scaler.eval()
# for e in range(epochs):
#     total_loss=0
#     print("EPOCH : ",e)
#     for x,logits,y in dataloader:
#         lin_model.train()
#         new_temps=lin_model(x)
#         optimizer.zero_grad()
#         # loss=acc_loss.forward(scaler(x),y,0.3)
#         loss,_,_=loss_func.uceloss(scaler.forward_ext_temp(logits,new_temps),y)
#         # loss=loss_func.precision_uncer(scaler.forward_ext_temp(logits,new_temps),y,0.4)
#         loss.backward()
#         total_loss+=loss.item()
#         optimizer.step()
#     # print(scaler.temp)
#     # loss=loss_func.precision_uncer(scaler(torch.tensor(logits)),torch.tensor(labels),0.4)
#     # loss=acc_loss.forward(scaler(torch.tensor(logits)),torch.tensor(labels),0.3)
#     if total_loss<best_loss:
#         best_loss=total_loss
#         best_model=copy.deepcopy(lin_model.state_dict())
#     print("Total_loss= ",total_loss)

# print("BEST LOSS :",best_loss)
# print("FOR VALIDATION DATA")

# lin_model.load_state_dict(best_model)
# lin_model.eval()
# scaler.eval()
# new_temps=lin_model(torch.tensor(val_x,dtype=torch.float32))

# print(new_temps)
# loss=loss_func.precision_uncer(scaler.forward_ext_temp(torch.tensor(val_logits,dtype=torch.float32),new_temps),torch.tensor(val_labels),0.4)
# # loss=loss_func.precision_uncer(scaler(torch.tensor(val_logits)),torch.tensor(val_labels),0.4)

# print("FOR VALIDATION WITHOUT SCALING")
# # loss=loss_func.precision_uncer(torch.tensor(val_logits),torch.tensor(val_labels),0.4)
# loss=loss_func.precision_uncer(torch.tensor(val_logits,dtype=torch.float32),torch.tensor(val_labels),0.4)




# # scaled_val_logits=scaler(torch.tensor(val_logits))
# # probs=F.softmax(scaled_val_logits,dim=-1).detach().numpy()

# # np.save("/Users/aragh/project/predictive_maintenance/temp_data/new_val_probs.npy",probs)
