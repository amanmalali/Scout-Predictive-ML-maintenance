import numpy as np
from torchvision.models import resnet101
from torchvision.models import ResNet101_Weights
import torch
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights

def create_loss_labels(predictions,losses,labels,problem_type,val=False,train_losses=None):
    # if problem_type=='reg':
    if val:
        if train_losses is None:
            raise Exception("To generate val losses, train loss must be provided")
        loss_avg=train_losses.mean()
        loss_std=train_losses.std()
    else:
        loss_avg=losses.mean()
        loss_std=losses.std()
    
    loss_labels=[]
    for i in range(len(losses)):
        if problem_type=='reg':
            if losses[i]>loss_avg+2*loss_std:
                loss_labels.append(1)
            else:
                loss_labels.append(0)
        elif problem_type=='class':
            if predictions[i].round()!=labels[i]:
                loss_labels.append(1)
            else:
                loss_labels.append(0)
        else:
            if losses[i]<loss_avg+1*loss_std:
                loss_labels.append(0)
            elif loss_avg+1*loss_std<=losses[i]<loss_avg+2*loss_std:
                loss_labels.append(1)
            elif loss_avg+2*loss_std<=losses[i]<loss_avg+3*loss_std:
                loss_labels.append(2)
            elif loss_avg+3*loss_std<=losses[i]:
                loss_labels.append(3)

            # if predictions[i].round()!=labels[i]:
            #     loss_labels.append(1)
            # else:
            #     loss_labels.append(0)
    
    if len(loss_labels)==len(losses)==len(predictions):
        print("Loss label length verified")
    else:
        raise Exception("Loss label length does not match the number of samples")
    
    return loss_labels


def build_sensitivity_training_set(features,predictions,losses,labels,embeddings=None,problem_type='reg',val=False,train_losses=None):

    print(features.shape)
    print(losses.shape)
    print(labels.shape)
    print(predictions.shape)
    
    if len(predictions)==len(losses)==len(labels)==len(features):
        print("All data lengths match")
    else:
        raise Exception("Data lengths do not match for predictions, losses and labels")
    if val:
        train_y=create_loss_labels(predictions,losses,labels,problem_type,val=True,train_losses=train_losses)
    else:
        train_y=create_loss_labels(predictions,losses,labels,problem_type)
    train_x=np.concatenate([features,predictions],axis=1)
    if embeddings is not None:
        train_x=np.concatenate([train_x,embeddings],axis=1)

    
    return train_x,train_y


def build_sensitivity_training_set_v2(features,predictions,losses,labels,embeddings=None,problem_type='reg',val=False,train_losses=None):

    print(features.shape)
    print(losses.shape)
    print(labels.shape)
    # if len(losses)==len(labels)==len(features):
    #     print("All data lengths match")
    # else:
    #     raise Exception("Data lengths do not match for predictions, losses and labels")
    if val:
        train_y=create_loss_labels(labels,losses,labels,problem_type,val=True,train_losses=train_losses)
    else:
        train_y=create_loss_labels(labels,losses,labels,problem_type)
    if embeddings is not None:
        train_x=np.concatenate([train_x,embeddings],axis=1)
    train_x=np.concatenate([features,predictions],axis=1)
    # train_x=features
    return train_x,train_y


def generate_image_embeddings(dataset,model_name='resnet-18'):
    #iterate through dataset and generate embeddings which will eventually be the features used for sensitivity model training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0].detach()
        return hook
    res_transforms=ResNet101_Weights.IMAGENET1K_V2.transforms()

    dataset.transform=res_transforms

    model=resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
    model.to(device)
    model.eval()
    model.fc.register_forward_hook(get_activation('fc'))
    features=[]

    dataloader=torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=2)
    batch=0
    print("Generating image features ")
    for x,y in dataloader:
        # print("Batch :",batch)
        batch+=1
        x=x.to(device)
        output = model(x)
        layer_output=activation['fc'].cpu().detach().numpy()
        features.extend(layer_output)

    return np.array(features)


def generate_image_embeddings_v2(dataset,model_name='resnet-18'):
    #iterate through dataset and generate embeddings which will eventually be the features used for sensitivity model training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    vit.to(device)
    vit.eval()
    preprocessing = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
    dataset.transform=preprocessing
    features=[]

    dataloader=torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2)
    batch=0
    print("Generating image features ")
    for x,y in dataloader:
        # print("Batch :",batch)
        x=x.to(device)
        feats = vit._process_input(x)
        batch_class_token = vit.class_token.expand(x.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)

        feats = vit.encoder(feats)

        # We're only interested in the representation of the CLS token that we appended at position 0
        feats = feats[:, 0]

        features.extend(feats.cpu().detach().numpy())

    return np.array(features)



def build_sensitivity_training_ddla(features,predictions,losses,labels,embeddings=None,problem_type='reg',val=False,train_losses=None):

    print(features.shape)
    print(losses.shape)
    print(labels.shape)
    print(predictions.shape)
    
    if len(predictions)==len(losses)==len(labels)==len(features):
        print("All data lengths match")
    else:
        raise Exception("Data lengths do not match for predictions, losses and labels")
    if val:
        train_y=create_loss_labels(predictions,losses,labels,problem_type,val=True,train_losses=train_losses)
    else:
        train_y=create_loss_labels(predictions,losses,labels,problem_type)
    train_x=features
    if embeddings is not None:
        train_x=np.concatenate([train_x,embeddings],axis=1)

    return train_x,train_y