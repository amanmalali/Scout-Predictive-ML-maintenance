import numpy as np
from torchvision.models import resnet101
from torchvision.models import ResNet101_Weights
import torch
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights

def create_loss_labels(predictions,losses,labels,problem_type,val=False,train_losses=None):
    """Create binary/multiclass “high-loss” labels from per-sample losses.

    Args:
        predictions: Model predictions (used only when problem_type == "class").
        losses: Per-sample loss values, shape (N,).
        labels: Ground-truth labels/targets, shape (N,).
        problem_type: "reg" -> binary outlier loss label; "class" -> prediction mismatch label;
            anything else -> 4-level loss binning by z-score thresholds.
        val: If True, compute loss mean/std from `train_losses` instead of `losses`.
        train_losses: Training losses used for normalization when `val=True`.

    Returns:
        loss_labels: List[int] of length N.
    """
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
            predictions = (
                np.argmax(predictions, axis=1).astype(np.float32)
                if predictions.ndim == 2 else predictions
            )
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


def build_metamodel_training_set(features,predictions,losses,labels,embeddings=None,problem_type='reg',val=False,train_losses=None):
    """Build (X, y) for a sensitivity model.

    X is formed by concatenating feature vectors (optionally with `embeddings`) and then `predictions`.
    y is produced by `create_loss_labels(...)`.

    Args:
        features: Base features, shape (N, D).
        predictions: Per-sample predictions, shape (N, P).
        losses: Per-sample losses, shape (N,).
        labels: Ground-truth labels/targets, shape (N,).
        embeddings: Optional extra features, shape (N, E).
        problem_type: use "reg" for binary labels on continuous loss function
                      use "class" for binary classification label mismatch
        val: If True, compute loss mean/std from `train_losses` instead of `losses`.
        train_losses: Training losses used for normalization when `val=True`.
    Returns:
        train_x: np.ndarray, shape (N, D+E+P) (E included only if embeddings is not None).
        train_y: List[int] of length N.
    """
    print("Feature shape: ",features.shape)
    print("Losses shape: ",losses.shape)
    print("Label shape: ",labels.shape)
    print("Predictions shape: ",predictions.shape)
    
    if len(predictions)==len(losses)==len(labels)==len(features):
        print("All data lengths match")
    else:
        raise Exception("Data lengths do not match for predictions, losses and labels")
    if val:
        train_y=create_loss_labels(predictions,losses,labels,problem_type,val=True,train_losses=train_losses)
    else:
        train_y=create_loss_labels(predictions,losses,labels,problem_type)
    if embeddings is not None:
        train_x=np.concatenate([features,embeddings],axis=1)
        train_x=np.column_stack((train_x,predictions))
    else:
        train_x=np.column_stack((features,predictions))
    
    return np.array(train_x),np.array(train_y)



def generate_image_embeddings(dataset):
    """Compute image embeddings for an entire dataset using a pretrained ResNet-101.

    Notes:
        - Applies the ResNet101 IMAGENET1K_V2 preprocessing transforms to `dataset.transform`.
        - Returns the input activation to the final `fc` layer (a 2048-d embedding per image).

    Args:
        dataset: torchvision-style dataset returning (image, label).
        model_name: Unused (kept for API compatibility).

    Returns:
        features: np.ndarray of shape (N, 2048).
    """
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


def generate_image_embeddings_v2(dataset):
    """
    Compute image embeddings for an entire dataset using a pretrained ViT-B/16.

    Notes:
        - Applies the ViT IMAGENET1K_SWAG_E2E_V1 preprocessing transforms to `dataset.transform`.
        - Returns the final CLS-token representation (one embedding per image).

    Args:
        dataset: torchvision-style dataset returning (image, label).

    Returns:
        features: np.ndarray of shape (N, D_vit) (CLS embedding dimension for ViT-B/16).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    vit.to(device)
    vit.eval()
    
    preprocessing = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
    dataset.transform = preprocessing
    
    features = []

    # Increased batch size and added pin_memory for faster CPU->GPU transfer
    batch_size = 128 if device == 'cuda' else 32
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=(device == 'cuda')
    )
    
    batch = 0
    print("Generating image features...")
    
    # Context manager to disable gradient calculation (massive speedup & memory savings)
    with torch.no_grad():
        for x, y in dataloader:
            # non_blocking=True allows asynchronous data transfers to GPU
            x = x.to(device, non_blocking=True)
            
            feats = vit._process_input(x)
            batch_class_token = vit.class_token.expand(x.shape[0], -1, -1)
            feats = torch.cat([batch_class_token, feats], dim=1)

            feats = vit.encoder(feats)

            # We're only interested in the representation of the CLS token that we appended at position 0
            feats = feats[:, 0]

            # Append the whole batch block instead of row-by-row extend
            features.append(feats.cpu().numpy())
            
            batch += 1

    # Concatenate all batches at once at the end
    return np.concatenate(features, axis=0)


"""
LEGACY CODE

def generate_image_embeddings_v2(dataset):

    Compute image embeddings for an entire dataset using a pretrained ViT-B/16.

    Notes:
        - Applies the ViT IMAGENET1K_SWAG_E2E_V1 preprocessing transforms to `dataset.transform`.
        - Returns the final CLS-token representation (one embedding per image).

    Args:
        dataset: torchvision-style dataset returning (image, label).
        model_name: Unused (kept for API compatibility).

    Returns:
        features: np.ndarray of shape (N, D_vit) (CLS embedding dimension for ViT-B/16).
    
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


def build_sensitivity_training_set_v2(features,predictions,losses,labels,embeddings=None,problem_type='reg',val=False,train_losses=None):

    print(features.shape)
    print(losses.shape)
    print(labels.shape)
    # if len(losses)==len(labels)==len(features):
    #     print("All data lengths match")
    # else:
    #     raise Exception("Data lengths do not match for predictions, losses and labels")
    if val:
        train_y=create_loss_labels(predictions,losses,labels,problem_type,val=True,train_losses=train_losses)
    else:
        train_y=create_loss_labels(predictions,losses,labels,problem_type)
    if embeddings is not None:
        train_x=np.concatenate([train_x,embeddings],axis=1)
    train_x=np.concatenate([features,predictions],axis=1)
    # train_x=features
    return train_x,train_y


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
"""