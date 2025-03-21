import argparse
import time
import os
import yaml
import pathlib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import nibabel as nib
import numpy as np
import lpips

from model import MLPv1, MLPv2, Siren, WireReal
from dataset import MultiModalDataset, InferDataset
from visualization_utils import show_slices_gt
from sklearn.preprocessing import MinMaxScaler
from utils import input_mapping, compute_metrics, dict2obj, get_string, compute_mi_hist, compute_mi
from loss_functions import MILossGaussian, NMI, NCC

def parse_args():
    parser = argparse.ArgumentParser(description='Train Neural Implicit Function for a single scan.')
    parser.add_argument('--config', default='config/config.yaml', help='config file (.yaml) containing the hyper-parameters for training.')
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0], help="GPU ID following PCI order.")

    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)

    # patient
    parser.add_argument('--subject_id', type=str, default=None)
    parser.add_argument('--experiment_no', type=int, default=None)
    return parser.parse_args()


def main(args):

    # Init arguments 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    # Load the config 
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config_dict)

    # we bypass lr, epoch and batch_size if we provide them via arparse
    if args.lr != None:
        config.TRAINING.LR = args.lr
        config_dict["TRAINING"]["LR"] = args.lr
    
    if args.batch_size != None:
        config.TRAINING.BATCH_SIZE = args.batch_size
        config_dict["TRAINING"]["BATCH_SIZE"] = args.batch_size
    
    if args.epochs != None:
        config.TRAINING.EPOCHS = args.epochs
        config_dict["TRAINING"]["EPOCHS"] = args.epochs

    # dataset specific
    if args.subject_id != None:
        config.DATASET.SUBJECT_ID = args.subject_id
        config_dict["DATASET"]["SUBJECT_ID"] = args.subject_id

    # experiment type
    if args.experiment_no == 1:
        # t1 / FLAIR
        config.DATASET.LR_CONTRAST1= 't1_LR' 
        config.DATASET.LR_CONTRAST2= 'flair_LR'
        config_dict["DATASET"]["LR_CONTRAST1"] = config.DATASET.LR_CONTRAST1
        config_dict["DATASET"]["LR_CONTRAST2"] = config.DATASET.LR_CONTRAST2

    elif args.experiment_no == 2:
        # DIR / FLAIR
        config.DATASET.LR_CONTRAST1= 'dir_LR' 
        config.DATASET.LR_CONTRAST2= 'flair_LR'
        config_dict["DATASET"]["LR_CONTRAST1"] = config.DATASET.LR_CONTRAST1
        config_dict["DATASET"]["LR_CONTRAST2"] = config.DATASET.LR_CONTRAST2

    elif args.experiment_no == 3:
        # T1w / T2w
        config.DATASET.LR_CONTRAST1= 't1_LR' 
        config.DATASET.LR_CONTRAST2= 't2_LR'
        config_dict["DATASET"]["LR_CONTRAST1"] = config.DATASET.LR_CONTRAST1
        config_dict["DATASET"]["LR_CONTRAST2"] = config.DATASET.LR_CONTRAST2
    
    else:
        # use the settings in config.yaml instead
        if args.experiment_no != None:
            raise ValueError("Experiment not defined.")


    # logging run
    if args.logging:
        wandb.login()
        wandb.init(config=config_dict, project=config.SETTINGS.PROJECT_NAME)

    # make directory for models
    weight_dir = f'runs/{config.SETTINGS.PROJECT_NAME}_weights'
    image_dir = f'runs/{config.SETTINGS.PROJECT_NAME}_images'

    pathlib.Path(weight_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)

    # seeding
    torch.manual_seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)
    
    device = f'cuda:{config.SETTINGS.GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load dataset
    dataset = MultiModalDataset(
                    image_dir = config.SETTINGS.DIRECTORY,
                    name = config.SETTINGS.PROJECT_NAME,
                    subject_id=config.DATASET.SUBJECT_ID,
                    contrast1_LR_str=config.DATASET.LR_CONTRAST1,
                    contrast2_LR_str=config.DATASET.LR_CONTRAST2, 
                    )


    # Model Selection
    model_name = (
                f'{config.SETTINGS.PROJECT_NAME}_subid-{config.DATASET.SUBJECT_ID}_'
                f'ct1LR-{config.DATASET.LR_CONTRAST1}_ct2LR-{config.DATASET.LR_CONTRAST2}_'
                f's_{config.TRAINING.SEED}_shuf_{config.TRAINING.SHUFFELING}_'
    )


    # output_size
    if config.TRAINING.CONTRAST1_ONLY or config.TRAINING.CONTRAST2_ONLY:
        output_size = 1
        if config.TRAINING.CONTRAST1_ONLY:
            model_name = f'{model_name}_CT1_ONLY_'
        else:
            model_name = f'{model_name}_CT2_ONLY_'

    else:
        output_size = 2

    # Embeddings
    if config.MODEL.USE_FF:
        mapping_size = config.FOURIER.MAPPING_SIZE  # of FF
        input_size = 2* mapping_size
        B_gauss = torch.tensor(np.random.normal(scale=config.FOURIER.FF_SCALE, size=(config.FOURIER.MAPPING_SIZE, 3)), dtype=torch.float32).to(device)
        input_mapper = input_mapping(B=B_gauss, factor=config.FOURIER.FF_FACTOR).to(device)
        model_name = f'{model_name}_FF_{get_string(config_dict["FOURIER"])}_'

    else:
        input_size = 3

    # Model Selection
    if config.MODEL.USE_SIREN:
        model = Siren(in_features=input_size, out_features=output_size, hidden_features=config.MODEL.HIDDEN_CHANNELS,
                    hidden_layers=config.MODEL.NUM_LAYERS, first_omega_0=config.SIREN.FIRST_OMEGA_0, hidden_omega_0=config.SIREN.HIDDEN_OMEGA_0)   # no dropout implemented
        model_name = f'{model_name}_SIREN_{get_string(config_dict["SIREN"])}_'
    elif config.MODEL.USE_WIRE_REAL:
        model = WireReal(in_features=input_size, out_features=output_size, hidden_features=config.MODEL.HIDDEN_CHANNELS,
                    hidden_layers=config.MODEL.NUM_LAYERS, 
                    first_omega_0=config.WIRE.WIRE_REAL_FIRST_OMEGA_0, hidden_omega_0=config.WIRE.WIRE_REAL_HIDDEN_OMEGA_0,
                    first_s_0=config.WIRE.WIRE_REAL_FIRST_S_0, hidden_s_0=config.WIRE.WIRE_REAL_HIDDEN_S_0
                    )
        model_name = f'{model_name}_WIRE_{get_string(config_dict["WIRE"])}_'   
    
    else:
        if config.MODEL.USE_TWO_HEADS:
            if (config.TRAINING.CONTRAST1_ONLY or config.TRAINING.CONTRAST2_ONLY) == True:
                raise ValueError('Do not use MLPv2 for single contrast.')

            model = MLPv2(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                        num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
            model_name = f'{model_name}_MLP2_'
        else:
            model = MLPv1(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                        num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
            model_name = f'{model_name}_MLP2_'

    model.to(device)

    print(f'Number of MLP parameters {sum(p.numel() for p in model.parameters())}')

    # model for lpips metric
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    model_name = f'{model_name}_NUML_{config.MODEL.NUM_LAYERS}_N_{config.MODEL.HIDDEN_CHANNELS}_D_{config.MODEL.DROPOUT}_'     

    if config.TRAINING.LOSS == 'L1Loss':
        criterion = nn.L1Loss()
    elif config.TRAINING.LOSS == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Loss function not defined!')

    model_name = f'{model_name}_{config.TRAINING.LOSS}__{config.TRAINING.LOSS_MSE_C1}__{config.TRAINING.LOSS_MSE_C2}_'     

    # custom losses in addition to normal loss
    if config.TRAINING.USE_MI:
        mi_criterion = MILossGaussian(num_bins=config.MI_CC.MI_NUM_BINS, sample_ratio=config.MI_CC.MI_SAMPLE_RATIO, gt_val=config.MI_CC.GT_VAL)
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'     
    
    if config.TRAINING.USE_CC:
        cc_criterion = NCC()
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'    
        
    if config.TRAINING.USE_NMI:
        mi_criterion = NMI(intensity_range=(0,1), nbins=config.MI_CC.MI_NUM_BINS, sigma=config.MI_CC.NMI_SIGMA)
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'    

    # optimizer
    if config.TRAINING.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR)#, weight_decay=5e-5)
        model_name = f'{model_name}_{config.TRAINING.OPTIM}_{config.TRAINING.LR}_'    
    else:
        raise ValueError('Optim not defined!')
    
    # image sizes
    [x_dim, y_dim, z_dim] = [176,224,256]

    for epoch in range(config.TRAINING.EPOCHS):
        # set model to train
        model.train()

        model_name_epoch = f'{model_name}_e{int(epoch)}_model.pt'  
        model_path = os.path.join(weight_dir, model_name_epoch)

        print(model_path)
        scaler = MinMaxScaler()

        ################ EVALUATION #######################

        if not config.TRAINING.CONTRAST1_ONLY and not config.TRAINING.CONTRAST2_ONLY:
            print("Generating NIFTIs.")
            gt_contrast1 = dataset.get_contrast1_gt()
            gt_contrast2 = dataset.get_contrast2_gt()

            gt_contrast1 = gt_contrast1.reshape((x_dim, y_dim, z_dim)).cpu().numpy()
            gt_contrast2 = gt_contrast2.reshape((x_dim, y_dim, z_dim)).cpu().numpy()

            label_arr = np.array(gt_contrast1, dtype=np.float32)
            gt_contrast1= scaler.fit_transform(gt_contrast1.reshape(-1, 1)).reshape((x_dim, y_dim, z_dim))

            label_arr = np.array(gt_contrast2, dtype=np.float32)
            gt_contrast2= scaler.fit_transform(gt_contrast2.reshape(-1, 1)).reshape((x_dim, y_dim, z_dim))

        else:
            print("Generating NIFTIs.")
            gt_contrast1 = dataset.get_contrast1_gt()
            gt_contrast2 = dataset.get_contrast2_gt()

            gt_contrast1 = gt_contrast1.reshape((x_dim, y_dim, z_dim)).cpu().numpy()
            gt_contrast2 = gt_contrast2.reshape((x_dim, y_dim, z_dim)).cpu().numpy()
            


if __name__ == '__main__':
    args = parse_args()
    main(args)
