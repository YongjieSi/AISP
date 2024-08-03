import copy
import logging
import numpy as np
import torch
from torch import nn, Tensor
import torchaudio
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear,SimpleContinualLinear
import timm
from models.ast_models import ASTModel
from torch.nn import functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from easydict import EasyDict



def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if args["dataset"] == 'FMC':
        model = ASTModel(input_tdim=44, label_dim=527,  imagenet_pretrain  = True, audioset_pretrain=True)
    if 'nsynth' in args["dataset"]:
        model = ASTModel(input_tdim=63, label_dim=527, imagenet_pretrain  = True,audioset_pretrain=True)
    if 'librispeech' in args["dataset"]:
        model = ASTModel(input_tdim=201, label_dim=527, imagenet_pretrain  = True,audioset_pretrain=True)
    model.out_dim = 768
    return model.eval()
    


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        # self.fc_for_train = None
        self._device = args["device"][0]
        self.args = args
        self.fc_for_train = None
        self.set_fea_extractor()
            

    @property
    def feature_dim(self):
        return self.backbone.out_dim
        
    def mel_feature(self, x):
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                                                 # (batch_size, 1, time_steps, mel_bins)
        x = x.squeeze(1)
        return x
        
    def set_fea_extractor(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        
    def extract_vector(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        return self.backbone(x)

    def forward(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self



import torch
import torch.nn as nn
from torch import Tensor

class MLPWeightedFeaturesFusion(nn.Module):
    def __init__(self, num_layers, feature_dim, hidden_dim):
        super(MLPWeightedFeaturesFusion, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 定义MLP作为权重学习模型
        self.mlp = nn.Sequential(
            nn.Linear(self.num_layers * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_layers),
            nn.Softmax(dim=-1)  # 使得权重参数之和为1
        )

    def forward(self, outputs):
        # 将每一层的输出特征堆叠在一起
        # stacked_outputs = torch.stack(outputs, dim=1)
        stacked_outputs = outputs #(6,25,768)
        stacked_outputs = stacked_outputs.transpose(1, 0) #(25,6,768)
        # 使用MLP学习加权参数
        weights = self.mlp(stacked_outputs.reshape(-1, self.num_layers * stacked_outputs.size(2))) #in:(25,6*768),out(25,6)

        # 对每一层的输出特征进行加权融合
        weighted_output = weights.unsqueeze(-1) * stacked_outputs
        sum_weighted_output = torch.sum(weighted_output, dim=1)

        return sum_weighted_output, weighted_output



class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.set_fea_extractor()
        # self.fc1 = SimpleLinear(768, 768)
        self.num_features = 768

    def set_fea_extractor(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            # fc.sigma.data = self.fc.sigma.data
            # bias = copy.deepcopy(self.fc.bias.data)
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
                # bias = torch.cat([bias, torch.zeros(nb_classes - nb_output).to(self._device)])
            fc.weight = nn.Parameter(weight)
            # fc.bias = nn.Parameter(bias)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.args['sigma'] == True:
            fc = CosineLinear(in_dim, out_dim,sigma=True)
        else:
            fc = CosineLinear(in_dim, out_dim)
        # fc = SimpleLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        with torch.no_grad():
            _,specific_features = self.backbone(x)
            features = torch.cat((specific_features[6:,:,:],specific_features[6:,:,:]),dim=0)
        return features


    def forward(self, x, old_layers_features = None):
        
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        x, layer_features= self.backbone(x)
        if old_layers_features is not None:
            x =torch.cat((x,old_layers_features.squeeze(0)),dim = 0)
        # logits1 = self.fc1(x)
        # out = self.fc(logits1['logits'])
        out = self.fc(x)
        out.update({"features": x})
        return out


    def mel_feature(self, x):
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                                                 # (batch_size, 1, time_steps, mel_bins)
        x = x.squeeze(1)
        return x
    
class FusionVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.set_fea_extractor()
        self.num_features = 768
        self.fusion_module= MLPWeightedFeaturesFusion(self.args['blocks'], 768, 64)
        # self.fusion_fast = MLPWeightedFeaturesFusion(6, 768, 64)
        # self.fusion_adam = MLPWeightedFeaturesFusion(2, 768, 64)
        self.fc1 = SimpleLinear(768, 768)
        
    def set_fea_extractor(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            # fc.sigma.data = self.fc.sigma.data
            # bias = copy.deepcopy(self.fc.bias.data)
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
                # bias = torch.cat([bias, torch.zeros(nb_classes - nb_output).to(self._device)])
            fc.weight = nn.Parameter(weight)
            # fc.bias = nn.Parameter(bias)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.args['sigma'] == True:
            fc = CosineLinear(in_dim, out_dim,sigma=True)
        else:
            fc = CosineLinear(in_dim, out_dim)
        # fc = SimpleLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        with torch.no_grad():
            _,specific_features = self.backbone(x)
            features = torch.cat((specific_features[6:,:,:],specific_features[6:,:,:]),dim=0)
        return features

    def forward(self, x, old_layers_features = None):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        x, layer_features= self.backbone(x)
        specific_features = layer_features[6:,:,:]
        if old_layers_features != None:
            # specific_old_features = old_layers_features[6:]
            specific_features =torch.cat((specific_features,old_layers_features[6:,:,:]),dim = 1)
            weighted_features,_ = self.fusion_module(specific_features)
        else:
            weighted_features,_ = self.fusion_module(specific_features)
        logits1 = self.fc1(weighted_features)
        out = self.fc(logits1['logits'])
        # out = self.fc(weighted_features)
        out.update({"features": weighted_features})
        return out

    def mel_feature(self, x):
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                                                 # (batch_size, 1, time_steps, mel_bins)
        x = x.squeeze(1)
        return x


class AEFEVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.set_fea_extractor()
        self.num_features = 768
        self.fusion_past = MLPWeightedFeaturesFusion(12-self.args['blocks'], 768, 64)
        self.fusion_fast = MLPWeightedFeaturesFusion(12-self.args['blocks'], 768, 64)
        self.fusion_adam = MLPWeightedFeaturesFusion(2, 768, 64)

    def set_fea_extractor(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            # fc.sigma.data = self.fc.sigma.data
            # bias = copy.deepcopy(self.fc.bias.data)
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
                # bias = torch.cat([bias, torch.zeros(nb_classes - nb_output).to(self._device)])
            fc.weight = nn.Parameter(weight)
            # fc.bias = nn.Parameter(bias)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.args['sigma'] == True:
            fc = CosineLinear(in_dim, out_dim,sigma=True)
        else:
            fc = CosineLinear(in_dim, out_dim)
        # fc = SimpleLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        with torch.no_grad():
            _,specific_features = self.backbone(x)
            features = torch.cat((specific_features[6:,:,:],specific_features[6:,:,:]),dim=0)
        return features
    
    def forward(self, x, old_layers_features = None):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        x, layer_features= self.backbone(x)
        specific_features = layer_features[6:,:,:]
        if self.mode == "fsa_train":
            out = self.fc(x)
            out.update({"features": x})
        elif self.mode == "weight_train":
            if old_layers_features != None:
                # specific_old_features = old_layers_features[6:]
                specific_features =torch.cat((specific_features,old_layers_features[6:,:,:]),dim = 1)
                weighted_features,_ = self.fusion_fast(specific_features)
            else:
                weighted_features,_ = self.fusion_fast(specific_features)
            out = self.fc(weighted_features)
            out.update({"features": weighted_features})
        elif self.mode == "add":
            add_layer_features = torch.sum(torch.stack(layer_features[6:,:,:]),dim=0)
            out = self.fc(add_layer_features)
            out.update({"add_layer_features": add_layer_features})
        elif self.mode == "cat":
            cat_layer_features = torch.cat(layer_features[6:,:,:],dim=1)
            out.update({"cat_layer_features": cat_layer_features})
        return out


    def mel_feature(self, x):
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                                                 # (batch_size, 1, time_steps, mel_bins)
        x = x.squeeze(1)
        return x

class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.aux_fc = None
        self.task_sizes = []
        self.out_dim = None
        # self.model_params_dict = model_params_dict
        # self.set_fea_extractor()
        # self.fusion_model = MLPWeightedFeaturesFusion(2, 768, 64)
        # no need the backbone.
        self.fc1 = SimpleLinear(768, 768)
        print('Clear the backbone in MultiBranchCosineIncrementalNet, since we are using self.backbones with dual branches')
        self.backbone=torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args=args


    def set_fea_extractor(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
    
    def mel_feature(self, x):
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.fs_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ns_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
            x = self.ls_logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
                                                 # (batch_size, 1, time_steps, mel_bins)
        x = x.squeeze(1)
        return x
    
    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            # fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.args['sigma'] == True:
            fc = CosineLinear(in_dim, out_dim,sigma=True)
        else:
            fc = CosineLinear(in_dim, out_dim)
        return fc
    

    def forward(self, x, old_layers_features = None):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        if old_layers_features != None:
            specific_old_features = old_layers_features[12-self.args['blocks']:]
            general_old_features = old_layers_features[:12-self.args['blocks']]
            _,specific_features = self.backbones[2](x)
            specific_features = specific_features[self.args['blocks']:,:,:]
            specific_features =torch.cat((specific_features,specific_old_features),dim = 1)
            weighted_specific_features,_ = self.backbones[3](specific_features)
            _,general_features = self.backbones[0](x)
            general_features = general_features[self.args['blocks']:,:,:]
            general_features =torch.cat((general_features,general_old_features),dim = 1)
            weighted_general_features,_ = self.backbones[1](general_features)
            features = torch.cat((weighted_specific_features.unsqueeze(0),weighted_general_features.unsqueeze(0)),dim = 0)
            sum_features_, features = self.backbones[-1](features)
            out1 = self.fc1(sum_features_)
            out = self.fc(out1['logits'])
            out.update({"features": sum_features_})
        else:
            _,specific_features = self.backbones[2](x)
            weighted_specific_features,_ = self.backbones[3](specific_features[self.args['blocks']:,:,:])
            _,general_features = self.backbones[0](x)
            weighted_general_features,_ = self.backbones[1](general_features[self.args['blocks']:,:,:])
            features =  torch.cat((weighted_specific_features.unsqueeze(0),weighted_general_features.unsqueeze(0)),dim = 0)
            sum_features_, features = self.backbones[-1](features)
            out1 = self.fc1(sum_features_)
            out = self.fc(out1['logits'])
            out.update({"features": sum_features_})
        return out

    def extract_vector(self, x, pool=True):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        with torch.no_grad():
            _,general_features = self.backbones[0](x)
            _,specific_features = self.backbones[2](x)
            features = torch.cat((general_features[self.args['blocks']:,:,:],specific_features[self.args['blocks']:,:,:]),dim=0)
        return features
    
    def construct_dual_branch_network(self, tuned_model):
        self.backbones.append(get_backbone(self.args)) #the pretrained model itself
        self.backbones.append(tuned_model.fusion_past) #adappted tuned model
        self.backbones.append(tuned_model.backbone) #
        self.backbones.append(tuned_model.fusion_fast)
        self.backbones.append(tuned_model.fusion_adam) # 
        self._feature_dim = self.backbones[0].out_dim
        self.fc=self.generate_fc(self._feature_dim,self.args['init_cls'])


