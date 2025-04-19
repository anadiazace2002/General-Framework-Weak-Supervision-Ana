from datetime import datetime
import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

from src.core.algorithmbase import AlgorithmBase
from src.core.utils import Argument, str2bool, get_optimizer, send_model_cuda
from src.core.criterions import CELoss
from src.core.hooks import CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, ParamUpdateHook, EvaluationHook, EMAHook
from src.datasets import get_sym_noisy_labels, get_cifar10_asym_noisy_labels, get_cifar100_asym_noisy_labels, get_data, get_dataloader, ImgBaseDataset, ImgTwoViewBaseDataset

from hoc import * 

class ImpreciseNoisyLabelLearning(AlgorithmBase):
    def __init__(self, args, tb_log=None, logger=None, **kwargs):
        self.init(args)

        super().__init__(args, tb_log, logger, **kwargs)
        self.dataset_dict = self.set_dataset()
        self.loader_dict = self.set_data_loader()
        
        self.num_train_iter = self.epochs * len(self.loader_dict['train'])
        self.num_eval_iter = len(self.loader_dict['train'])
        self.ce_loss = CELoss()
        
        self.transition_matrix = None
    
    def init(self, args):
        # extra arguments 
        self.average_entropy_loss = args.average_entropy_loss
        self.noise_ratio = args.noise_ratio
        self.noise_type = args.noise_type
        self.noise_matrix_scale = args.noise_matrix_scale



    def set_hooks(self):
        # parameter update hook is called inside each train_step
        # self.register_hook(NoiseParamUpdateHook(), None, "HIGHEST")   our transition is fixed
        if self.ema_model is not None:
            self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay, self.args.layer_decay, nesterov=False, bn_wd_skip=False)
        if self.args.dataset == 'webdataset':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50 * len(self.loader_dict['train'])]) 
        elif self.args.dataset == 'clothing1m':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[7 * len(self.loader_dict['train'])]) 
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(self.loader_dict['train'])), eta_min=2e-4)
        return optimizer, scheduler
        
    # --------------------------------------------------------------------------------------------------

    # Functions for the instance-dependent noise transition matrix 

    def get_TP_real(self, clean_label, noisy_label):
        T_real = np.zeros((self.num_classes,self.num_classes))
        for i in range(clean_label.shape[0]):
            T_real[clean_label[i]][noisy_label[i]] += 1
        P_real = [sum(T_real[i])*1.0 for i in range(self.num_classes)] # random selection
        for i in range(self.num_classes):
            if P_real[i]>0:
                T_real[i] /= P_real[i]
        P_real = np.array(P_real)/sum(P_real)
        print(f'Check: P = {P_real},\n T = \n{np.round(T_real,3)}')
        return T_real, P_real

    def get_T_global_min(self, record, clean_label,noisy_label, max_step = 501, T0 = None, p0 = None, lr = 0.1, NumTest = 50, all_point_cnt = 15000):
        print()
        print("Entering into get_T_global_min")
        print()
        total_len = sum([len(a) for a in record])
        origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
        origin_label = torch.zeros(total_len).long()
        cnt, lb = 0, 0
        for item in record:
            for i in item:
                origin_trans[cnt] = i['feature']
                origin_label[cnt] = lb
                cnt += 1
            lb += 1
        data_set = {'feature': origin_trans, 'noisy_label': origin_label}
    
        # Build Feature Clusters --------------------------------------
        KINDS = self.num_classes
        # NumTest = 50
        # all_point_cnt = 15000
        T_real, P_real = self.get_TP_real(clean_label,noisy_label)
    
        p_estimate = [[] for _ in range(3)]
        p_estimate[0] = torch.zeros(KINDS)
        p_estimate[1] = torch.zeros(KINDS, KINDS)
        p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
        p_estimate_rec = torch.zeros(NumTest, 3)
        for idx in range(NumTest):
            print(idx, flush=True)
            # global
            sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
            # final_feat, noisy_label = get_feat_clusters(data_set, sample)
            final_feat = data_set['feature'][sample]
            noisy_label = data_set['noisy_label'][sample]
            cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
            for i in range(3):
                cnt_y_3[i] /= all_point_cnt
                p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]
        for j in range(3):
            p_estimate[j] = p_estimate[j] / NumTest
    
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_min, E_calc, P_calc, T_init = calc_func(KINDS, p_estimate, False, self.args.device, max_step, T0, p0, lr = lr)
    
        E_calc = E_calc.cpu().numpy()
        T_init = T_init.cpu().numpy()
        print(f"L11 Error (Global): {np.sum(np.abs(E_calc - np.array(T_real))) * 1.0 / (KINDS*KINDS) * 100}")
        return E_calc, T_init
    
    
    def find_trans_mat(self, lr):
        # estimate each component of matrix T based on training with noisy labels
        print("\nEstimating transition matrix...")

        clean_label = np.array(self.train_dataset.true_labels) 
        noisy_label = np.array(self.train_dataset.noisy_labels)
        print()
        record = [[] for _ in range(self.num_classes)]
        # collect all the outputs
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loader_dict['train']):
                # Extract data correctly
                x_w = batch["x_w"]  # Should be a tensor
                x_s = batch["x_s"]  # Should be a tensor
                y = batch["y"]  # Labels
                
                # Ensure tensors are on GPU
                x_w = x_w.float().cuda()
                x_s = x_s.float().cuda()
                # En este caso, 'y' es lo mismo que 'label' y corresponde a la etiqueta ruidosa
                _noisy_label = y
            
                # Accedemos a las etiquetas verdaderas y las ruidosas
                true_label = self.train_dataset.true_labels[batch_idx]  # O como accedas a las etiquetas verdaderas
                
                # Variable para contar el índice global de las imágenes
                global_idx = 0  # Este será el índice global de las imágenes
                
                # Usamos la vista 'x_w' (o 'x_s', dependiendo de tu elección)
                data = x_w  # O también podrías usar `x_s` en lugar de `x_w` si lo prefieres
                # Hook function to capture the features

                features = []
                
                def hook_fn(module, input, output):
                    features.append(output)
                
                # Register the hook on the last layer (layer4) before the fully connected layer
                hook = self.model.module.layer4[-1].register_forward_hook(hook_fn)  # Adjust this based on your model's structure
                # Asegúrate de que los datos estén en el formato adecuado
                data = data.float().cuda()
                _ = self.model.module(data) 
                # Extraemos las características con la red
                
                extracted_feature = torch.flatten(features[0], start_dim=1)
                
                hook.remove()
                # Crear el registro para las características extraídas
                for i in range(extracted_feature.shape[0]):
                    record[_noisy_label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': global_idx})
                    global_idx += 1
        new_estimate_T, _ = self.get_T_global_min(record, clean_label, noisy_label, max_step=1500, lr=0.1, NumTest=50)
        return torch.tensor(new_estimate_T).float().cuda()
        
    # ---------------------------------------------------------------------------
    
    def set_dataset(self):
        # get initial data
        train_data, train_targets, test_data, test_targets, extra_data = get_data(self.args.data_dir, self.args.dataset)
        
        if self.noise_type == 'sym':
            assert self.args.dataset in ['cifar10', 'cifar100']
            noise_idx, train_data, train_noisy_targets = get_sym_noisy_labels(train_data, train_targets, self.num_classes, self.noise_ratio)
        elif self.noise_type == 'asym':
            if self.args.dataset == 'cifar10':
                noise_idx, train_data, train_noisy_targets = get_cifar10_asym_noisy_labels(train_data, train_targets, self.num_classes, self.noise_ratio)
            elif self.args.dataset == 'cifar100':
                noise_idx, train_data, train_noisy_targets = get_cifar100_asym_noisy_labels(train_data, train_targets, self.num_classes, self.noise_ratio)
            else:
                raise NotImplementedError
        elif self.noise_type == 'ins':
            if self.args.dataset == 'cifar10n':
                noise_file = torch.load(os.path.join(self.args.data_dir, 'cifar10n', 'CIFAR-10_human.pt'))
                assert self.noise_ratio in ['clean_label', 'worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']
                train_noisy_targets = noise_file[self.noise_ratio]
            elif self.args.dataset == 'cifar100n':
                noise_file = torch.load(os.path.join(self.args.data_dir, 'cifar100n', 'CIFAR-100_human.pt'))
                assert self.noise_ratio in ['clean_label', 'noisy_label']
                train_noisy_targets = noise_file[self.noise_ratio]
            else:
                # noisy labels is directly loaded in train_targets
                train_noisy_targets = train_targets
        else:
            raise NotImplementedError
        
        if self.args.dataset in ['cifar10', 'cifar100', 'cifar10n', 'cifar100n']:
            resize = 'resize_crop_pad'
            if self.args.dataset == 'cifar10' or self.args.dataset == 'cifar10n':
                # autoaug = 'randaug'
                autoaug = 'autoaug_cifar'
            else:
                autoaug = 'autoaug_cifar'
            test_resize = 'resize'
        elif self.args.dataset == 'webvision':
            resize = 'resize_rpc'
            autoaug = 'autoaug'
            test_resize = 'resize'
        elif self.args.dataset == 'clothing1m':
            resize = 'resize_crop'
            autoaug = 'autoaug'
            test_resize = 'resize_crop'
        else:
            resize = 'rpc'
            autoaug = 'autoaug'
            test_resize = 'resize_crop'
            
        if not self.strong_aug:
            autoaug = None
        
        # make dataset
        train_dataset = ImgTwoViewBaseDataset(self.args.dataset, train_data, train_noisy_targets, 
                                              num_classes=self.num_classes, is_train=True,
                                              img_size=self.args.img_size, crop_ratio=self.args.crop_ratio,
                                              autoaug=autoaug, resize=resize,
                                              return_target=True, return_keys=['x_w', 'x_s', 'y'])
        test_dataset = ImgBaseDataset(self.args.dataset, test_data, test_targets, 
                                      num_classes=self.num_classes, is_train=False,
                                      img_size=self.args.img_size, crop_ratio=self.args.crop_ratio, resize=test_resize,
                                      return_keys=['x', 'y'])
        
        train_dataset.true_labels = train_targets
        train_dataset.noisy_labels = train_noisy_targets  
        test_dataset.true_labels = test_targets
        
        self.train_dataset = train_dataset  # ✅ Now train_dataset is set
        self.test_dataset = test_dataset
    
        self.print_fn("Datasets and transition matrix created!")

        return {'train': train_dataset, 'eval': test_dataset}
        
    def set_data_loader(self):
        loader_dict = {}

        loader_dict['train'] = get_dataloader(self.dataset_dict['train'], 
                                              num_epochs=self.epochs, 
                                              batch_size=self.args.batch_size, 
                                              shuffle=True, 
                                              num_workers=self.args.num_workers, 
                                              pin_memory=True, 
                                              drop_last=True,
                                              distributed=self.args.distributed)
        loader_dict['eval'] = get_dataloader(self.dataset_dict['eval'], 
                                             num_epochs=self.epochs, 
                                             batch_size=self.args.eval_batch_size, 
                                             shuffle=False, 
                                             num_workers=self.args.num_workers, 
                                             pin_memory=True, 
                                             drop_last=False)
        self.print_fn("Create data loaders!")
        return loader_dict


    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        # Solo se calcula una vez antes de entrenar
        if self.transition_matrix is None:
            self.print_fn("Calculating transition matrix once before training...")
            self.transition_matrix = self.find_trans_mat(lr=0.1).detach()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_dir = os.path.join(self.args.save_dir, 'matrix', f'matrix_{timestamp}')
            os.makedirs(experiment_dir, exist_ok=True)
            
            np.savetxt(os.path.join(experiment_dir, 'matrix.csv'), self.transition_matrix.cpu().numpy(), delimiter=',')



        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")
            

            for data in self.loader_dict['train']:
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break
                
                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data))
                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    

    
    def train_step(self, x_w, x_s, y):    
            
        inputs = torch.cat((x_w, x_s))
        true_outputs = self.model(inputs)
        logits_x_w, logits_x_s = true_outputs.chunk(2)    # logits computation
        # noise_matrix = self.noise_model(logits_x_w)       # noise matrix creation BEFORE
        noise_matrix = self.transition_matrix.cuda()
        # noise_matrix *= 2
        
        # convert logits_w to probs
        probs_x_w = logits_x_w.softmax(dim=-1).detach()
             
        # convert logits_s to probs
        probs_x_s = logits_x_s.softmax(dim=-1)
        # Compute EM estimates using the fixed transition matrix
        noise_matrix_col = noise_matrix[:, y].detach().transpose(0, 1)
        em_y = probs_x_w * noise_matrix_col                                # compute p(y|A_w(x), y_hat; θ, ω^t)
        em_y = em_y / em_y.sum(dim=1, keepdim=True)

        # compute forward_backward on graph x_s
        em_probs_x_s = probs_x_s * noise_matrix_col                                # compute p(y|A_s(x), y_hat; θ, ω^t)
        em_probs_x_s = em_probs_x_s / em_probs_x_s.sum(dim=1, keepdim=True)

        # Compute noisy loss using the fixed transition matrix
        noise_matrix_row = noise_matrix
        noisy_probs_x_w = torch.matmul(probs_x_w, noise_matrix_row)                        # p(y_hat|A_w(x);θ,ω)
        noisy_probs_x_w = noisy_probs_x_w / noisy_probs_x_w.sum(dim=-1, keepdim=True)

        # compute noisy loss = LCE(p(y_hat|A_w(x);θ,ω),y_hat)    	y_hat --> y
        noise_loss = torch.mean(-torch.sum(F.one_hot(y, self.num_classes) * torch.log(noisy_probs_x_w), dim=-1))
        
        # compute em loss = LCE(p(y|A_s(x), y_hat; θ^t, ω^t), p(y|A_w(x), y_hat; θ, ω^t))
        em_loss = torch.mean(-torch.sum(em_y * torch.log(em_probs_x_s), dim=-1), dim=-1)
        con_loss = self.ce_loss(logits_x_s, probs_x_w, reduction='mean')
        
        # Total loss
        loss = noise_loss + em_loss + con_loss
        
        # computer average entropy loss -- to make the model be more balanced in its predictions
        if self.average_entropy_loss:
            avg_prediction = torch.mean(logits_x_w.softmax(dim=-1), dim=0)
            prior_distr = 1.0/self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min = 1e-6, max = 1.0)
            balance_kl =  torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = 0.1 * balance_kl
            loss += entropy_loss

        out_dict = self.process_out_dict(loss=loss)
        log_dict = self.process_log_dict(loss=loss.item(), 
                                         noise_loss=noise_loss.item(),
                                         em_loss=em_loss.item(),
                                         con_loss=con_loss.item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # save_dict['noise_model'] = self.noise_model.cpu().numpy()  # Save fixed transition matrix
        return save_dict



    @staticmethod
    def get_argument():
        return [
            Argument('--average_entropy_loss', str2bool, True, 'use entropy loss'),
            Argument('--noise_ratio', float, 0.1, 'noise ratio for noisy label learning'),
            Argument('--noise_type', str, 'sym', 'noise type (sym, asym, ins) noisy label learning'),
            Argument('--noise_matrix_scale', float, 1.0, 'scale for noise matrix'),
        ]
