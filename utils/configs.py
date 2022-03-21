"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import torch
import numpy as np
import os



SDFormer_plate_config = {'name':'SwinTransformer',
                         'hidden_dim':[64,128,256],
                         'layers':[2,2,16],
                         'heads':[4,8,16],
                         'window_size': 4,
                         'down_scaling_factors':[2,2,2],
                         'head_dim':32,
                         'relative_pos_embedding':True,
                         'skip_connect':True}

SDFormer_sleeper_config = {'name':'SwinTransformer',
                           'hidden_dim':[64,128,256],
                           'layers':[2,2,16],
                           'heads':[4,8,16],
                           'window_size':[[4,4],[4,4],[4,4]],
                           'down_scaling_factors':[2,2,2],
                           'head_dim':32,
                           'relative_pos_embedding':True,
                           'skip_connect':True}


def get_segmodel_config(model_type='Unet', backbone_name='resnet18'):
    segmodelConfig = {'name':'Segmodel',
                      'model_name':model_type,
                      'encoder':backbone_name,
                      'activation':'softmax',
                      'init_by_imagenet':False}
    return segmodelConfig

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

class config(object):
    def __init__(self):
        super(config, self).__init__()
        #==================select comparison model==================#
        model_type='Unet'
        backbone_name='resnet18'
        #==================select dataset==================#
        self.datasetnames = ['plate', 'sleeper_beam']
        self.datasetindex = 0
        self.task = 'seg'
        #self.task = 'gen'
        self.is_seg = True
        if self.task =='gen':
            self.is_seg=False
        #==================model setting==================#
        self.model_index = 1
        self.basic_model_names = ['model','SDFormer64', 'SDFormer264']
        self.model_name = self.task + '_' + self.basic_model_names[self.model_index] + '_' + self.datasetnames[self.datasetindex]
        
        self.input_channel = 3
        self.n_classes = 4

        # model config setting
        self.segmodelConfig = get_segmodel_config(model_type=model_type, backbone_name=backbone_name)
        if self.model_index==0:
            self.model_name = self.model_name+'_'+self.segmodelConfig['model_name']+'_'+self.segmodelConfig['encoder']
        self.configs = [self.segmodelConfig,  
                        SDFormer_plate_config, 
                        SDFormer_sleeper_config]
        #==================data path==================#
        self.test_result_save_path = './prediction_result/EN'
        self.save_model_path = './user_data/EN/model_data'
        self.train_log = './user_data/EN/train_log'
        
        self.train_path = './data/Abaqus_ModelBase'
        self.val_path = './data/Abaqus_ModelBase/ENtestdata'
        #self.train_path = './data/Abaqus_ModelBase/sleeper_beam'

        if not os.path.exists(self.train_path):
            self.train_path = '../data/Abaqus_ModelBase'
            self.val_path = '../data/Abaqus_ModelBase/ENtestdata'
            #self.train_path = '../data/Abaqus_ModelBase/sleeper_beam'
            #self.test_result_save_path = '../prediction_result'
            #self.save_model_path = '../user_data/model_data'
            #self.train_log = '../user_data/train_log'
        self.train_path = os.path.join(self.train_path, self.datasetnames[self.datasetindex])
        self.val_path = os.path.join(self.val_path, self.datasetnames[self.datasetindex])
        self.test_result_save_path = os.path.join(self.test_result_save_path, self.model_name)
        self.train_log = os.path.join(self.train_log, self.model_name)
        self.save_model_path = os.path.join(self.save_model_path, self.model_name)
        #self.datapath=os.path.join(self.train_path, 'input.npy')
        #self.labelpath = os.path.join(self.train_path, 'output.npy')
        
        self.ground_truth_path = os.path.join(self.test_result_save_path, 'ground_truth')
        self.input_image_path = os.path.join(self.test_result_save_path, 'input')
        self.predict_path = os.path.join(self.test_result_save_path, 'predict')

        self.xx_img_path = os.path.join(self.input_image_path, 'xx')
        self.yy_img_path = os.path.join(self.input_image_path, 'yy')
        self.xy_img_path = os.path.join(self.input_image_path, 'xy')
        self.color_img_path = os.path.join(self.input_image_path, 'color')
        
        if not os.path.exists(self.test_result_save_path):
            os.makedirs(self.test_result_save_path)

        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
            
        if not os.path.exists(self.train_log):
            os.makedirs(self.train_log)
            
        if not os.path.exists(self.ground_truth_path):
            os.makedirs(self.ground_truth_path)
            
        if not os.path.exists(self.input_image_path):
            os.makedirs(self.input_image_path)
        
        if not os.path.exists(self.predict_path):
            os.makedirs(self.predict_path)

        if not os.path.exists(self.xx_img_path):
            os.makedirs(self.xx_img_path)
        
        if not os.path.exists(self.yy_img_path):
            os.makedirs(self.yy_img_path)

        if not os.path.exists(self.xy_img_path):
            os.makedirs(self.xy_img_path)
            
        if not os.path.exists(self.color_img_path):
            os.makedirs(self.color_img_path)
            
        #==================main model train setting==================#
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.epochs = 51
        self.batch_size = 4
        self.lr = 0.00008
        self.weight_decay = 5e-4
        self.momentum=0.8
        
        self.val_rate = 0.005
        self.shuffle_dataset = False
        
        self.use_fp16 = False
        
        #==================dataset setting==================#
        self.gen_min_mask = 20
        self.gen_mask_num = 50
        self.noise_level = 0.2
        
        #==================result save path==================#
        self.best_model_name = self.save_model_path +'/best_loss_model.pth'
        self.save_model_name = self.save_model_path +'/epoch_model.pth'
        self.best_iou_model = os.path.join(self.save_model_path, 'checkpoint-best.pth')

        #==================visual setting==================#
        self.is_pretrain = False
        filename = os.path.join(self.save_model_path, self.model_name + 'checkpoint-best.pth')
        if os.path.exists(filename):
            self.is_pretrain = True
        
        self.show_iter = 10
        self.save_iter = 10
        self.min_iter = 5
        
        self.binary_threshold = 0.5
        
        #self.color_array = np.random.randint(255, size=(self.n_classes,3))
        '''
        color_array = [[68,108,164],
                       [72,57,168],
                       [206,178,73],
                       [220,12,17]]'''
        color_array = [[255,225,0],
                        [225,105,65],
                        [92,92,205],
                        [0,215,255]]
        self.color_array = np.array(color_array)

        # ====================================================
        self.alpha = 0.25
      
if __name__ == "__main__":
    Opt = config()
