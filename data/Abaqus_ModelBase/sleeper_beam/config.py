"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import os

class config:
    def __init__(self):
        #===============set path===============#
        self.path = 'E:\MasterProgram\Paper\Damage-detection-of-sleeper-beam-in-Subway\code\data\Abaqus_ModelBase\sleeper_beam'
        self.mdb_name = 'v1.cae'
        self.mdb_path = os.path.join(self.path, self.mdb_name)
        
        self.data_path = os.path.join(self.path, 'data')
        self.mask_path = os.path.join(self.path, 'mask')
        self.img_data_path = os.path.join(self.path, 'img')
        
        #===============check path===============#
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.img_data_path):
            os.makedirs(self.img_data_path)
            
        #===============set parameter===============#
        # base
        self.table_size = (70, 264)
        self.number_of_data = 2000
        self.mask_num = len(os.listdir(self.mask_path))
        #print('there have '+ str(self.mask_num)+' masks')
        # name
        self.base_model_name = 'Models-'
        self.base_job_name = 'JobZhaoyang'
        self.base_ODB_name = '.odb'
        
        # material
        self.material_names = ['STEEL', 'STEEL-20damage', 'STEEL-40damage', 'STEEL-60damage', 'STEEL-PLASTIC']
        self.material_section_name = 'up-well'
        
        # damage set
        #self.damage_rate = [0.2, 0.4, 0.6]    
        self.damage_name = ['STEEL-60damage', 'STEEL-40damage', 'STEEL-20damage']
        self.damage_section_name = ['up-60damage', 'up-40damage', 'up-20damage']
        self.class_num = len(self.damage_name)+1  
          
        self.min_area_rate = 0.001
        self.max_area_rate = 0.5
        self.angle_num = 8
        self.spot_num = 500
        self.Max_R = 7
        self.min_R = 1
        self.thresh = 0.5
        self.win_size = 5
