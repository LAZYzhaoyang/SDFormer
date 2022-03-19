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
        self.path = 'E:\MasterProgram\Paper\Damage-detection-of-sleeper-beam-in-Subway\code\data\Abaqus_ModelBase\plate'
        self.mdb_name = 'abaqus.cae'
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
        self.table_size = (64, 64)
        self.number_of_data = 1000
        self.mask_num = len(os.listdir(self.mask_path))
        #print('there have '+ str(self.mask_num)+' masks')
        # name
        self.base_model_name = 'Models-'
        self.base_part_name = 'table-'
        self.base_job_name = 'JobZhaoyang'
        self.base_BC_name = 'BC-'
        self.base_Load_name = 'Load-'
        self.step_name = 'Step-'
        self.base_ODB_name = '.odb'
        
        # material
        self.density = 2700
        self.youngs_modulus = 7e9
        self.poisson_ratio = 0.33
        self.material_name = 'Aluminium'
        self.material_section_name = self.material_name + '_section'
        
        # damage set
        self.damage_rate = [0.2, 0.4, 0.6]
        self.class_num = len(self.damage_rate)+1
        
        self.damage_name = []
        self.damage_section_name = []
        for i in range(len(self.damage_rate)):
            name = 'damage'+str(i+1)
            section_name = 'damage'+str(i+1)+'section'
            self.damage_name.append(name)
            self.damage_section_name.append(section_name)
            
        print(self.damage_name)
        print(self.damage_section_name)
            
        self.min_area_rate = 0.01
        self.max_area_rate = 0.1
        self.angle_num = 6
        self.spot_num = 80
        self.Max_R = 5
        self.min_R = 1
        self.thresh = 0.5
        self.win_size = 5

        # mesh
        self.mesh_size=1.0
        
        # step
        self.step_time = 0.2
        self.initialInc=0.2
        self.minInc=2e-06
        self.maxInc=0.2
        # BC and Load
        self.cf1 = 10000
        self.cf2 = 0
        
        
        # Job
        
        # output
