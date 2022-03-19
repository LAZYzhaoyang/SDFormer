# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import os


import numpy as np
from config import config


def create_model(mdb, model_name, model_type='standard'):
    if model_type == 'standard':
        mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)
    elif model_type == 'eletromagenetic':
        mdb.Model(name=model_name, modelType=EMAG)
    else:
        ValueError('please check the model_type')
    
    return mdb.models[model_name]

def create_part(mdb, model_name, part_name='table', h=64, w=64):
    s1 = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=200.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    # 新建一个矩形2D壳单元
    s1.rectangle(point1=(0.0, 0.0), point2=(w, h))
    p = mdb.models[model_name].Part(name=part_name, dimensionality=TWO_D_PLANAR, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts[part_name]
    p.BaseShell(sketch=s1)
    s1.unsetPrimaryObject()
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    #删除之前的草图
    del mdb.models[model_name].sketches['__profile__']
    p = mdb.models[model_name].parts[part_name]
    
    return p
    

def create_MeshPart(mdb, model_name, part_name, mesh_part_name, size=1.0, deviationFactor=0.1, minSizeFactor=0.1):
    p = mdb.models[model_name].parts[part_name]
    p.seedPart(size=size, deviationFactor=deviationFactor, minSizeFactor=minSizeFactor)
    p.generateMesh()
    p.PartFromMesh(name=mesh_part_name, copySets=True)
    mesh_part = mdb.models[model_name].parts[mesh_part_name]
    return mesh_part


def create_material(mdb, model_name, material_name, material_section_name, density=2.7e3, Youngs_modulus=7e9, Poisson_Ratio=0.3):
    #设置材料
    mdb.models[model_name].Material(name=material_name)
    mdb.models[model_name].materials[material_name].Density(table=((density, ), ))
    mdb.models[model_name].materials[material_name].Elastic(table=((Youngs_modulus, 
        Poisson_Ratio), ))
    #设置材料section
    mdb.models[model_name].HomogeneousSolidSection(name=material_section_name, 
            material=material_name, thickness=None)
    

def set_part_material(mdb, model_name, part_name, material_section_name, region):
    p = mdb.models[model_name].parts[part_name]
    p.SectionAssignment(region=region, sectionName=material_section_name, offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)


def assembly(mdb, model_name, part_name, instance_name):
    a = mdb.models[model_name].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models[model_name].parts[part_name]
    a.Instance(name=instance_name, part=p, dependent=ON)
    return a
    

def set_step(mdb, model_name, step_name, previous_step='Initial', 
             timePeriod=0.2, initialInc=0.2, 
             minInc=2e-06, maxInc=0.2):
    mdb.models[model_name].StaticStep(name=step_name, previous=previous_step)
    #session.viewports['Viewport: 1'].assemblyDisplay.setValues(step=step_name)
    mdb.models[model_name].steps[step_name].setValues(timePeriod=timePeriod, initialInc=initialInc, 
                                                      minInc=minInc, maxInc=maxInc)

def get_set(part, index, set_name, type_name='elements'):
    if type_name is 'elements':
        region = part.SetFromElementLabels(set_name, index)
    elif type_name is 'node':
        region = part.SetFromNodeLabels(set_name, index)
    else:
        ValueError('type_name must be elements of node')
    
    return region


def mask2index(mask):
    h, w = mask.shape
    listmask = mask.reshape(h*w)
    index = np.argwhere(listmask==1)
    index = index.reshape(len(index))+1
    index = np.trunc(index).astype(int).tolist()
    return index



def set_BC(mdb, model_name, BC_name,  region, step_name='Initial'):
    mdb.models[model_name].EncastreBC(name=BC_name, createStepName=step_name, 
                                     region=region, localCsys=None)

def set_Load(mdb, model_name, load_name, step_name, region, cf1=1000, cf2=0):
    mdb.models[model_name].ConcentratedForce(name=load_name, createStepName=step_name, 
            region=region, cf1=cf1, cf2=cf2, distributionType=UNIFORM, 
            field='', localCsys=None)

def create_job(mdb, model_name, job_name):
    mdb.Job(name=job_name, model=model_name, description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, 
        numDomains=4, numGPUs=1)


def get_output(session, odb_path, output_file_name):
    
    o3 = session.openOdb(name = odb_path)
    odb = session.odbs[odb_path]
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    session.fieldReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES)
    nf = NumberFormat(numDigits=6, precision=0, format=ENGINEERING)
    session.writeFieldReport(fileName=output_file_name, append=OFF, 
        sortItem='Element Label', odb=odb, step=0, frame=1, 
        outputPosition=INTEGRATION_POINT, variable=(('E', INTEGRATION_POINT, ((
        COMPONENT, 'E11'), (COMPONENT, 'E22'), (COMPONENT, 'E12'), )), ), 
        stepFrame=SPECIFY)
    session.odbs[odb_path].close()

        

def main(config):
    mdb = Mdb(config.mdb_path)
    os.chdir(config.path)
    
    h, w = config.table_size
    BCmask = np.zeros((h+1,w+1))
    Loadmask = np.zeros_like(BCmask)
    
    BCmask[:,0]=1
    Loadmask[:,-1]=1
    
    BCindex = mask2index(BCmask)
    Loadindex = mask2index(Loadmask)
    
    for i in range(config.mask_num):
        # set name
        model_name = config.base_model_name + str(i)
        part_name = config.base_part_name + str(i)
        mesh_part_name = config.base_part_name + 'mesh-' + str(i)
        Job_name = config.base_job_name
        BC_name = config.base_BC_name + str(i)
        Load_name = config.base_Load_name + str(i)
        step_name = config.step_name + str(i)
        instance_name=mesh_part_name+'-1'
        
        odb_name = Job_name+config.base_ODB_name
        odb_path = os.path.join(config.path, odb_name)
        
        mask_file_name = '{:0>5d}.npy'.format(i)
        mask_file_path = os.path.join(config.mask_path, mask_file_name)
        data_file_name = '{:0>5d}.csv'.format(i)
        data_file_path = os.path.join(config.data_path, data_file_name)
        
        # create model and part
        model = create_model(mdb, model_name=model_name)
        table = create_part(mdb=mdb, model_name=model_name, part_name=part_name, h=h, w=w)
        TableMesh = create_MeshPart(mdb=mdb, model_name=model_name, part_name=part_name, mesh_part_name=mesh_part_name,
                                    size=config.mesh_size)
        
        # set material
        create_material(mdb=mdb, model_name=model_name, material_name=config.material_name, 
                        material_section_name=config.material_section_name, density=config.density,
                        Youngs_modulus=config.youngs_modulus, Poisson_Ratio=config.poisson_ratio)
        
        for i in range(len(config.damage_rate)):
            youngs_modules = config.youngs_modulus * (1-config.damage_rate[i])
            #print(config.damage_name[i],config.damage_section_name[i])
            create_material(mdb=mdb, model_name=model_name, material_name=config.damage_name[i], 
                            material_section_name=config.damage_section_name[i], density=config.density, 
                            Youngs_modulus=youngs_modules, Poisson_Ratio=config.poisson_ratio)
        # put material
        damage_mask = np.load(mask_file_path)
        
        #np.save(mask_file_path, damage_mask)
        
        normal_index = mask2index(damage_mask==0)
        normal_region = get_set(TableMesh, normal_index, 'normal_set')
        set_part_material(mdb=mdb, model_name=model_name, part_name=mesh_part_name, 
                          material_section_name=config.material_section_name, 
                          region=normal_region)
        for i in range(len(config.damage_rate)):
            damage_index = mask2index(damage_mask==i+1)
            if len(damage_index)==0:
                continue
            damage_region = get_set(TableMesh, damage_index, config.damage_name[i], type_name='elements')
            set_part_material(mdb=mdb, model_name=model_name, part_name=mesh_part_name, 
                              material_section_name=config.damage_section_name[i], 
                              region=damage_region)
            
        # assembly
        a = assembly(mdb=mdb, model_name=model_name, part_name=mesh_part_name, instance_name=instance_name)
        a = mdb.models[model_name].rootAssembly
        #print(type(a), type(TableMesh), type(a.instances[instance_name]))
        print(type(a.instances[instance_name].nodes))
        nodes = a.instances[instance_name].nodes
        # set step
        set_step(mdb=mdb, model_name=model_name, step_name=step_name, timePeriod=config.step_time, 
                 initialInc=config.initialInc, minInc=config.minInc, maxInc=config.maxInc)
        # set BC
        
        #BC_region = a.SetFromNodeLabels(name='set-BC', nodeLabels=((instance_name, BCindex)))
        #BC_region = get_set(part=a, index=BCindex, set_name='set-BC', type_name='node')
        #nodesBC= nodes.getSequenceFromMask(mask=BCindex)
        nodesBC= nodes.sequenceFromLabels(BCindex)
        BC_region = a.Set(nodes=nodesBC, name='Set-BC')
        set_BC(mdb=mdb, model_name=model_name, BC_name=BC_name,  region=BC_region)
        
        # set load
        #Load_region = get_set(part=a, index=Loadindex, set_name='set-Load', type_name='node')
        nodesLoad = nodes.sequenceFromLabels(Loadindex)
        Load_region = a.Set(nodes=nodesLoad, name='Set-Load')
        set_Load(mdb=mdb, model_name=model_name, load_name=Load_name, step_name=step_name, region=Load_region)
        
        # create and submit Job
        create_job(mdb=mdb, model_name=model_name, job_name=Job_name)
        mdb.jobs[Job_name].submit(consistencyChecking=OFF)
        mdb.jobs[Job_name].waitForCompletion()#等待计算结束
        # save result
        get_output(session=session, odb_path=odb_path, output_file_name=data_file_path)
        del mdb.models[model_name]
        del mdb.jobs[Job_name]
        
        
        
if __name__ == "__main__":
    config = config()
    main(config)
        
    
    
    
