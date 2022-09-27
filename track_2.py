from PIL import Image

from math import ceil
from requests import HTTPError
import streamlit as st
import pandas as pd

import datetime
import time
from dateutil.parser import parse
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import requests
import torch
import torchvision
import sqlite3
#st.write('### Contagem de frutos')


#st.write('Opa, clicastes?')
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import numpy as np
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


#from modulos_dash.contagem.modulo_pagina_contagem import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
data = []
count_fly = 0
count_fly3 = 0
count_fly2 = 0
data_fly = []
data_fly3 = []

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

    # Initialize
    
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
        
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    
    




    #with st.empty():

    start_time = time.time() 
    #st.session_state.percent_processado = 0
    #st.session_state.controle = 'Controle...'
    #st.session_state.toneladas_passadas = 0 

    url_control = 'http://sia:3000/backend/busca_generica/buscaGenerica?view=MGCLI.AGDTI_VW_DX_BALANCEAMENTO_PH'
    dataset_mega = pd.read_json(url_control)
    tons_tot = dataset_mega['PESO_CONTROLE'][0]
    count = 0
    tons_tot_3 = dataset_mega['PESO_CONTROLE'][0]


    url_percentual_MAF_inicial = 'http://sia:3000/backend/maf/percentuaisCalibre'
    dataset_MAF_inicial = pd.read_json(url_percentual_MAF_inicial)
    controle_maf_inicial = dataset_MAF_inicial['CONTROLE_MEGA'][0]


### Enviar tons_tot - refugo contado nas mensagens

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        count +=1

        current_time = time.time()
        time_passed = current_time - start_time

            
        t1 = time_sync()

        ### SALVO PESO TOT E REFUGO MAF NO BANCO
        ### E A CADA 5 SEGUNDOS EU MANDO A ATUALIZADO DELE
        ### CONSULTANDO SO O BANCO

                

                
        percent_passed = 0
        if time_passed > 70:

            #### 500 é o primeiro minuto mais ou menos de interação

            if count > 680 and count < 1600:


                                      
                conn = sqlite3.connect(r"C:\Users\bernard.collin\Desktop\planilha_denilton\banco_impressora\Chinook.db", timeout=15)
                cursor = conn.cursor()
                
                df_sql_count = pd.read_sql_query("""SELECT * FROM frutos_count """, conn)
                frutos_passados = df_sql_count['frutos_contados'][0]
                frutos_passados = int(frutos_passados)

                # tons_tot 
                # percent_passed 

                percent_passed = df_sql_count['percent_passed'][0]
                tons_passed = df_sql_count['tons_passed'][0]
                
 
                #### VARIAVEIS CONTAGEM 3
                frutos_contados_3 = int(df_sql_count['frutos_contados_3'][0])
                frutos_passados_atual_3 = int(df_sql_count['frutos_passados_atual_3'][0])
                tons_tot_3 = int(df_sql_count['tons_tot_3'][0])
                percent_passed_3 = int(df_sql_count['percent_passed_3'][0])
                tons_passed_3 = int(df_sql_count['tons_passed_3'][0])



                cursor.execute("""
                DELETE FROM frutos_count; 
                """)

            
                count_fly2 = int(count_fly2)

                count_atualizado__22inicio =  int(frutos_passados + count_fly2 + frutos_contados_3)
                refugo_contado = count_atualizado__22inicio

                cursor.execute(f"""
                INSERT INTO frutos_count (Controle, frutos_contados, frutos_passados_atual, tons_tot, percent_passed, tons_passed,
                frutos_contados_3, frutos_passados_atual_3, tons_tot_3, percent_passed_3, tons_passed_3)
                VALUES (0, {count_atualizado__22inicio},{count_fly2}, {tons_tot} ,{percent_passed}, {tons_passed},
                {frutos_contados_3},{frutos_passados_atual_3},{tons_tot_3},{percent_passed_3},{tons_passed_3})
                """)

                conn.commit()
                conn.close()

            

            if count > 1600:
                ## CONSULTAR BASE E ADICIONAR O VALOR GUARDADO
                #st.write('Dados armazenados')

                conn = sqlite3.connect(r"C:\Users\bernard.collin\Desktop\planilha_denilton\banco_impressora\Chinook.db", timeout=15)
                cursor = conn.cursor()
                
                df_sql_count = pd.read_sql_query("""SELECT * FROM frutos_count """, conn)


                frutos_totais = df_sql_count['frutos_contados'][0]
                quantidade_primeira_interacao = df_sql_count['frutos_passados_atual'][0]
                quantidade_primeira_interacao = int(quantidade_primeira_interacao)

                

                percent_passed = df_sql_count['percent_passed'][0]
                tons_passed = df_sql_count['tons_passed'][0]

                #### VARIAVEIS CONTAGEM 3
                frutos_contados_3 = int(df_sql_count['frutos_contados_3'][0])
                frutos_passados_atual_3 = int(df_sql_count['frutos_passados_atual_3'][0])
                tons_tot_3 = int(df_sql_count['tons_tot_3'][0])
                percent_passed_3 = int(df_sql_count['percent_passed_3'][0])
                tons_passed_3 = int(df_sql_count['tons_passed_3'][0])



                frutos_totais = int(frutos_totais)
                count_fly2 = int(count_fly2)
                

                cursor.execute("""
                DELETE FROM frutos_count; 
                """)
                
                
                frutos_passados = abs(quantidade_primeira_interacao - count_fly2)
                frutos_passados = int(frutos_passados)


                count_atualizado =  frutos_totais + frutos_passados + tons_passed_3
                refugo_contado = count_atualizado

                cursor.execute(f"""
                INSERT INTO frutos_count (Controle, frutos_contados, frutos_passados_atual, tons_tot, percent_passed, tons_passed,
                frutos_contados_3, frutos_passados_atual_3, tons_tot_3, percent_passed_3, tons_passed_3)
                VALUES (0, {count_atualizado},{count_fly2}, {tons_tot}, {percent_passed}, {tons_passed},
                {frutos_contados_3},{frutos_passados_atual_3},{tons_tot_3},{percent_passed_3},{tons_passed_3})
                """)

                conn.commit()
                conn.close()



            start_time = current_time

            from retrying import retry


            @retry (wait_fixed = 4000, stop_max_attempt_number = 4)
            def get_maf():
                url_percentual_MAF = 'http://sia:3000/backend/maf/percentuaisCalibre'
                dataset_MAF = pd.read_json(url_percentual_MAF)
                return dataset_MAF

            dataset_MAF = get_maf()

            controle_MAF = dataset_MAF['CONTROLE_MEGA'][0]

            tons_passesed = dataset_MAF['PESO_KG'].sum() / 1000

            variedade_MAF = dataset_MAF['VARIEDADE'][0]
                
            

            dataset_MAF['Calibre'] = dataset_MAF['CALIBRE_QUALIDADE'].str[:3]
            dataset_MAF['Qualidade'] = dataset_MAF['CALIBRE_QUALIDADE'].str[3:]


            def correcao_calibre_MAF(dataset_MAF):
                if dataset_MAF['Calibre'] == 'C05':
                    return 5
                elif dataset_MAF['Calibre'] == 'C04':
                    return 4
                elif dataset_MAF['Calibre'] == 'C06':
                    return 6
                elif dataset_MAF['Calibre'] == 'C07':
                    return 7
                elif dataset_MAF['Calibre'] == 'C08':
                    return 8
                elif dataset_MAF['Calibre'] == 'C09':
                    return 9
                elif dataset_MAF['Calibre'] == 'C10':
                    return 10
                elif dataset_MAF['Calibre'] == 'C12':
                    return 12
                elif dataset_MAF['Calibre'] == 'C14':
                    return 14
                elif dataset_MAF['Calibre'] == 'C16':
                    return 16
                elif dataset_MAF['Calibre'] == 'Ref':
                    return 0

            dataset_MAF['Calibre'] = dataset_MAF.apply(correcao_calibre_MAF, axis = 1)
            dataset_MAF['Calibre'] = dataset_MAF['Calibre'].astype(str)


            dataset_MAF['Calibre_22'] = dataset_MAF['Calibre'].astype(float)


            def ajuste_final(dataset_MAF):
                if dataset_MAF['Calibre'] == '0':
                    return 'Refugo'
                else:
                    return dataset_MAF['Calibre']
            dataset_MAF['Calibre'] = dataset_MAF.apply(ajuste_final, axis = 1)


            dataset_MAF = dataset_MAF.drop(columns = ['CALIBRE_QUALIDADE'])

            def correcao_variedade_maf(dataset_MAF):
                if dataset_MAF['VARIEDADE'] == 'TOMMY':
                    return "Tommy Atkins"
                elif dataset_MAF['VARIEDADE'] == 'TAMMY':
                    return "Tommy Atkins"
                elif dataset_MAF['VARIEDADE'] == 'KEITT':
                    return "Keitt"
                elif dataset_MAF['VARIEDADE'] == 'KENT':
                    return "Kent"
                elif dataset_MAF['VARIEDADE'] == 'PALMER':
                    return "Palmer"
                elif dataset_MAF['VARIEDADE'] == 'OMER':
                    return 'Omer'
                elif dataset_MAF['VARIEDADE'] == 'OSTEEN':
                    return 'Osteen'
            dataset_MAF['VARIEDADE'] = dataset_MAF.apply(correcao_variedade_maf, axis = 1)

            
            somatorio_frutos_peso = pd.pivot_table(dataset_MAF, index = 'Calibre', values = ['QTD_FRUTOS','PESO_KG'],aggfunc= 'sum')
            somatorio_frutos_peso = somatorio_frutos_peso.reset_index()

            somatorio_frutos_peso['Percentual'] = (somatorio_frutos_peso['QTD_FRUTOS'] / somatorio_frutos_peso['QTD_FRUTOS'].sum()) * 100

            filtro_refugo = somatorio_frutos_peso['Calibre'] != 'Refugo'
            somatorio_frutos_peso = somatorio_frutos_peso[filtro_refugo]


            somatorio_frutos_peso = somatorio_frutos_peso[['Calibre','Percentual']]

            refugo_total = refugo_contado + 1

            somatorio_frutos_peso['Frutos_refugo'] = (somatorio_frutos_peso['Percentual'] * refugo_total) / 100
            

            VARIEDADE = dataset_MAF['VARIEDADE'][0]

            ### DEFININDO PESO MEDIO DOS CALIBRES E MULTIPLICANDO PELA QUANTIDADE DE FRUTOS
            def frutos_controle(somatorio_frutos_peso):
                if VARIEDADE == 'Palmer':
                    if somatorio_frutos_peso['Calibre'] == '4':
                        return somatorio_frutos_peso['Frutos_refugo'] * 1.055

                    elif somatorio_frutos_peso['Calibre'] == '5':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.8785

                    elif somatorio_frutos_peso['Calibre'] == '6':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.666

                    elif somatorio_frutos_peso['Calibre'] == '7':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.593

                    elif somatorio_frutos_peso['Calibre'] == '8':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.5175

                    elif somatorio_frutos_peso['Calibre'] == '9':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.458

                    elif somatorio_frutos_peso['Calibre'] == '10':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.407

                    elif somatorio_frutos_peso['Calibre'] == '12':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.3355

                    elif somatorio_frutos_peso['Calibre'] == '14':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.2875

                    elif somatorio_frutos_peso['Calibre'] == '16':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.230


                #################################################### TOMMY ATKINS #####################################################
                elif VARIEDADE == 'Tommy Atkins':
                    if somatorio_frutos_peso['Calibre'] == '4':
                        return somatorio_frutos_peso['Frutos_refugo'] * 1.1

                    elif somatorio_frutos_peso['Calibre'] == '5':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.940

                    elif somatorio_frutos_peso['Calibre'] == '6':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.760

                    elif somatorio_frutos_peso['Calibre'] == '7':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.5985

                    elif somatorio_frutos_peso['Calibre'] == '8':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.5185

                    elif somatorio_frutos_peso['Calibre'] == '9':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.461

                    elif somatorio_frutos_peso['Calibre'] == '10':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.4065

                    elif somatorio_frutos_peso['Calibre'] == '12':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.3335

                    elif somatorio_frutos_peso['Calibre'] == '14':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.2875

                    elif somatorio_frutos_peso['Calibre'] == '16':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.2655
                    
                #################################################### KEITT #####################################################
                elif VARIEDADE == 'Keitt' or VARIEDADE == 'Omer':
                    if somatorio_frutos_peso['Calibre'] == '4':
                        return somatorio_frutos_peso['Frutos_refugo'] * 1.2

                    elif somatorio_frutos_peso['Calibre'] == '5':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.825
                        
                    elif somatorio_frutos_peso['Calibre'] == '6':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.676
                        
                    elif somatorio_frutos_peso['Calibre'] == '7':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.604
                        
                    elif somatorio_frutos_peso['Calibre'] == '8':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.5145
                        
                    elif somatorio_frutos_peso['Calibre'] == '9':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.4575
                        
                    elif somatorio_frutos_peso['Calibre'] == '10':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.412
                        
                    elif somatorio_frutos_peso['Calibre'] == '12':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.345
                        
                    elif somatorio_frutos_peso['Calibre'] == '14':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.292
                        
                    elif somatorio_frutos_peso['Calibre'] == '16':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.256
                    
                    

                #################################################### KENT #####################################################
                elif VARIEDADE == 'Kent':

                    if somatorio_frutos_peso['Calibre'] == '4':
                        return somatorio_frutos_peso['Frutos_refugo'] * 1.115

                    elif  somatorio_frutos_peso['Calibre'] == '5':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.845
                        
                    elif  somatorio_frutos_peso['Calibre'] == '6':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.693
                        
                    elif  somatorio_frutos_peso['Calibre'] == '7':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.5855
                        
                    elif  somatorio_frutos_peso['Calibre'] == '8':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.5105
                        
                    elif  somatorio_frutos_peso['Calibre'] == '9':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.460
                        
                    elif  somatorio_frutos_peso['Calibre'] == '10':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.4095
                        
                    elif  somatorio_frutos_peso['Calibre'] == '12':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.339
                        
                    elif  somatorio_frutos_peso['Calibre'] == '14':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.286
                        
                    elif  somatorio_frutos_peso['Calibre'] == '16':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.2545
                    
                
                #################################################### OSTEEN #####################################################
                elif VARIEDADE == 'Osteen':

                    if somatorio_frutos_peso['Calibre'] == '4':
                        return somatorio_frutos_peso['Frutos_refugo'] * 1.243

                    elif somatorio_frutos_peso['Calibre'] == '5':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.882
                        
                    elif somatorio_frutos_peso['Calibre'] == '6':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.705
                        
                    elif somatorio_frutos_peso['Calibre'] == '7':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.594
                        
                    elif somatorio_frutos_peso['Calibre'] == '8':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.516
                        
                    elif somatorio_frutos_peso['Calibre'] == '9':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.4565
                        
                    elif somatorio_frutos_peso['Calibre'] == '10':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.4045
                        
                    elif somatorio_frutos_peso['Calibre'] == '12':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.337
                        
                    elif somatorio_frutos_peso['Calibre'] == '14':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.2855
                        
                    elif somatorio_frutos_peso['Calibre'] == '16':
                        return somatorio_frutos_peso['Frutos_refugo'] * 0.249

            somatorio_frutos_peso['KGS_REFUGO_CONTROLE'] = somatorio_frutos_peso.apply(frutos_controle, axis = 1)

            somatorio_frutos_peso['KGS_REFUGO_CONTROLE_tons'] = somatorio_frutos_peso['KGS_REFUGO_CONTROLE'] / 1000

            

            tons_count_ref = round(somatorio_frutos_peso['KGS_REFUGO_CONTROLE_tons'].sum(),3)


            ## CRIAR UMA COLUNA [REFUGO CALIBRE] EM SOMATORIO FRUTOS QUE É O COUNT_FLY VS A COLUNA PERCENTE
            ## E DEPOIS ACHO O PESO MÉDIO PASSADO DESSA COLUNA 


            ton_total_passado = (tons_passesed + tons_count_ref)
            #st.session_state.toneladas_passadas = ton_total_passado

            PE_REF = round(tons_tot - tons_count_ref,2)
            
            
            #coluna_4.write(tons_tot)

            controle_p = dataset_mega['CONTROLE'][0].item()

            percent_passed = round((100 * ton_total_passado) / tons_tot,2)

            ### TENHO QUE COLOCAR PERCENT PASSED E COLOCAR NO BANCO


            conn = sqlite3.connect(r"C:\Users\bernard.collin\Desktop\planilha_denilton\banco_impressora\Chinook.db", timeout=15)
            cursor = conn.cursor()
            
            df_sql_count = pd.read_sql_query("""SELECT * FROM frutos_count """, conn)
            count_atualizado = df_sql_count['frutos_contados'][0]


            cursor.execute("""
            DELETE FROM frutos_count; 
            """)
            
            cursor.execute(f"""
            INSERT INTO frutos_count (Controle, frutos_contados, frutos_passados_atual, tons_tot, percent_passed, tons_passed,
            frutos_contados_3, frutos_passados_atual_3, tons_tot_3, percent_passed_3, tons_passed_3)
            VALUES (0, {count_atualizado},{count_fly2}, {tons_tot}, {percent_passed}, {ton_total_passado},
            {frutos_contados_3},{frutos_passados_atual_3},{tons_tot_3},{percent_passed_3},{tons_passed_3})
            """)

            conn.commit()
            conn.close()

            
            #### COLOCAR A CONDICIONAL QUE SO VAI MANDAR SE A SOMA DE FRUTO RECENTE FOR DIFERENTE DE ZERO PRA N ENTRAR NO LOOP

            if (percent_passed > 47 and percent_passed < 51.5) and (dataset_MAF['QTD_FRUTOS_RECENTE'].sum() != 0):

                import requests
                TOKEN = "1730182387:AAEH3SPAeRjbOba3mDPS7A9K9J-rIdFj_Kg"
                chat_id = "-678146203"

                a = controle_MAF
                b = f'{percent_passed}% Processado !'

                horario = time.localtime()
                horario_convert = time.asctime(horario)
                mode = 'Markdown'

                message = f"*Controle:* {a}, {b} \n *Variedade:* {variedade_MAF} \n Refugo estimado:*{tons_count_ref}* \n Entrada - Refugo Estimado :*{PE_REF}* \n *Data:* {horario_convert}"

                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}&parse_mode={mode}"
                print(requests.get(url).json())


            if (percent_passed > 67 and percent_passed < 71) and (dataset_MAF['QTD_FRUTOS_RECENTE'].sum() != 0):

                import requests
                TOKEN = "1730182387:AAEH3SPAeRjbOba3mDPS7A9K9J-rIdFj_Kg"
                chat_id = "-678146203"

                a = controle_MAF
                b = f'{percent_passed}% Processado !'

                horario = time.localtime()
                horario_convert = time.asctime(horario)
                mode = 'Markdown'

                message = f"*Controle:* {a}, {b} \n *Variedade:* {variedade_MAF} \n Refugo estimado:*{tons_count_ref}* \n Entrada - Refugo Estimado :*{PE_REF}* \n *Data:* {horario_convert}"

                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}&parse_mode={mode}"
                print(requests.get(url).json())
                

            if (percent_passed > 85 and percent_passed < 90) and (dataset_MAF['QTD_FRUTOS_RECENTE'].sum() != 0) :

                import requests
                TOKEN = "1730182387:AAEH3SPAeRjbOba3mDPS7A9K9J-rIdFj_Kg"
                chat_id = "-678146203"

                a = controle_MAF
                b = f'{percent_passed}% Processado !'

                horario = time.localtime()
                horario_convert = time.asctime(horario)
                mode = 'Markdown'

                message = f"*Controle:* {a}, {b} \n *Variedade:* {variedade_MAF} \n Refugo estimado:*{tons_count_ref}* \n Entrada - Refugo Estimado :*{PE_REF}* \n *Data:* {horario_convert}"

                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}&parse_mode={mode}"
                print(requests.get(url).json())

            if (percent_passed >= 90 and percent_passed < 100) and (dataset_MAF['QTD_FRUTOS_RECENTE'].sum() != 0):

                import requests
                TOKEN = "1730182387:AAEH3SPAeRjbOba3mDPS7A9K9J-rIdFj_Kg"
                chat_id = "-678146203"

                a = controle_MAF
                b = f'{percent_passed}% Processado !'

                horario = time.localtime()
                horario_convert = time.asctime(horario)
                mode = 'Markdown'

                message = f"*Controle:* {a}, {b} \n *Variedade:* {variedade_MAF} \n Refugo estimado:*{tons_count_ref}* \n  Entrada - Refugo Estimado :*{PE_REF}* \n *Data:* {horario_convert}"

                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}&parse_mode={mode}"
                print(requests.get(url).json())

            if percent_passed > 102:

                url_control = 'http://sia:3000/backend/busca_generica/buscaGenerica?view=MGCLI.AGDTI_VW_DX_BALANCEAMENTO_PH'
                dataset_mega = pd.read_json(url_control)
                tons_tot = dataset_mega['PESO_CONTROLE'][0]

                conn = sqlite3.connect(r"C:\Users\bernard.collin\Desktop\planilha_denilton\banco_impressora\Chinook.db", timeout=15)
                cursor = conn.cursor()
                
                df_sql_count = pd.read_sql_query("""SELECT * FROM frutos_count """, conn)


                frutos_totais = df_sql_count['frutos_contados'][0]
                quantidade_primeira_interacao = df_sql_count['frutos_passados_atual'][0]
                quantidade_primeira_interacao = int(quantidade_primeira_interacao)

                

                percent_passed = df_sql_count['percent_passed'][0]
                tons_passed = df_sql_count['tons_passed'][0]

                #### VARIAVEIS CONTAGEM 3
                frutos_contados_3 = int(df_sql_count['frutos_contados_3'][0])
                frutos_passados_atual_3 = int(df_sql_count['frutos_passados_atual_3'][0])
                tons_tot_3 = int(df_sql_count['tons_tot_3'][0])
                percent_passed_3 = int(df_sql_count['percent_passed_3'][0])
                tons_passed_3 = int(df_sql_count['tons_passed_3'][0])

                frutos_totais = int(frutos_totais)
                count_fly2 = int(count_fly2)
                

                cursor.execute("""
                DELETE FROM frutos_count; 
                """)
                
                
                frutos_passados = abs(quantidade_primeira_interacao - count_fly2)
                frutos_passados = int(frutos_passados)


                count_atualizado =  frutos_totais + frutos_passados + tons_passed_3
                refugo_contado = count_atualizado

                cursor.execute(f"""
                INSERT INTO frutos_count (Controle, frutos_contados, frutos_passados_atual, tons_tot, percent_passed, tons_passed,
                frutos_contados_3, frutos_passados_atual_3, tons_tot_3, percent_passed_3, tons_passed_3)
                VALUES (0, {count_atualizado},{count_fly2}, {tons_tot}, {percent_passed}, {tons_passed},
                {frutos_contados_3},{frutos_passados_atual_3},{tons_tot_3},{percent_passed_3},{tons_passed_3})
                """)

                conn.commit()
                conn.close()

            
            if controle_MAF != controle_maf_inicial:

                    def reset_count():
                        global count_fly, count_fly3, count_fly4
                        count_fly = 0
                        count_fly3 = 0
                        count_fly4 = 0
                    reset_count()

                    conn = sqlite3.connect(r"C:\Users\bernard.collin\Desktop\planilha_denilton\banco_impressora\Chinook.db", timeout=15)
                    cursor = conn.cursor()

                    ### TRAGO AQUI O NOVO COUNT
                    
                    cursor.execute("""
                    DELETE FROM frutos_count; 
                    """)

                    frutos_reset =  0
                    controle_reset = 0 
                    cont_reset = 0
                    tons_tot_reset = 0
                    percent_passed = 0
                    tons_passed = 0

                    ### INSIRO AQUI O NOVO COUNT
                    ### E ESTE COUNT EU DEFINO COMO O NOVO COUNTFLY

                    cursor.execute(f"""
                    INSERT INTO frutos_count (Controle, frutos_contados, frutos_passados_atual, tons_tot, percent_passed, tons_passed,
                    frutos_contados_3, frutos_passados_atual_3, tons_tot_3, percent_passed_3, tons_passed_3)
                    VALUES ({controle_reset}, {frutos_reset},{cont_reset}, {tons_tot_reset}, {percent_passed},{tons_passed},
                    {frutos_contados_3},{frutos_passados_atual_3},{tons_tot_3},{percent_passed_3},{tons_passed_3})
                    """)

                    conn.commit()
                    conn.close() 


                    url_control = 'http://sia:3000/backend/busca_generica/buscaGenerica?view=MGCLI.AGDTI_VW_DX_BALANCEAMENTO_PH'
                    dataset_mega = pd.read_json(url_control)
                    tons_tot = dataset_mega['PESO_CONTROLE'][0]
                    crtl_mega = dataset_mega['CONTROLE'][0]
                    vrdd_mega = dataset_mega['VARIEDADE'][0]
                    talhao_mega = dataset_mega['TALHAO'][0]

                    url_percentual_MAF_inicial = 'http://sia:3000/backend/maf/percentuaisCalibre'
                    dataset_MAF_inicial = pd.read_json(url_percentual_MAF_inicial)
                    controle_maf_inicial = dataset_MAF_inicial['CONTROLE_MEGA'][0]        
                    mode = 'Markdown'
                   
                    
                    horario = time.localtime()
                    horario_convert = time.asctime(horario)
                    
                    TOKEN = "1730182387:AAEH3SPAeRjbOba3mDPS7A9K9J-rIdFj_Kg"
                    chat_id = "-678146203"

                    
                    message = f"*Próximo controle (MEGA):* {crtl_mega} \n *Controle MAF atual:* {controle_maf_inicial} \n *Variedade:* {vrdd_mega} \n *Talhão:* {talhao_mega} \n *Data:* {horario_convert}"
                    import requests
                    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}&parse_mode={mode}"
                    print(requests.get(url).json())
      


                ### AQUI AGORA EU TENHO QUE ATRIBUIR O COUNT PARA O PROXIMO CONTROLE
                ### OU SEJA GUARDO NO BANCO ESSA NOVA VARIAVEL
                ### E QUANDO EU RODAR DNV NA PRIMEIRA CONSULTA DO BANCO DESSE CONTROLE EU CONSULTO ESSA VARIAVEL E ZERO ELA
            
       #### MANGA CORTADA TENHO QUE VERIFICAR A CADA INTERAÇÃO DO RT PARA N PERDER O TIME DO AVISO
                ### AQUI AGORA EU TENHO QUE ATRIBUIR O COUNT PARA O PROXIMO CONTROLE
                ### OU SEJA GUARDO NO BANCO ESSA NOVA VARIAVEL
                ### E QUANDO EU RODAR DNV NA PRIMEIRA CONSULTA DO BANCO DESSE CONTROLE EU CONSULTO ESSA VARIAVEL E ZERO ELA

                ## OU SEJA< PRIMEIRO TENHO QUE ZERAR COUNT
                ## E CHAMAR COUNT NOVO ATE A MAF MUDAR

        
        
## FAZER UM IF DE SE FRUTOS CONTADOIS FOR ZERO EU UTILIZO A LOGICA QUE DEU CERTO DE INICIO
## SE ESSE VALOR NAO FOR ZERO, EU UTILIZO A OUTRA LOGICA DE COLOCAR SO O COUNT FLY

            
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3
        
        # tempo_passado = current_time - start_time
        
        # Process detections
        
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            w, h = im0.shape[1], im0.shape[0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

            
                # pass detections to deepsort
                t4 = time_sync()

                
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                t5 = time_sync()
                dt[3] += t5 - t4
                
                # draw boxes for visualization

                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        classe = output[5]
                        #count
                        count_obj(bboxes,w,h,id,classe)
                        c = int(classe)  # integer class
                        label = f'{id}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

                fps = 1 / (t3 - t2)
                fps2 = 1 / (t5 - t4)

                LOGGER.info(f'{s}FPS_yolo: {fps:.3f}s')
                LOGGER.info(f'{s}FPS_track: {fps2:.3f}s')
                LOGGER.info(f'{s}TEMPO_PASSADO: {time_passed:.3f}s')
                LOGGER.info(f'{s}CONTADOR: {count:.3f}s')
                
                
    
                
                
                
    

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')
                LOGGER.info(f'{s}TEMPO_PASSADO: {time_passed:.3f}s')
                LOGGER.info(f'{s}CONTADOR: {count:.3f}s')

            
            im0 = annotator.result()


            if show_vid:
                global count_fly, count_fly3
                color = (0,255,0)

                start_point = (0, h-100)
                end_point = (w, h-100)

                cv2.line(im0, start_point, end_point, color, thickness=2)

                thickness = 3
                org = (50,50)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1

                color2 = (0,255,255)
                fontScale2 = 1
                org2 = (50,100)

                cv2.putText(im0, str(count_fly) + ' ' + 'Mangas' , org, font,
                    fontScale,color,thickness, cv2.LINE_AA)


                cv2.putText(im0, str(count_fly3) + ' ' +  'Caixa:', org2, font,
                  fontScale2,color2,thickness, cv2.LINE_AA)
            #  cv2.putText(im0, 'Flyss:' + str(count_fly), org, font,
                #  fontScale,color,thickness, cv2.LINE_AA)
            #   cv2.putText(im0, 'Flys2:' + str(count_fly2), org, font,
                #  fontScale,color,thickness, cv2.LINE_AA)


                cv2.imshow(str(p), im0)
                
   

                #percent_passed = st.session_state.percent_processado 
                #controle_p = st.session_state.controle

                count_fly2 = count_fly + 0
                count_fly4 = count_fly3 + 0
   
                
                if cv2.waitKey(1) == ord('q'):  # q to quit  
                    #st.stop()                              
                    raise StopIteration 
                                           
                                    
                # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                vid_writer.write(im0)
            

            #data2 = requests.get("http://177.52.21.58:3000/backend/maf/#percentuaisCalibre")
            #json_data2 = data2.json()
            # dataset_MAF = pd.DataFrame.from_dict(json_data2)
                    
            #st.write(dataset_MAF)

        
        if time_passed > 10:
            if count_fly4 >= 1:

                import requests
                TOKEN = "1730182387:AAEH3SPAeRjbOba3mDPS7A9K9J-rIdFj_Kg"
                chat_id = "-678146203"


                @retry (wait_fixed = 4000, stop_max_attempt_number = 4)
                def get_maf():
                    url_percentual_MAF = 'http://sia:3000/backend/maf/percentuaisCalibre'
                    dataset_MAF = pd.read_json(url_percentual_MAF)
                    return dataset_MAF
                dataset_MAF = get_maf()
                controle_MAF = dataset_MAF['CONTROLE_MEGA'][0]
                a = controle_MAF

                horario = time.localtime()
                horario_convert = time.asctime(horario)
                tons_passesed = dataset_MAF['PESO_KG'].sum() / 1000
                variedade_MAF = dataset_MAF['VARIEDADE'][0]

                dataset_MAF['Calibre'] = dataset_MAF['CALIBRE_QUALIDADE'].str[:3]
                dataset_MAF['Qualidade'] = dataset_MAF['CALIBRE_QUALIDADE'].str[3:]


                def correcao_calibre_MAF(dataset_MAF):
                    if dataset_MAF['Calibre'] == 'C05':
                        return 5
                    elif dataset_MAF['Calibre'] == 'C04':
                        return 4
                    elif dataset_MAF['Calibre'] == 'C06':
                        return 6
                    elif dataset_MAF['Calibre'] == 'C07':
                        return 7
                    elif dataset_MAF['Calibre'] == 'C08':
                        return 8
                    elif dataset_MAF['Calibre'] == 'C09':
                        return 9
                    elif dataset_MAF['Calibre'] == 'C10':
                        return 10
                    elif dataset_MAF['Calibre'] == 'C12':
                        return 12
                    elif dataset_MAF['Calibre'] == 'C14':
                        return 14
                    elif dataset_MAF['Calibre'] == 'C16':
                        return 16
                    elif dataset_MAF['Calibre'] == 'Ref':
                        return 0

                dataset_MAF['Calibre'] = dataset_MAF.apply(correcao_calibre_MAF, axis = 1)
                dataset_MAF['Calibre'] = dataset_MAF['Calibre'].astype(str)


                dataset_MAF['Calibre_22'] = dataset_MAF['Calibre'].astype(float)


                def ajuste_final(dataset_MAF):
                    if dataset_MAF['Calibre'] == '0':
                        return 'Refugo'
                    else:
                        return dataset_MAF['Calibre']
                dataset_MAF['Calibre'] = dataset_MAF.apply(ajuste_final, axis = 1)

                dataset_MAF = dataset_MAF.drop(columns = ['CALIBRE_QUALIDADE'])

                def correcao_variedade_maf(dataset_MAF):
                    if dataset_MAF['VARIEDADE'] == 'TOMMY':
                        return "Tommy Atkins"
                    elif dataset_MAF['VARIEDADE'] == 'TAMMY':
                        return "Tommy Atkins"
                    elif dataset_MAF['VARIEDADE'] == 'KEITT':
                        return "Keitt"
                    elif dataset_MAF['VARIEDADE'] == 'KENT':
                        return "Kent"
                    elif dataset_MAF['VARIEDADE'] == 'PALMER':
                        return "Palmer"
                    elif dataset_MAF['VARIEDADE'] == 'OMER':
                        return 'Omer'
                    elif dataset_MAF['VARIEDADE'] == 'OSTEEN':
                        return 'Osteen'
                dataset_MAF['VARIEDADE'] = dataset_MAF.apply(correcao_variedade_maf, axis = 1)
   
                somatorio_frutos_peso = pd.pivot_table(dataset_MAF, index = 'Calibre', values = ['QTD_FRUTOS','PESO_KG'],aggfunc= 'sum')
                somatorio_frutos_peso = somatorio_frutos_peso.reset_index()
                somatorio_frutos_peso['Percentual'] = (somatorio_frutos_peso['QTD_FRUTOS'] / somatorio_frutos_peso['QTD_FRUTOS'].sum()) * 100

                filtro_refugo = somatorio_frutos_peso['Calibre'] != 'Refugo'
                somatorio_frutos_peso = somatorio_frutos_peso[filtro_refugo]

                somatorio_frutos_peso = somatorio_frutos_peso[['Calibre','Percentual']]

                refugo_total = refugo_contado + 1

                somatorio_frutos_peso['Frutos_refugo'] = (somatorio_frutos_peso['Percentual'] * refugo_total) / 100
                
                VARIEDADE = dataset_MAF['VARIEDADE'][0]

                ### DEFININDO PESO MEDIO DOS CALIBRES E MULTIPLICANDO PELA QUANTIDADE DE FRUTOS
                def frutos_controle(somatorio_frutos_peso):
                    if VARIEDADE == 'Palmer':
                        if somatorio_frutos_peso['Calibre'] == '4':
                            return somatorio_frutos_peso['Frutos_refugo'] * 1.055

                        elif somatorio_frutos_peso['Calibre'] == '5':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.8785

                        elif somatorio_frutos_peso['Calibre'] == '6':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.666

                        elif somatorio_frutos_peso['Calibre'] == '7':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.593

                        elif somatorio_frutos_peso['Calibre'] == '8':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.5175

                        elif somatorio_frutos_peso['Calibre'] == '9':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.458

                        elif somatorio_frutos_peso['Calibre'] == '10':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.407

                        elif somatorio_frutos_peso['Calibre'] == '12':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.3355

                        elif somatorio_frutos_peso['Calibre'] == '14':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.2875

                        elif somatorio_frutos_peso['Calibre'] == '16':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.230

                   #################################################### TOMMY ATKINS #####################################################
                    elif VARIEDADE == 'Tommy Atkins':
                        if somatorio_frutos_peso['Calibre'] == '4':
                            return somatorio_frutos_peso['Frutos_refugo'] * 1.1

                        elif somatorio_frutos_peso['Calibre'] == '5':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.940

                        elif somatorio_frutos_peso['Calibre'] == '6':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.760

                        elif somatorio_frutos_peso['Calibre'] == '7':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.5985

                        elif somatorio_frutos_peso['Calibre'] == '8':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.5185

                        elif somatorio_frutos_peso['Calibre'] == '9':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.461

                        elif somatorio_frutos_peso['Calibre'] == '10':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.4065

                        elif somatorio_frutos_peso['Calibre'] == '12':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.3335

                        elif somatorio_frutos_peso['Calibre'] == '14':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.2875

                        elif somatorio_frutos_peso['Calibre'] == '16':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.2655
                        
                    #################################################### KEITT #####################################################
                    elif VARIEDADE == 'Keitt' or VARIEDADE == 'Omer':
                        if somatorio_frutos_peso['Calibre'] == '4':
                            return somatorio_frutos_peso['Frutos_refugo'] * 1.2

                        elif somatorio_frutos_peso['Calibre'] == '5':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.825
                            
                        elif somatorio_frutos_peso['Calibre'] == '6':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.676
                            
                        elif somatorio_frutos_peso['Calibre'] == '7':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.604
                            
                        elif somatorio_frutos_peso['Calibre'] == '8':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.5145
                            
                        elif somatorio_frutos_peso['Calibre'] == '9':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.4575
                            
                        elif somatorio_frutos_peso['Calibre'] == '10':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.412
                            
                        elif somatorio_frutos_peso['Calibre'] == '12':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.345
                            
                        elif somatorio_frutos_peso['Calibre'] == '14':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.292
                            
                        elif somatorio_frutos_peso['Calibre'] == '16':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.256
                        
                    #################################################### KENT #####################################################
                    elif VARIEDADE == 'Kent':

                        if somatorio_frutos_peso['Calibre'] == '4':
                            return somatorio_frutos_peso['Frutos_refugo'] * 1.115

                        elif  somatorio_frutos_peso['Calibre'] == '5':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.845
                            
                        elif  somatorio_frutos_peso['Calibre'] == '6':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.693
                            
                        elif  somatorio_frutos_peso['Calibre'] == '7':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.5855
                            
                        elif  somatorio_frutos_peso['Calibre'] == '8':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.5105
                            
                        elif  somatorio_frutos_peso['Calibre'] == '9':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.460
                            
                        elif  somatorio_frutos_peso['Calibre'] == '10':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.4095
                            
                        elif  somatorio_frutos_peso['Calibre'] == '12':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.339
                            
                        elif  somatorio_frutos_peso['Calibre'] == '14':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.286
                            
                        elif  somatorio_frutos_peso['Calibre'] == '16':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.2545
                                  
                    #################################################### OSTEEN #####################################################
                    elif VARIEDADE == 'Osteen':

                        if somatorio_frutos_peso['Calibre'] == '4':
                            return somatorio_frutos_peso['Frutos_refugo'] * 1.243

                        elif somatorio_frutos_peso['Calibre'] == '5':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.882
                            
                        elif somatorio_frutos_peso['Calibre'] == '6':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.705
                            
                        elif somatorio_frutos_peso['Calibre'] == '7':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.594
                            
                        elif somatorio_frutos_peso['Calibre'] == '8':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.516
                            
                        elif somatorio_frutos_peso['Calibre'] == '9':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.4565
                            
                        elif somatorio_frutos_peso['Calibre'] == '10':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.4045
                            
                        elif somatorio_frutos_peso['Calibre'] == '12':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.337
                            
                        elif somatorio_frutos_peso['Calibre'] == '14':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.2855
                            
                        elif somatorio_frutos_peso['Calibre'] == '16':
                            return somatorio_frutos_peso['Frutos_refugo'] * 0.249

                somatorio_frutos_peso['KGS_REFUGO_CONTROLE'] = somatorio_frutos_peso.apply(frutos_controle, axis = 1)

                somatorio_frutos_peso['KGS_REFUGO_CONTROLE_tons'] = somatorio_frutos_peso['KGS_REFUGO_CONTROLE'] / 1000

            
                tons_count_ref = somatorio_frutos_peso['KGS_REFUGO_CONTROLE_tons'].sum()

                ## CRIAR UMA COLUNA [REFUGO CALIBRE] EM SOMATORIO FRUTOS QUE É O COUNT_FLY VS A COLUNA PERCENTE
                ## E DEPOIS ACHO O PESO MÉDIO PASSADO DESSA COLUNA 


                ton_total_passado = (tons_passesed + tons_count_ref)
                #st.session_state.toneladas_passadas = ton_total_passado

                PE_REF = round(tons_tot - tons_count_ref,2)

                    ### POSSO CONSULTAR BASE PRA TRAZER O ENTRADA-PESO
                    ### PARA ISSO TENHO QUE ARMAZENAR ELA


                mode = 'Markdown'
                message = f" CONTROLE: {a} \n *CAIXA IDENTIFICADA !!!* \n ENTRADA - COUNT: {PE_REF}"

                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}&parse_mode={mode}"
                print(requests.get(url).json())

                def reset_count():
                    global count_fly3, count_fly4 
                    count_fly3 = 0
                    count_fly4 = 0
                    
                reset_count()



    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    

    
    

    
    # import subprocess
    # cmdCommand = "python track_2.py"   #specify your cmd command
    # process = subprocess.Popen(cmdCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    # print (output)

    
    
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

#def count_obj(box,w,h,id):
    #global count,data 
    #center_coordinates = (int(box[0]+(box[2]-box[0])/2), int(box[1]+(box[3]-box[1])/2))
    #if int(box[1]+(box[3]-box[1])/2) > (h - 350):
        #if id not in data:
        # count += 1
        # data.append(id)
    

def count_obj(box,w,h,id,classe):
    global count_fly,data_fly, count_fly3 , data_fly3
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    
    if  classe == 1:
        if int(box[1]+(box[3]-box[1])/2) > (h-100):
            if id not in data_fly:
                    count_fly += 1
                    data_fly.append(id)

    if  classe == 0:
        if int(box[1]+(box[3]-box[1])/2) > (h-100):
            if id not in data_fly3:
                    count_fly3 += 1
                    data_fly3.append(id)
                    
                                        
# try:
    #with st.empty():
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='best_caixa2.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='rtsp://admin:admin102030@192.168.3.92:554/cam/realmonitor?channel=1&subtype=1', help='source') 

    #'rtsp://admin:admin102030@192.168.3.92:554/cam/realmonitor?channel=1&subtype=1'
    ### file/folder, 0 for webcam 'videos/video_controle_450_keitt.mp4' 
    ### link camera1: 'rtsp://admin:admin102030@192.168.3.92:554/cam/realmonitor?channel=1&subtype=0'
    ### link youtube fuciona tambem
    ### rtsp://localhost:554/live

    ## TODA VEZ QUE REINICIAR O APP TENHO QUE CALIOBRAR COM A WEBCAM ANTES E AI RODAR COM A LOCAL

#### DEIXAR EM 640 E ABRIR WEBCAM E DPS CONECTA O VIDEO

### A CAMERA FICA NO 640
## IDEAL É O 480x640 com 14 fps

    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.85, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results',default = True)
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results', default = True)
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    #parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--classes', default=[0,1], type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
# except NameError:
#     print('Opaa')

