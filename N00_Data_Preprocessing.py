#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
###################################################################################################################
###################################################################################################################
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if __name__ == "__main__":
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")
#--------------------------------------------------#
    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")
#--------------------------------------------------#
    if "__file__" in globals().keys():
        print('CurrentDir: ', os.getcwd())
        try:
            os.chdir(os.path.dirname(__file__))
        except:
            print("Problems with navigating to the file dir.")
        print('CurrentDir: ', os.getcwd())
    else:
        print("Running in python jupyter notebook.")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
#--------------------------------------------------#
#########################################################################################################
#########################################################################################################
#--------------------------------------------------#
# import
import ast
import copy
import time
import scipy
import random
import pickle
import scipy.io
import argparse
import subprocess
import numpy as np
import pandas as pd

#--------------------------------------------------#
# 
from numpy import *
from tqdm import tqdm
from pathlib import Path
from random import shuffle

#--------------------------------------------------#
# 
from pypdb import *



###################################################################################################################
###################################################################################################################
# Print the DataFrame obtained.
def beautiful_print(df):
    # Print the dataset in a well-organized format.
    with pd.option_context('display.max_rows', 20, 
                           'display.min_rows', 20, 
                           'display.max_columns', 6, 
                           #"display.max_colwidth", None,
                           "display.width", None,
                           "expand_frame_repr", True,
                           "max_seq_items", None,):  # more options can be specified
        # Once the display.max_rows is exceeded, 
        # the display.min_rows options determines 
        # how many rows are shown in the truncated repr.
        print(df)
    return 


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#              `7MM"""Yb.      db  MMP""MM""YMM  db     `7MM"""YMM `7MM"""Mq.       db     `7MMM.     ,MMF'`7MM"""YMM                           M      #
#   __,          MM    `Yb.   ;MM: P'   MM   `7 ;MM:      MM    `7   MM   `MM.     ;MM:      MMMb    dPMM    MM    `7                           M      #
#  `7MM          MM     `Mb  ,V^MM.     MM     ,V^MM.     MM   d     MM   ,M9     ,V^MM.     M YM   ,M MM    MM   d                             M      #
#    MM          MM      MM ,M  `MM     MM    ,M  `MM     MM""MM     MMmmdM9     ,M  `MM     M  Mb  M' MM    MMmmMM                         `7M'M`MF'  #
#    MM          MM     ,MP AbmmmqMA    MM    AbmmmqMA    MM   Y     MM  YM.     AbmmmqMA    M  YM.P'  MM    MM   Y  ,                        VAM,V    #
#    MM  ,,      MM    ,dP'A'     VML   MM   A'     VML   MM         MM   `Mb.  A'     VML   M  `YM'   MM    MM     ,M                         VVV     #
#  .JMML.db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.JMML.     .JMML. .JMM.AMA.   .AMMA.JML. `'  .JMML..JMMmmmmMMM                          V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#  
# Import data and obtain a DataFrame.
def Get_processed_data_df(data_folder, data_file, data_file_binary, binary_class_bool, y_prpty_cls_threshold, target_nme, max_seqs_len):
    #====================================================================================================#
    processed_data_df = pd.read_csv(data_folder / data_file, index_col=0, header=0)


    if type(target_nme) == str:
        processed_data_df_1 = processed_data_df.groupby('SEQ').agg( 
                                val_avg = pd.NamedAgg(column = target_nme, aggfunc = 'mean') , 
                                val_SD  = pd.NamedAgg(column = target_nme, aggfunc = 'std' ) ,
                                                                                            )
        #beautiful_print(processed_data_df_1)
        processed_data_df_1["val_SD"] = processed_data_df_1["val_SD"].fillna(value=0)                                                    
        processed_data_df_1.reset_index(inplace = True)
        processed_data_df_2 = processed_data_df_1[ np.isnan(processed_data_df_1['val_avg']) == False]
        #beautiful_print(processed_data_df_2)
        processed_data_df = pd.merge(processed_data_df, processed_data_df_2, on = ['SEQ'], how = 'left')
        processed_data_df.drop(columns = [target_nme, ], inplace = True)
        processed_data_df = processed_data_df.dropna(subset = ['val_avg'])
        processed_data_df.rename(columns={'val_avg': target_nme}, inplace = True)
        processed_data_df.drop_duplicates(subset = ['SEQ'], keep = 'first', inplace = True)
        processed_data_df.reset_index(inplace = True)
        processed_data_df = processed_data_df[['SEQ', target_nme]]
        #beautiful_print(processed_data_df)

    
    
    # Remove sequences that are too long (remove those longer than max_seqs_len)
    processed_data_df["seqs_length"] = processed_data_df.SEQ.str.len()
    processed_data_df = processed_data_df[processed_data_df.seqs_length <= max_seqs_len]
    processed_data_df.reset_index(drop = True, inplace = True)

    #====================================================================================================#
    # Current version doesnt support binary classifica
    if binary_class_bool and ((data_folder / data_file_binary).exists()):
        processed_data_df_bi = pd.read_csv(data_folder / data_file_binary, index_col=0, header=0)

    else:
        print("binary classification file does not exist.")
        processed_data_df_bi = pd.read_csv(data_folder / data_file, index_col=0, header=0)

        if type(target_nme) == dict or type(target_nme) == list:
            for one_target in target_nme:
                processed_data_df_bi[one_target] = [1 if one_cvsn>y_prpty_cls_threshold else 0 
                                                      for one_cvsn in list(processed_data_df_bi[one_target])]
        elif type(target_nme) == str:
            processed_data_df_bi[target_nme] = [1 if one_cvsn>y_prpty_cls_threshold else 0 
                                                  for one_cvsn in list(processed_data_df_bi[target_nme])]
    #--------------------------------------------------#
    # Remove sequences that are too long (remove those longer than max_seqs_len)
    processed_data_df_bi["seqs_length"] = processed_data_df_bi.SEQ.str.len()
    processed_data_df_bi = processed_data_df_bi[processed_data_df_bi.seqs_length <= max_seqs_len]
    processed_data_df_bi.reset_index(drop = True, inplace = True)

    
    return processed_data_df, processed_data_df_bi


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                  `7MMF'     A     `7MF'`7MM"""Mq.  `7MMF'MMP""MM""YMM `7MM"""YMM      `7MM"""YMM    db       .M"""bgd MMP""MM""YMM   db      
#                    `MA     ,MA     ,V    MM   `MM.   MM  P'   MM   `7   MM    `7        MM    `7   ;MM:     ,MI    "Y P'   MM   `7  ;MM:     
#   pd*"*b.           VM:   ,VVM:   ,V     MM   ,M9    MM       MM        MM   d          MM   d    ,V^MM.    `MMb.          MM      ,V^MM.    
#  (O)   j8            MM.  M' MM.  M'     MMmmdM9     MM       MM        MMmmMM          MM""MM   ,M  `MM      `YMMNq.      MM     ,M  `MM    
#      ,;j9            `MM A'  `MM A'      MM  YM.     MM       MM        MM   Y  ,       MM   Y   AbmmmqMA   .     `MM      MM     AbmmmqMA   
#   ,-='    ,,          :MM;    :MM;       MM   `Mb.   MM       MM        MM     ,M       MM      A'     VML  Mb     dM      MM    A'     VML  
#  Ammmmmmm db           VF      VF      .JMML. .JMM..JMML.   .JMML.    .JMMmmmmMMM     .JMML..  AMA.   .AMMA.P"Ybmmd"     .JMML..AMA.   .AMMA.
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Output #1: Write a fasta file including all sequences
def output_fasta(processed_data_df, output_folder, output_file_0):
    # Also get a seqs_list including all sequences
    processed_data_df_row_num = processed_data_df.shape[0]
    with open(output_folder / output_file_0 , 'w') as f:
        count_x=0
        seqs_list=[]
        max_len = 0
        print("processed_data_df_row_num: ", processed_data_df_row_num)
        for i in range(processed_data_df_row_num):
            one_seq = (processed_data_df.loc[i,"SEQ"]).replace("-", "")
            max_len = len(one_seq) if len(one_seq)>max_len else max_len
            if one_seq not in seqs_list and len(one_seq)<=max_seqs_len:
                seqs_list.append(one_seq)
                count_x+=1
                if len(one_seq) <= 1024-2:
                    f.write(">seq"+str(count_x)+"\n")
                    f.write(one_seq.upper()+"\n")
                else:
                    f.write(">seq"+str(count_x)+"\n")
                    f.write(one_seq.upper()[0 : 1024-2]+"\n")
    print("number of seqs: ", len(seqs_list))
    print("number of seqs duplicates removed: ", len(set(seqs_list)))
    return seqs_list





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#       ,AM            .g8""8q. `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq.`7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                                       M      #
#      AVMM          .dP'    `YM. MM       M  P'   MM   `7   MM   `MM. MM       M  P'   MM   `7 ,MI    "Y                                       M      #
#    ,W' MM          dM'      `MM MM       M       MM        MM   ,M9  MM       M       MM      `MMb.                                           M      #
#  ,W'   MM          MM        MM MM       M       MM        MMmmdM9   MM       M       MM        `YMMNq.                                   `7M'M`MF'  #
#  AmmmmmMMmm        MM.      ,MP MM       M       MM        MM        MM       M       MM      .     `MM                                     VAM,V    #
#        MM   ,,     `Mb.    ,dP' YM.     ,M       MM        MM        YM.     ,M       MM      Mb     dM                                      VVV     #
#        MM   db       `"bmmd"'    `bmmmmd"'     .JMML.    .JMML.       `bmmmmd"'     .JMML.    P"Ybmmd"                                        V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def output_formated_dataset(processed_data_df, 
                            processed_data_df_bi  , 
                            seqs_list             , 
                            y_prpty_cls_threshold , 
                            target_nme            , 
                            data_folder           , 
                            output_folder         , 
                            output_file_1         ,
                            ):
    #--------------------------------------------------#
    # Initialize.
    seq_prpty_dict = dict([])
    #--------------------------------------------------#
    # 
    if type(target_nme) == dict or type(target_nme) == list:
        seq_prpty_dict["sequence"] = seqs_list
        for one_header in target_nme:
            seq_prpty_dict[one_header] = processed_data_df[one_header]

    #print(np.isnan(processed_data_df.loc[17, "FC1"]))
    #print(seq_prpty_dict["FC1"][18])

    pickle.dump( seq_prpty_dict, open( output_folder / output_file_1, "wb" ) )

    return seq_prpty_dict



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#`7MM"""Yb.      db  MMP""MM""YMM  db       `7MM"""Mq.`7MM"""Mq.   .g8""8q.     .g8"""bgd `7MM"""YMM   .M"""bgd  .M"""bgd `7MMF'`7MN.   `7MF' .g8"""bgd 
#  MM    `Yb.   ;MM: P'   MM   `7 ;MM:        MM   `MM. MM   `MM..dP'    `YM. .dP'     `M   MM    `7  ,MI    "Y ,MI    "Y   MM    MMN.    M .dP'     `M 
#  MM     `Mb  ,V^MM.     MM     ,V^MM.       MM   ,M9  MM   ,M9 dM'      `MM dM'       `   MM   d    `MMb.     `MMb.       MM    M YMb   M dM'       ` 
#  MM      MM ,M  `MM     MM    ,M  `MM       MMmmdM9   MMmmdM9  MM        MM MM            MMmmMM      `YMMNq.   `YMMNq.   MM    M  `MN. M MM          
#  MM     ,MP AbmmmqMA    MM    AbmmmqMA      MM        MM  YM.  MM.      ,MP MM.           MM   Y  , .     `MM .     `MM   MM    M   `MM.M MM.    `7MMF
#  MM    ,dP'A'     VML   MM   A'     VML     MM        MM   `Mb.`Mb.    ,dP' `Mb.     ,'   MM     ,M Mb     dM Mb     dM   MM    M     YMM `Mb.     MM 
#.JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA. .JMML.    .JMML. .JMM. `"bmmd"'     `"bmmmd'  .JMMmmmmMMM P"Ybmmd"  P"Ybmmd"  .JMML..JML.    YM   `"bmmmdPY 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def N00_Data_Processing(binary_class_bool      ,
                        data_folder            ,
                        target_nme             , 
                        data_file              ,
                        data_file_binary       ,
                        max_seqs_len           ,
                        y_prpty_cls_threshold  ,
                        output_folder          ,
                        output_file_0          ,
                        output_file_1          ,
                        ):
    # Process target_nme



    # Get_processed_data_df
    processed_data_df, processed_data_df_bi = Get_processed_data_df(data_folder           , 
                                                                    data_file             , 
                                                                    data_file_binary      , 
                                                                    binary_class_bool     , 
                                                                    y_prpty_cls_threshold , 
                                                                    target_nme            ,
                                                                    max_seqs_len          ,
                                                                    )
    beautiful_print(processed_data_df)
    beautiful_print(processed_data_df_bi)

    #--------------------------------------------------#
    # Output #1: Write a fasta file including all sequences
    seqs_list = output_fasta(processed_data_df, output_folder, output_file_0)

    #--------------------------------------------------#
    # Output #2: Write compounds_properties_list
    seqs_prpty_dict = output_formated_dataset(processed_data_df     , 
                                              processed_data_df_bi  , 
                                              seqs_list             , 
                                              y_prpty_cls_threshold , 
                                              target_nme            , 
                                              data_folder           , 
                                              output_folder         , 
                                              output_file_1         ,
                                              )
    #--------------------------------------------------#

    return








#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'      db      `7MMF'`7MN.   `7MF'                 M             M             M                                                    #
#     MMMb    dPMM       ;MM:       MM    MMN.    M                   M             M             M                                                    #
#     M YM   ,M MM      ,V^MM.      MM    M YMb   M                   M             M             M                                                    #
#     M  Mb  M' MM     ,M  `MM      MM    M  `MN. M               `7M'M`MF'     `7M'M`MF'     `7M'M`MF'                                                #
#     M  YM.P'  MM     AbmmmqMA     MM    M   `MM.M                 VAMAV         VAMAV         VAMAV                                                  #
#     M  `YM'   MM    A'     VML    MM    M     YMM                  VVV           VVV           VVV                                                   #
#   .JML. `'  .JMML..AMA.   .AMMA..JMML..JML.    YM                   V             V             V                                                    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

if __name__ == "__main__":

    ###################################################################################################################
    ###################################################################################################################
    # Args
    #--------------------------------------------------#
    # Inputs
    Step_code = "N00_"
    #--------------------------------------------------#
    dataset_nme_list = ["NovoEnzyme",            # 0
                        "PafAVariants",          # 1
                        ]

    dataset_nme      = dataset_nme_list[1]
    #--------------------------------------------------#
    # Additional Informationm for certain datasets.
    PafAVariants_val_dict = { "kcat_cMUP"             : "kcat_cMUP_s-1"               ,
                              "KM_cMUP"               : "KM_cMUP_uM"                  ,
                              "kcatOverKM_cMUP"       : "kcatOverKM_cMUP_M-1s-1"      ,
                              "kcatOverKM_MeP"        : "kcatOverKM_MeP_M-1s-1"       ,
                              "kcatOverKM_MecMUP"     : "kcatOverKM_MecMUP_M-1s-1"    ,
                              "Ki_Pi"                 : "Ki_Pi_uM"                    ,
                              "fa"                    : "fa"                          ,
                              "kcatOverKM_MePkchem"   : "kcatOverKM_MePkchem_M-1s-1"  ,
                              "FC1"                   : "FC1"                         ,
                              "FC2_3"                 : "FC2/3"                       ,
                              "FC4"                   : "FC4_s-1"                     ,
                             }

    #--------------------------------------------------#
    # A dictionary of different datasets. 
    #                    dataset_nme-----------value_col--------------dataset_path----------------------------------seqs_len
    data_info_dict   = {"NovoEnzyme"       : ["tm"                     , "./NovoEnzyme/train.csv"                     ,   1200, ],  # 0
                        "PafAVariants"     : ["PafAVariants_val_dict"  , "./PafA_Variants/abf8761_data_processed.csv" ,   1200, ],  # 1
                        ""                 : [""                       , ""                                           ,   1200, ],  # 2
                       }

    #--------------------------------------------------#
    target_nme   = data_info_dict[dataset_nme][0]
    data_file    = data_info_dict[dataset_nme][1]
    max_seqs_len = data_info_dict[dataset_nme][2]
    #--------------------------------------------------#
    if target_nme in locals().keys():
        target_nme = locals()[target_nme]
    print(target_nme)
    #--------------------------------------------------#
    binary_class_bool = True
    #--------------------------------------------------#
    # Inputs
    data_folder = Path("N_DataProcessing/N00_raw_datasets_processed/")
    data_file_binary = data_file.replace(".csv", "_binary.csv") # y_prpty_cls_threshold = around 1e-2
    #--------------------------------------------------#
    # Settings
    y_prpty_cls_threshold = 1e-5 # Used for type II screening
    #--------------------------------------------------#
    # Outputs
    output_folder = Path("N_DataProcessing/")
    output_file_0 = Step_code + dataset_nme + ".fasta"
    output_file_1 = Step_code + dataset_nme + "_seqs_prpty_list.p"
    ###################################################################################################################
    ###################################################################################################################
    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--binary_class_bool"      , type = bool, default = binary_class_bool     , help = "If there is a file for binary classification."   )
    parser.add_argument("--data_folder"            , type = Path, default = data_folder           , help = "Path to the directory containing your datasets." )
    parser.add_argument("--target_nme"             , type = str , default = target_nme            , help = "Column name/header in the processed data file"   )
    parser.add_argument("--data_file"              , type = str , default = data_file             , help = "Filename to be read."                            )
    parser.add_argument("--data_file_binary"       , type = str , default = data_file_binary      , help = "Filename (binary classification) to be read."    )
    parser.add_argument("--max_seqs_len"           , type = int , default = max_seqs_len          , help = "Maximum Sequence Length."                        )
    parser.add_argument("--y_prpty_cls_threshold"  , type = int , default = y_prpty_cls_threshold , help = "y_prpty_cls_threshold."                          )
    parser.add_argument("--output_folder"          , type = Path, default = output_folder         , help = "Path to the directory containing output."        )
    parser.add_argument("--output_file_0"          , type = str , default = output_file_0         , help = "Filename of output_file_0_1."                    )
    parser.add_argument("--output_file_1"          , type = str , default = output_file_1         , help = "Filename of output_file_1."                      )
    args = parser.parse_args()
    #====================================================================================================#
    # Main
    #--------------------------------------------------#
    # Run Main

    N00_Data_Processing(**vars(args))
    print("*" * 50)
    print(Step_code + " Done!")
    #====================================================================================================#




#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#   `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'       `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'  #
#     VAMAV          VAMAV          VAMAV          VAMAV          VAMAV           VAMAV          VAMAV          VAMAV          VAMAV          VAMAV    #
#      VVV            VVV            VVV            VVV            VVV             VVV            VVV            VVV            VVV            VVV     #
#       V              V              V              V              V               V              V              V              V              V      #

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
###################################################################################################################
###################################################################################################################
#====================================================================================================#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#--------------------------------------------------#
#------------------------------

#                                                                                                                                                          
#      `MM.              `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.       
#        `Mb.              `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.     
# MMMMMMMMMMMMD     MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD   
#         ,M'               ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'     
#       .M'               .M'              .M'              .M'              .M'              .M'              .M'              .M'              .M'       
#                                                                                                                                                          

#------------------------------
#--------------------------------------------------#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#====================================================================================================#
###################################################################################################################
###################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#       A              A              A              A              A               A              A              A              A              A      #
#      MMM            MMM            MMM            MMM            MMM             MMM            MMM            MMM            MMM            MMM     #
#     MMMMM          MMMMM          MMMMM          MMMMM          MMMMM           MMMMM          MMMMM          MMMMM          MMMMM          MMMMM    #
#   ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.       ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.  #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
















































































