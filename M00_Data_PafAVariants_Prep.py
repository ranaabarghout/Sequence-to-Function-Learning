#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
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

###################################################################################################################
###################################################################################################################
# Imports
#--------------------------------------------------#
import re
import time
import copy
import pickle
import requests
import argparse
import numpy as np
import pandas as pd
#--------------------------------------------------#
import requests
import xmltodict
#--------------------------------------------------#
from timeit import timeit
from pypdb import get_pdb_file

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Basic Functions
# Print the DataFrame obtained.
def beautiful_print(df): # Print the DataFrame obtained.
    # Print the dataset in a well-organized format.
    with pd.option_context('display.max_rows'       , 20   , 
                           'display.min_rows'       , 20   , 
                           'display.max_columns'    , 5    , 
                           #"display.max_colwidth"  , None ,
                           "display.width"          , None ,
                           "expand_frame_repr"      , True ,
                           "max_seq_items"          , None , ):  # more options can be specified
        # Once the display.max_rows is exceeded, 
        # the display.min_rows options determines 
        # how many rows are shown in the truncated repr.
        print(df)
    return 


###################################################################################################################
#    `7MMF'`7MN.   `7MF'`7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                                           #
#      MM    MMN.    M    MM   `MM.  MM       M  P'   MM   `7 ,MI    "Y                                           #
#      MM    M YMb   M    MM   ,M9   MM       M       MM      `MMb.                                               #
#      MM    M  `MN. M    MMmmdM9    MM       M       MM        `YMMNq.                                           #
#      MM    M   `MM.M    MM         MM       M       MM      .     `MM                                           #
#      MM    M     YMM    MM         YM.     ,M       MM      Mb     dM                                           #
#    .JMML..JML.    YM  .JMML.        `bmmmmd"'     .JMML.    P"Ybmmd"                                            #
################################################################################################################### 
# Input Arguments
data_folder      = Path("N_DataProcessing/N00_raw_datasets_processed/")
data_file        = "./PafA_Variants/abf8761_data.csv"

output_folder    = Path("N_DataProcessing/N00_raw_datasets_processed/")
output_file      = "./PafA_Variants/abf8761_data_processed.csv"




###################################################################################################################
#    `7MMM.     ,MMF'      db      `7MMF'`7MN.   `7MF'     `7MM"""YMM `7MMF'`7MMF'      `7MM"""YMM  
#      MMMb    dPMM       ;MM:       MM    MMN.    M         MM    `7   MM    MM          MM    `7  
#      M YM   ,M MM      ,V^MM.      MM    M YMb   M         MM   d     MM    MM          MM   d    
#      M  Mb  M' MM     ,M  `MM      MM    M  `MN. M         MM""MM     MM    MM          MMmmMM    
#      M  YM.P'  MM     AbmmmqMA     MM    M   `MM.M         MM   Y     MM    MM      ,   MM   Y  , 
#      M  `YM'   MM    A'     VML    MM    M     YMM         MM         MM    MM     ,M   MM     ,M 
#    .JML. `'  .JMML..AMA.   .AMMA..JMML..JML.    YM       .JMML.     .JMML..JMMmmmmMMM .JMMmmmmMMM 
###################################################################################################################

# Read the main data file.
raw_df_0 = pd.read_csv(filepath_or_buffer   =   data_folder / data_file, 
                       on_bad_lines         =   'skip', 
                       index_col            =   None, 
                       #names               =   ["", "", ], 
                       header               =   0, 
                       sep                  =   ',', 
                       encoding             =   'cp1252')


print("\n\n"+"="*90+"\n#0.0 Raw data read from the csv file, raw_df_0: ")
beautiful_print(raw_df_0)
print("len(raw_df_0): ", len(raw_df_0))

#====================================================================================================#
# Filter Something (optional).
raw_df_0_1 = copy.deepcopy(raw_df_0)


###################################################################################################################
#   .M"""bgd `7MM"""YMM    .g8""8q.   `7MMF'   `7MF'`7MM"""YMM  `7MN.   `7MF'  .g8"""bgd `7MM"""YMM  
#  ,MI    "Y   MM    `7  .dP'    `YM.   MM       M    MM    `7    MMN.    M  .dP'     `M   MM    `7  
#  `MMb.       MM   d    dM'      `MM   MM       M    MM   d      M YMb   M  dM'       `   MM   d    
#    `YMMNq.   MMmmMM    MM        MM   MM       M    MMmmMM      M  `MN. M  MM            MMmmMM    
#  .     `MM   MM   Y  , MM.      ,MP   MM       M    MM   Y  ,   M   `MM.M  MM.           MM   Y  , 
#  Mb     dM   MM     ,M `Mb.    ,dP'   YM.     ,M    MM     ,M   M     YMM  `Mb.     ,'   MM     ,M 
#  P"Ybmmd"  .JMMmmmmMMM   `"bmmd"'      `bmmmmd"'  .JMMmmmmMMM .JML.    YM    `"bmmmd'  .JMMmmmmMMM 
#                              MMb                                                                   
#                               `bood'                                                               
###################################################################################################################
# According to the supporting information, the wild type PafA protein can be found in PDB with a code "5TJ3".
# Also, it can be found in the UniProtKB with a code "Q9KJX5", which gives a sequence that is a little different.
# The supporting info contains a sequence starting with "MQKTNA...",
# However, the variants are named after a four-character code which shows how the PafA sequence are modified.
# The code shows that starting from the 21th amino acid, the WT sequence is "QKTNA...".
# This is a little bit different from the sequence mentioned earlier in the SI.
# Here, four sequences are listed out, while we use a made-up sequence that satifies all conditions above.
# The made-up sequence is UniProtKB's Q9KJX5 followed by unknown parts found in the SI.
# 1. Starting from 21th amino acid, the sequence have "QKTNA..."
# 2. Contains the entire sequence of "Q9KJX5" in UniProtKB and "5TJ3" in PDB.
# 3. Contains everything of the sequence in SI.
# The made-up sequence is named, PafA_seq_experiment.

#====================================================================================================#
def get_pdb_sequence(pdb_id):
    #--------------------------------------------------#
    pdb_sequence_item_name = ["_entity_poly.pdbx_seq_one_letter_code_can",
                              "_entity_poly.pdbx_seq_one_letter_code ",
                              "_struct_ref.pdbx_seq_one_letter_code"]
    #--------------------------------------------------#
    pdb_file = get_pdb_file(pdb_id, filetype="cif", compression=False)
    pdb_file_list = pdb_file.split("\n")
    search_results = []
    for item_name in pdb_sequence_item_name:
        for one_line in pdb_file_list:
            #print(one_line)
            if one_line.find(item_name) != -1:
                sequence_info_line_index = pdb_file_list.index(one_line)
                seq_str=""
                while(1):
                    sequence_info_line_index+=1
                    seq_str = seq_str + pdb_file_list[sequence_info_line_index]
                    if pdb_file_list[sequence_info_line_index] == ";":
                        break
    #--------------------------------------------------#
        seq_str = seq_str.replace(";","")
        search_results.append(seq_str)
    #--------------------------------------------------#
    return search_results

#====================================================================================================#
def get_unip_sequence(unip_id):
    # unip_id: uniprot_id 
    xml_url = 'https://rest.uniprot.org/uniprotkb/' + str(unip_id) + '.xml'
    xml_response = requests.get(xml_url)
    xml_data = xmltodict.parse(xml_response.content)
    seqs = xml_data["uniprot"]["entry"]["sequence"]['#text']
    return seqs

#====================================================================================================#
PafA_seq_PDB      =  get_pdb_sequence(pdb_id = "5TJ3")[2]
PafA_seq_UniProt  =  get_unip_sequence(unip_id = "Q9KJX5") #"MLTPKKWLLGVLVVSGMLGAQKTNAVPRPKLVVGLVVDQMRWDYLYRYYSKYGEGGFKRMLNTGYSLNNVHIDYVPTVTAIGHTSIFTGSVPSIHGIAGNDWYDKELGKSVYCTSDETVQPVGTTSNSVGQHSPRNLWSTTVTDQLGLATNFTSKVVGVSLKDRASILPAGHNPTGAFWFDDTTGKFITSTYYTKELPKWVNDFNNKNVPAQLVANGWNTLLPINQYTESSEDNVEWEGLLGSKKTPTFPYTDLAKDYEAKKGLIRTTPFGNTLTLQMADAAIDGNQMGVDDITDFLTVNLASTDYVGHNFGPNSIEVEDTYLRLDRDLADFFNNLDKKVGKGNYLVFLSADHGAAHSVGFMQAHKMPTGFFVEDMKKEMNAKLKQKFGADNIIAAAMNYQVYFDRKVLADSKLELDDVRDYVMTELKKEPSVLYVLSTDEIWESSIPEPIKSRVINGYNWKRSGDIQIISKDGYLSAYSKKGTTHSVWNSYDSHIPLLFMGWGIKQGESNQPYHMTDIAPTVSSLLKIQFPSGAVGKPITEVIGR"
PafA_seq_SI       =  "MQKTNAVPRPKLVVGLVVDQMRWDYLYRYYSKYGEGGFKRMLNTGYSLNNVHIDYVPTVTAIGHTSIFTGSVPSIHGIAGNDWYDKELGKSVYCTSDETVQPVGTTSNSVGQHSPRNLWSTTVTDQLGLATNFTSKVVGVSLKDRASILPAGHNPTGAFWFDDTTGKFITSTYYTKELPKWVNDFNNKNVPAQLVANGWNTLLPINQYTESSEDNVEWEGLLGSKKTPTFPYTDLAKDYEAKKGLIRTTPFGNTLTLQMADAAIDGNQMGVDDITDFLTVNLASTDYVGHNFGPNSIEVEDTYLRLDRDLADFFNNLDKKVGKGNYLVFLSADHGAAHSVGFMQAHKMPTGFFVEDMKKEMNAKLKQKFGADNIIAAAMNYQVYFDRKVLADSKLELDDVRDYVMTELKKEPSVLYVLSTDEIWESSIPEPIKSRVINGYNWKRSGDIQIISKDGYLSAYSKKGTTHSVWNSYDSHIPLLFMGWGIKQGESNQPYHMTDIAPTVSSLLKIQFPSGAVGKPITEVIGR" + \
                     "GGGSGGGGSGMVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
PafA_seq_SI_agg   =  "GGGSGGGGSGMVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"

print(len(PafA_seq_SI)) # 776, confirmed.

PafA_seq_experiment = PafA_seq_UniProt 
# PafA_seq_experiment = PafA_seq_UniProt + PafA_seq_SI_agg



###################################################################################################################
#  `7MM"""Mq.`7MM"""Mq.   .g8""8q.     .g8"""bgd `7MM"""YMM   .M"""bgd  .M"""bgd 
#    MM   `MM. MM   `MM..dP'    `YM. .dP'     `M   MM    `7  ,MI    "Y ,MI    "Y 
#    MM   ,M9  MM   ,M9 dM'      `MM dM'       `   MM   d    `MMb.     `MMb.     
#    MMmmdM9   MMmmdM9  MM        MM MM            MMmmMM      `YMMNq.   `YMMNq. 
#    MM        MM  YM.  MM.      ,MP MM.           MM   Y  , .     `MM .     `MM 
#    MM        MM   `Mb.`Mb.    ,dP' `Mb.     ,'   MM     ,M Mb     dM Mb     dM 
#  .JMML.    .JMML. .JMM. `"bmmd"'     `"bmmmd'  .JMMmmmmMMM P"Ybmmd"  P"Ybmmd"  
###################################################################################################################
# Modify a AASeq using a "variant code" found in the SI data file.
def seq_vrnt_modify(seq_str, vrnt_code = "Q21G"):
    #--------------------------------------------------#
    def replace_n(str1, n, str2): # replace (n+1)_th char in str1 with str2
        letters = (
            str2 if i == n else char
            for i, char in enumerate(str1)
        )
        return ''.join(letters)
    #--------------------------------------------------#
    vrnt_original_bool = True
    if vrnt_code == "WT":
        seq_vrnt_modified  = seq_str
    #--------------------------------------------------#
    elif vrnt_code.find("/") == -1:
        vrnt_original_AA   = vrnt_code[0]
        vrnt_replaced_AA   = vrnt_code[-1]
        vrnt_position      = int(vrnt_code[1:-1])
        vrnt_original_bool = (seq_str[vrnt_position-1] == vrnt_original_AA)
        seq_vrnt_modified  = replace_n(seq_str, vrnt_position-1, vrnt_replaced_AA)
    #--------------------------------------------------#
    else: # Multiple modifications
        variant_code_list = vrnt_code.split("/")
        for one_vrnt_code in variant_code_list:
            vrnt_original_AA   = one_vrnt_code[0]
            vrnt_replaced_AA   = one_vrnt_code[-1]
            vrnt_position      = int(one_vrnt_code[1:-1])
            vrnt_original_bool = (seq_str[vrnt_position-1] == vrnt_original_AA) and vrnt_original_bool
            seq_vrnt_modified  = replace_n(seq_str, vrnt_position-1, vrnt_replaced_AA)
            seq_str            = seq_vrnt_modified
    #--------------------------------------------------#
    return vrnt_original_bool, seq_vrnt_modified

vrnt_list = raw_df_0_1["variant"].tolist()
seqs_list = []

for one_vrnt in vrnt_list:
    vali, seqs = seq_vrnt_modify(PafA_seq_experiment, vrnt_code = one_vrnt)
    #print(seqs, one_vrnt, vali)
    seqs_list.append(seqs)

raw_df_0_1["SEQ"] = seqs_list

#====================================================================================================#
# Deal with columns.
PafAVariants_val_dict = {   "kcat_cMUP"             : "kcat_cMUP_s-1"               ,
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

PafAVariants_val_dict_re = dict([])
for one_key in PafAVariants_val_dict.keys():
    PafAVariants_val_dict_re[PafAVariants_val_dict[one_key]] = one_key

raw_df_0_1.rename(columns = PafAVariants_val_dict_re, inplace=True)


useless_col_list = [ "variant", "library", ]
for one_head in raw_df_0_1.head():
    if one_head.find("_limit") != -1 or one_head.find("_p-value") != -1:
        useless_col_list.append(one_head)
raw_df_0_1.drop(columns = useless_col_list, inplace=True)


#====================================================================================================#
# Print.
print("\n\n"+"="*90+"\n#0.1 Data processed, sequences generated, raw_df_0_1: ")
beautiful_print(raw_df_0_1)
print("len(raw_df_0): ", len(raw_df_0_1))





###################################################################################################################
#      .g8""8q. `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq.`7MMF'   `7MF'MMP""MM""YMM 
#    .dP'    `YM. MM       M  P'   MM   `7   MM   `MM. MM       M  P'   MM   `7 
#    dM'      `MM MM       M       MM        MM   ,M9  MM       M       MM      
#    MM        MM MM       M       MM        MMmmdM9   MM       M       MM      
#    MM.      ,MP MM       M       MM        MM        MM       M       MM      
#    `Mb.    ,dP' YM.     ,M       MM        MM        YM.     ,M       MM      
#      `"bmmd"'    `bmmmmd"'     .JMML.    .JMML.       `bmmmmd"'     .JMML.    
###################################################################################################################
# Output.
raw_df_0_1.to_csv(output_folder / output_file)


print("Done!")

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#




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

