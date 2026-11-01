import os
import pickle

import lmdb
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
import warnings

warnings.filterwarnings(action="ignore")
from multiprocessing import Pool
import json
import copy
from functools import partial

## Helper functions for adding np_ratios to the json files - code from notebooks/add_np_ratio_or_ILRNA_wt_ratio.ipynb
smiles_to_names = {
    "OCCCCN(CCCCCCOC(C(CCCCCC)CCCCCCCC)=O)CCCCCCOC(C(CCCCCC)CCCCCCCC)=O": "ALC-0315",
    "CCCCC/C=C\\C/C=C\\CCCCCCCCC1(OC(CCN(C)C)CO1)CCCCCCCC/C=C\\C/C=C\\CCCCC": "KC2",
    # "CCCCCC\\C=C/C\\C=C/CCCCCCCCC(CCCCCCCC\\C=C/C\\C=C/CCCCC)OC(=O)CCCN(C)C": "XMC3",
    "CCCCCC=CCC=CCCCCCCCCC(CCCCCCCCC=CCC=CCCCCC)OC(=O)CCCN(C)C": "MC3",
    "OCCN(CCCCCCCC(OC(CCCCCCCC)CCCCCCCC)=O)CCCCCC(OCCCCCCCCCCC)=O": "SM-102",
    "OC(CCCCCCCCCC)CN(CCN(CC(CCCCCCCCCC)O)CC(CCCCCCCCCC)O)CCN1CCN(CCN(CC(CCCCCCCCCC)O)CC(CCCCCCCCCC)O)CC1": "C12-200",
    "CN(C)CCCC(OC(CCCCCCCC(OC/C=C\\CCCCCC)=O)CCCCCCCC(OC/C=C\\CCCCCC)=O)=O": "L319",
    "O=C1NC(CCCCN(CC(CCCCCCCCCC)O)CC(CCCCCCCCCC)O)C(NC1CCCCN(CC(CCCCCCCCCC)O)CC(CCCCCCCCCC)O)=O": "CKK-E12",
    "CCCCC/C=C\\CCCC(CCC/C=C\\CCCCC)C(CCC/C=C\\CCCCC)OC(=O)CCCCN1CCN(CCO)CC1": "BP322",
    "CC(C)CCCCCCCOC(=O)CCN(CCOCCOCCOCCOCCN(CCC(=O)OCCCCCCCC(C)C)CCC(=O)OCCCCCCCC(C)C)CCC(=O)OCCCCCCCC(C)C": "BP374",
    "CCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCC)COC(=O)CCCCCN1CC(CO)C1": "BP344",
    "CCCCCCC(CCCC)C(=O)OCCCCCCCCCN(CCCO)CCCCCCCCCOC(=O)C(CCCC)CCCCCC": "Lipid III-45",
    "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCN(C)C": "BP390",
    "CCCCCCCCCCCOC(=O)CCCCCCN(CCCO)CCCCCCCC(=O)OC(CCCCCCCC)CCCCCCCC": "BP148",
    "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\CC=CC/C=C\\CCC(=O)OCC(COC(=O)CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CC)(COC(=O)CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CC)COC(=O)CCCN(C)C": "BP403",
    "CCCCCCCCC(CCCCCCCC)OC(=O)CCCCCN(CCCO)CCCCCC(=O)OCCCCCCC": "BP128",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)(COC(=O)CCCCCCC/C=C\\CCCCCCCC)COC(=O)CCCCN1CCOCC1": "BP324",
    "CCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCC)COC(=O)CCCCCN1CC(O)C1": "BP343",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)CN(C)C": "BP392",
    "CCCCCCCCCCC(O)CN(CCN1CCN(CCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)CC1)CCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC": "C12-200",
    "CCCCCCCCC(CCCCCC)C(=O)OCCCCCCCCN(CCO)CCCCCCCCOC(=O)C(CCCCCC)CCCCCCCC": "BP217",
    "CCCCCCCCCCCC(O)CNCCN(C)CCN(CC(O)CCCCCCCCCCC)CC(O)CCCCCCCCCCC": "C13-113-tri-tail",
    "CCCCCCCCC(CCCCCCCC)OC(=O)CCCCCCN(CCCO)CCCCCCCC(=O)OCCCCCCC": "BP152",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)(COC(=O)CCCCCCC/C=C\\CCCCCCCC)COC(=O)CCN(C)C": "BP328",
    "CCCCCCCCCOC(=O)CCCCCN(CCCO)CCCCCC(=O)OC(CCCCCCCC)CCCCCCCC": "BP129",
    "CCCCCCCCC(CCCCCC)C(=O)OCCCCCCCN(CCCO)CCCCCCCOC(=O)C(CCCCCC)CCCCCCCC": "BP219",
    "CCCCCCCCCOC(=O)CCCCCCN(CCCO)CCCCCCC(=O)OC(CCCCCCCC)CCCCCCCC": "BP132",
    "CCCCCCCCCCCCC(O)CN(CCCCN(CCCN(CC(O)CCCCCCCCCCCC)CC(O)CCCCCCCCCCCC)CC(O)CCCCCCCCCCCC)CCCN(CC(O)CCCCCCCCCCCC)CC(O)CCCCCCCCCCCC": "C14-SPM",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)COC(=O)CCCCCN1CCC1": "BP352",
    "CCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCC)COC(=O)CCCCCN1CCC1": "BP342",
    "CCCCCCCCCCC(O)CN(CCCCN(CCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)CCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC": "C12-SPM",
    "CCCCCCCCCCCOC(=O)CCCCCN(CCCO)CCCCCC(=O)OC(CCCCCCCC)CCCCCCCC": "BP130",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCN1CCN(CCO)CC1)COC(=O)C(CCCCCC)CCCCCCCC.CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCN1CCN(CCO)CC1)COC(=O)C(CCCCCC)CCCCCCCC": "BP396",
    "CCCCCC/C=C\\COC(=O)CCN(CCOCCOCCOCCOCCN(CCC(=O)OC/C=C\\CCCCCC)CCC(=O)OC/C=C\\CCCCCC)CCC(=O)OC/C=C\\CCCCCC": "BP372",
    "CCCCCCCCCCCCCCOCCOC(=O)CCN(CCCn1ccnc1)CCC(=O)OCCOCCCCCCCCCCCCCC": "93-O17O",
    "CCCCCCCCCCCCC(O)CN(CCCN1CCN(CCCN(CC(O)CCCCCCCCCCCC)CC(O)CCCCCCCCCCCC)CC1)CC(O)CCCCCCCCCCCC": "IC8",
    "CCCCCCCC/C=C\\CCCCCCCCOCC(CN(C)C)OCCCCCCCC/C=C\\CCCCCCCC": "DODMA",
    "CCCCCCCCC(CCCCCC)C(=O)OCCCCCCCN(CCO)CCCCCCCOC(=O)C(CCCCCC)CCCCCCCC": "BP216",
    "CCCCCCCCC(CCCCCC)C(=O)OCCCCCCCCN(CCCO)CCCCCCCCOC(=O)C(CCCCCC)CCCCCCCC": "BP220",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)COC(=O)CCCCCN1CC(O)C1": "BP353",
    "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCN1CC(O)C1": "BP332",
    "CCCCCCCCCCCCCCSCCOC(=O)CCN(CCCn1ccnc1)CCC(=O)OCCSCCCCCCCCCCCCCC": "93-O17S",
    "CCCCCCCCCCCOCCOCCOCCOC(=O)CCCCCN(CCO)CCCCCCCC(=O)OC(CCCCCCCC)CCCCCCCC": "BP801",
    "CCCCCCCCCCCC(O)CN(CCN(C)CCN(CC(O)CCCCCCCCCCC)CC(O)CCCCCCCCCCC)CC(O)CCCCCCCCCCC": "C13-113-tetra-tail",
    "CCCCCCCCCOC(=O)CCCN(CCO)CCCCCCCC(=O)OC(CCCCCCCC)CCCCCCCC": "L13",
    "CCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCC(OCCCCCCCC)OCCCCCCCC)COC(=O)OCCCN(CC)CC": "BP312",
    "CCCCC/C=C\\C/C=C\\CCCCCCCCN(CCCCCCCC/C=C\\C/C=C\\CCCCC)CCOC(=O)CCN1CCN(C)CC1": "Lipid 10",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)(COC(=O)CCCCCCC/C=C\\CCCCCCCC)COC(=O)CCN1CCCC1": "BP329",
    "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)COC(=O)CCCCCN1CC(CO)C1": "BP354",
    "CCCCCCCCCCC(O)CN(CCCN1CCN(CCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)CC1)CC(O)CCCCCCCCCC": "246C10",
    "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCC(OCCCCCCCC)OCCCCCCCC)COC(=O)OCCCN1CCCCC1": "BP401",
    "CCCCCCCCC(CCCCCC)C(=O)OCCCCCCN(CCO)CCCCCCOC(=O)C(CCCCCC)CCCCCCCC": "BP226",
    "CCCCCC/C=C\\COC(=O)CCN(CCOCCOCCN(CCC(=O)OC/C=C\\CCCCCC)CCC(=O)OC/C=C\\CCCCCC)CCC(=O)OC/C=C\\CCCCCC": "BP373",
    "CCCCC/C=C\\C/C=C\\CCCCCCCCOCC(CN(C)C)OCCCCCCCC/C=C\\C/C=C\\CCCCC": "DMA",
    "CCCCCCCCCCC(O)CN(CCN(C)CCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC": "C12-113"
}

ils_to_mws = {
    "ALC-0315": 766.3,
    "KC2": 642.1,
    "MC3": 642.1,
    "SM-102": 710.2,
    "C12-200": 1136.9,
    "L319": 678,
    "CKK-E12": 993.6,
    'BP217': 794.4,
    'BP128': 640.1,
    'BP801': 842.3,
    '93-O17O': 750.2,
    'C12-200': 1136.9,
    'BP324': 1098.7,
    'BP390': 1022.6,
    'BP401': 864.3,
    'BP353': 804.3,
    'BP322': 673.1,
    'C13-113-tetra-tail': 910.6,
    'BP372': 1021.5,
    'BP226': 738.2,
    'BP216': 766.3,
    'Lipid III-45': 724.2,
    'BP219': 780.3,
    'BP220': 808.4,
    'BP129': 668.1,
    'BP130': 696.2,
    'BP132': 696.2,
    'BP148': 738.2,
    'BP152': 682.1,
    'MC3': 642.1,
    'KC2': 642.1,
    'DMA': 616.1,
    'DODMA': 620.1,
    'BP328': 1028.6,
    'BP329': 1054.7,
    'BP403': 1180.8,
    'BP312': 826.3,
    'BP396': 807.3,
    'BP342': 732.2,
    'BP352': 788.3,
    'BP343': 748.1,
    'BP344': 762.2,
    'BP354': 818.3,
    'BP392': 662.1,
    'C13-113-tri-tail': 712.3,
    'BP373': 933.4,
    'Lipid 10': 712.2,
    '93-O17S': 782.3,
    'BP374': 1085.7,
    'C12-113': 854.5,
    'BP332': 800.2,
    'L13': 654.1,
    'IC8': 1049.8,
    '246C10': 937.6,
    'C12-SPM': 1308.3,
    'C14-SPM': 1476.6,
}

il_amine_heads = {
    "ALC-0315": 1,
    "KC2": 1,
    "MC3": 1,
    "SM-102": 1,
    "C12-200": 5,
    "L319": 1,
    "CKK-E12": 2,
    'BP217': 1,
    'BP128': 1,
    'BP801': 1,
    '93-O17O': 3,
    'C12-200': 5,
    'BP324': 1,
    'BP390': 1,
    'BP401': 1,
    'BP353': 1,
    'BP322': 2,
    'C13-113-tetra-tail': 3,
    'BP372': 2,
    'BP226': 1,
    'BP216': 1,
    'Lipid III-45': 1,
    'BP219': 1,
    'BP220': 1,
    'BP129': 1,
    'BP130': 1,
    'BP132': 1,
    'BP148': 1,
    'BP152': 1,
    'MC3': 1,
    'KC2': 1,
    'DMA': 1,
    'DODMA': 1,
    'BP328': 1,
    'BP329': 1,
    'BP403': 1,
    'BP312': 1,
    'BP396': 2,
    'BP342': 1,
    'BP352': 1,
    'BP343': 1,
    'BP344': 1,
    'BP354': 1,
    'BP392': 1,
    'C13-113-tri-tail': 3,
    'BP373': 2,
    'Lipid 10': 3,
    '93-O17S': 3,
    'BP374': 2,
    'C12-113': 3,
    'BP332': 1,
    'L13': 1,
    'IC8': 4,
    '246C10': 4,
    'C12-SPM': 4,
    'C14-SPM': 4,
}

ilrna_dict = {
    "10:1": 10,
    "15:1": 15,
    "20:1": 20,
}

ref_il = "KC2"
rna_nt_mw = 339.5


def compute_il_rna_wt_ratio(
    il_rna_wt_ratio_str,
    il_name=None,
    il_mw=None,
    ilrna_str_to_float=ilrna_dict,
    ils_to_mws=ils_to_mws,
    ref_il=ref_il,
):
    if il_mw == None:
        if il_name == None:
            raise ValueError("Either il_name or il_mw must be provided")
        else:
            il_mw = ils_to_mws[il_name]

    if type(il_rna_wt_ratio_str) == float:
        ilrna_ratio = il_rna_wt_ratio_str
    elif il_rna_wt_ratio_str not in ilrna_str_to_float.keys():
        ilrna_ratio = float(il_rna_wt_ratio_str.split(":")[0])
    else:
        ilrna_ratio = ilrna_str_to_float[il_rna_wt_ratio_str]
    ref_il_mw = ils_to_mws[ref_il]
    actual_ilrna_ratio = ilrna_ratio * il_mw / ref_il_mw

    return actual_ilrna_ratio


def compute_np_ratio(
    ilrna_ratio,
    il_name=None,
    il_mw=None,
    num_N_per_il=None,
    ils_to_mws=ils_to_mws,
    nt_mw=rna_nt_mw,
):
    if il_mw == None:
        if il_name == None:
            raise ValueError("Either il_name or il_mw must be provided")
        else:
            il_mw = ils_to_mws[il_name]
    if num_N_per_il == None:
        if il_name == None:
            raise ValueError("Either il_name or num_N_per_il must be provided")
        else:
            num_N_per_il = il_amine_heads[il_name]

    np_ratio = ilrna_ratio * num_N_per_il / il_mw * nt_mw

    return np_ratio


# main function to compute actual_ilrna_wt_ratio and NP_ratio
def compute_lnp_ilrna_np_ratio(lnp_obj):
    components = lnp_obj["components"]
    total_il_mol = 0
    il_mols = []
    il_smis = []
    ilrna_wt_ratios = []
    np_ratios = []

    il_rna_wt_ratio_str = lnp_obj["il_rna_wt_ratio"]
    for component in components:
        if component["component_type"] == "IL":
            il_smi = component["smi"]
            il_name = smiles_to_names[il_smi]
            total_il_mol += component["mol"]
            il_mols.append(component["mol"])
            il_smis.append(il_smi)

            # compute ILRNA_wt_ratio and np_ratio for each IL
            actual_ilrna_wt_ratio = compute_il_rna_wt_ratio(
                il_rna_wt_ratio_str, il_name
            )
            np_ratio = compute_np_ratio(actual_ilrna_wt_ratio, il_name)
            ilrna_wt_ratios.append(actual_ilrna_wt_ratio)
            np_ratios.append(np_ratio)

    if len(il_mols) > 1:  # handle synergistic LNP with more than 1 IL
        cur_ilrna_wt_ratio_sum = 0
        cur_np_ratio_sum = 0
        total_il_mol = sum(il_mols)
        for il_i, mol in enumerate(il_mols):
            cur_ilrna_wt_ratio = ilrna_wt_ratios[il_i]
            cur_np_ratio = np_ratios[il_i]

            cur_ilrna_wt_ratio_sum += cur_ilrna_wt_ratio * (mol / total_il_mol)
            cur_np_ratio_sum += cur_np_ratio * (mol / total_il_mol)
        # print("cur_ilrna_wt_ratio_sum, cur_np_ratio_sum: ", cur_ilrna_wt_ratio_sum, cur_np_ratio_sum)
        return cur_ilrna_wt_ratio_sum, cur_np_ratio_sum

    elif len(il_mols) == 1:
        # print("ilrna_wt_ratios[0], np_ratios[0]: ", ilrna_wt_ratios[0], np_ratios[0])
        return ilrna_wt_ratios[0], np_ratios[0]
    else:
        raise ValueError("No IL found in LNP")


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(
        coordinates
    ), "2D coordinates shape is not align with {}".format(smi)
    return coordinates

def smi2_3Dcoords(smi, cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(
                mol, randomSeed=seed
            )  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(
                        mol
                    )  # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D A, smi: ", smi)
                    coordinates = smi2_2Dcoords(smi)

            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(
                        mol_tmp
                    )  # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D B, smi: ", smi)
                    coordinates = smi2_2Dcoords(smi)
        except:
            print("Failed to generate 3D, replace with 2D C, sim: ", smi)
            coordinates = smi2_2Dcoords(smi)

        assert len(mol.GetAtoms()) == len(
            coordinates
        ), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

def inner_lnp2data(smi2mol_id, content, pickle_output=True):
    components_list = content['components']
    if "labels" in content:
        raw_labels = content['labels']
    else:
        raw_labels = {}
    dataset_name = content['dataset_name']
    lnp_id = content['lnp_id']

    # handle non-core (optional) attributes of LNPs, e.g. NP_ratio
    np_props = {}
    if "NP_ratio" in content:
        np_props['NP_ratio'] = content['NP_ratio']
    if "actual_ilrna_wt_ratio" in content:
        np_props['actual_ilrna_wt_ratio'] = content['actual_ilrna_wt_ratio']
    if "volumetric_ratio" in content:
        np_props['volumetric_ratio'] = content['volumetric_ratio'] 
    

    if "process" in content:
        np_props['process'] = content['process'] # freezing, lyophilization, storage, etc.
    else:
        np_props['process'] = "[PAD]" # default value for fresh synthesis and transfection, for backward compatibility with 1st COMET dataset


    labels = raw_labels # raw_labels is already normalized

    output_components_list = []
    
    mol_ids = []
    percents = []
    component_types = []
    composition_enc_types = []

    for component in components_list:

        # component_output = component.copy()
        component_output = copy.deepcopy(component)
        mol_id = smi2mol_id[component['smi']]
        component_output['mol_id'] = mol_id

        output_components_list.append(component_output)

        mol_ids.append(mol_id)
        component_type = component['component_type']
        component_types.append(component_type)
        composition_enc_types.append(0) # 0 for lipids, 1 for excipients, for different gaussian encoders to encode their composition values

        percent = component['percent']
        percents.append(percent)

    # SSLNP_todo: add SS LNP process, including a none type for backward compatibility with 1st COMET dataset
    if 'excipients' in content:
        excipients_list = content['excipients']
        for excipient in excipients_list:

            # excipient_output = excipient.copy()
            excipient_output = copy.deepcopy(excipient)
            mol_id = smi2mol_id[excipient['smi']]
            excipient_output['mol_id'] = mol_id

            # output_excipients_list.append(excipient_output)
            output_components_list.append(excipient_output)

            # add excipient to mol_ids and component_types
            mol_ids.append(mol_id)
            component_types.append('Excipient')
            composition_enc_types.append(1) # 0 for lipids, 1 for excipients, for different gaussian encoders to encode their composition values

            # add excipient molar conc
            if 'molar_conc' in excipient:
                excipient_molar_conc = excipient['molar_conc']
            elif 'mol' in excipient:
                excipient_molar_conc = excipient['mol']
            else:
                raise ValueError("excipient must have molar_conc or mol attribute")
            percents.append(excipient_molar_conc)

    output = {
        'mol_id': mol_ids, # lipids and excipients will be encoded by the same unimol encoder
        'percent': percents, 'component_type': component_types, 
        'target': labels, 
        'dataset_name': dataset_name, 
        'components': output_components_list,
        'lnp_id': lnp_id,
        "composition_enc_types": composition_enc_types,
        **np_props # fold non-core (optional) attributes of LNPs into sample dict, e.g. NP_ratio, volumetric_ratio
        }

    # print("inner_lnp2data output: ", output)
    
    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        output

def lnp2data(smi2mol_id, content):
    try:
        return inner_lnp2data(smi2mol_id, content)
    except:
        print("failed lnp: {}".format(content[0]))
        return None


def inner_smi2coords(content, pickle_output=False):
    # content: {
    #     "name": "IL-1",
    #     "smi": "[NH+]=1[C@](N=C(C)C=1N)(C1CC1)c1cc(ccc1)-c1cc(cnc1)C#CC",
    #     "percent": 100
    # }

    smi = content

    cnt = 10  # conformer num,all==11, 10 3d + 1 2d
    # print("before mol = Chem.MolFromSmiles smi: ", smi)
    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list = [smi2_2Dcoords(smi)] * (cnt + 1)
        print("atom num >400,use 2D coords", smi)
    else:
        coordinate_list = smi2_3Dcoords(smi, cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H

    output = {"atoms": atoms, "coordinates": coordinate_list, "mol": mol, "smi": smi}

    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        return output


def smi2coords_onlymol(content):
    try:
        return inner_smi2coords_onlymol(content)
    except:
        print("failed smiles: {}".format(content))
        return None


def inner_smi2coords_onlymol(content, pickle_output=True):
    output = inner_smi2coords(content)
    # print("inner_smi2coords_onlymol, output: ", output)
    if pickle_output:
        return pickle.dumps(output, protocol=-1)
    else:
        output


# helper function to process json data file and save them as lmdb datasets
def write_lmdb_concatdataset_for_inference(
    inpath="./", outpath="./", nthreads=16, debug=False
):
    # Data wil be stored in JSON format, e.g. each sample: {components: [{smi: <SMILES>, percent: <%>, name: IL-1}, {..}], label: <label_value>}
    with open(os.path.join(inpath), "r") as openfile:
        # Reading from json file
        json_obj = json.load(openfile)

    dataset_name_list = []
    dataset_dict = {}

    # collate into list of lnps
    json_list = []
    for lnp_id in json_obj:
        lnp_dict = json_obj[lnp_id]
        lnp_dict["lnp_id"] = lnp_id

        if "dataset_name" in lnp_dict:
            lnp_dataset_name = "in_house_lnp"
            # lnp_dataset_name = lnp_dict['dataset_name']
            if lnp_dataset_name not in dataset_name_list:
                dataset_name_list.append(lnp_dataset_name)
                dataset_dict[lnp_dataset_name] = []
            dataset_dict[lnp_dataset_name].append(lnp_dict)

        # process components' percent value
        np_components = lnp_dict["components"]
        # total_weight = 0

        # record percent composition in component dict
        for c_id, component in enumerate(np_components):
            lnp_dict["components"][c_id]["percent"] = lnp_dict["components"][c_id][
                "mol"
            ]

        json_list.append(lnp_dict)

    sz = len(json_list)
    print("sz: ", sz)

    # make master mol lmdb dataset as a list of all unique mols in datasets
    smi_list = []
    for sample_i, np_obj in enumerate(json_list):
        np_components = np_obj["components"]
        unique_smi_count = 0
        for component in np_components:
            smi = component["smi"]
            if smi not in smi_list:
                unique_smi_count += 1
                print(sample_i, " adding smi to unique smi list: ", smi)
                smi_list.append(smi)
        # if unique_smi_count == 0:
        #     print(sample_i, " no unique smi added")

    for sample_i, np_obj in enumerate(json_list):
        if 'excipients' in np_obj:
            np_excipients = np_obj['excipients']
            for excipient in np_excipients:
                smi = excipient['smi']
                if smi not in smi_list:
                    unique_smi_count += 1
                    smi_list.append(smi)

    print("len(smi_list): ", len(smi_list))

    mol_filename = "mol.lmdb"
    os.makedirs(outpath, exist_ok=True)
    mol_output_name = os.path.join(outpath, mol_filename)
    try:
        os.remove(mol_output_name)
    except:
        pass
    env_new = lmdb.open(
        mol_output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(smi2coords_onlymol, smi_list)):
            if inner_output is not None:
                print(
                    "i=",
                    i,
                    " data = pickle.loads(datapoint_pickled) smi: ",
                    pickle.loads(inner_output)["smi"],
                )
                txn_write.put(f"{i}".encode("ascii"), inner_output)
                i += 1
        print("{} process {} lines".format(mol_filename, i))
        txn_write.commit()
        env_new.close()

    print("finished processing mol.lmdb")

    # make smi2mol_id dict
    smi2mol_id = {}
    mol_id2smi = {}

    # for ind, lnp_id in enumerate(json_obj):
    for mol_id, smi in enumerate(smi_list):
        smi2mol_id[smi] = mol_id
        mol_id2smi[mol_id] = smi

    lnp2data_w_smi2mol_id = partial(
        lnp2data, smi2mol_id
    )  # smi2mol_id is the dict to map smi to mol_id in mol.lmdb

    def make_data_lmdb(infer, dataset_outpath, debug=False):
        for name, content_list in [("infer.lmdb", infer)]:
            os.makedirs(dataset_outpath, exist_ok=True)

            if debug:
                output_json_name = os.path.join(
                    dataset_outpath, name.replace(".lmdb", ".json")
                )
                json_object = json.dumps(content_list, indent=4)
                with open(output_json_name, "w") as outfile:
                    outfile.write(json_object)

            output_name = os.path.join(dataset_outpath, name)
            try:
                os.remove(output_name)
            except:
                pass
            env_new = lmdb.open(
                output_name,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(100e9),
            )
            txn_write = env_new.begin(write=True)
            with Pool(nthreads) as pool:
                i = 0
                for inner_output in tqdm(
                    pool.imap(lnp2data_w_smi2mol_id, content_list)
                ):
                    if inner_output is not None:
                        txn_write.put(f"{i}".encode("ascii"), inner_output)
                        i += 1
                print("{} process {} lines".format(name, i))
                txn_write.commit()
                env_new.close()

    # make_data_lmdb end

    for dataset_name in dataset_dict:
        dataset_json_list = dataset_dict[dataset_name]

        dataset_outpath = os.path.join(outpath, dataset_name)
        make_data_lmdb(dataset_json_list, dataset_outpath, debug)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    vlib_path = args.path
    output_path = args.output_dir

    for json_file in os.listdir(vlib_path):
        write_lmdb_concatdataset_for_inference(os.path.join(vlib_path, json_file), os.path.join(output_path, json_file))