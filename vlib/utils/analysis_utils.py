import hashlib
import json
import os

import numpy as np


def numeratorFromRatio(ratio):
    """
    Returns the numerator of a ratio string.
    """
    numerator = float(ratio.split(":")[0])
    if numerator % 1 == 0:
        return int(numerator)
    else:
        return numerator


def check_if_directory_is_empty(path):
    if len(os.listdir(path)) > 0:
        raise Exception("Output directory is not empty.")


def get_smile_to_name_map():
    # Use lipid smiles as source of truth for vector index mapping.
    lipid_smiles_template = "./../../virtual_library/misc/lipid_smilesl.json"
    # Join the nested lipid smiles template into one list of names.
    with open(lipid_smiles_template, "r") as f:
        lipid_smiles = json.load(f)

    smile_to_name_map = {
        smile: name
        for lipid_type in lipid_smiles.values()
        for smile, name in lipid_type.items()
    }

    return smile_to_name_map


def get_name_to_smile_map():
    # Use lipid smiles as source of truth for vector index mapping.
    lipid_smiles_template = "./../../virtual_library/misc/lipid_smilesl.json"
    # Join the nested lipid smiles template into one list of names.
    with open(lipid_smiles_template, "r") as f:
        lipid_smiles = json.load(f)

    name_to_smile_map = {
        name: smile
        for lipid_type in lipid_smiles.values()
        for smile, name in lipid_type.items()
    }

    return name_to_smile_map

def full_vectorization_excipients(
    library: dict,
    lipid_smiles_path,
    booster_clusters_path,
    il_clusters_path,
    features,
    excipients=True,
    il_mol_pcts=True,
    ratios=True,
    max_volumetric_ratio=None,
    max_il_rna_wt_ratio=None 
):
    """
    Vectorizes an LNP library.

    3 types of vectorization:
    1. Lipid only: Vectorizes the lipid components only. Requires lipids_only to be specified.
    2. Full: Vectorizes the lipid components, volumetric ratio, and IL RNA wt ratio. Extracts numerator only from volumetric and IL RNA wt ratios. Set lipids_only to False, max_volumetric_ratio to None, and max_il_rna_wt_ratio to None.
    3. Full with normalization: Vectorizes the lipid components, volumetric ratio, and IL RNA wt ratio. Normalizes the volumetric ratio and IL RNA wt ratio by their respective max values. Set lipids_only to False, max_volumetric_ratio to the max volumetric ratio, and max_il_rna_wt_ratio to the max IL RNA wt ratio.

    Args:
        lipids_only:
        library (dict): LNP entry.
        max_volumetric_ratio (float): Max volumetric ratio.
        max_il_rna_wt_ratio (float): Max IL RNA wt ratio.

    Returns:
        ndarray: Vectorized LNP entry.
    """

    def vectorize_components(components, vector):
        for component in components:
            # Handle ILs (cluster)
            if component["component_type"] == "IL":
                cluster = il_clusters[component["smi"]]["cluster"]
                vector[vector_index_mapping[cluster]] = component["mol"] if il_mol_pcts else 1.0
            else: # Handle other lipid components
                vector[vector_index_mapping[component["smi"]]] = component["mol"]

    def vectorize_excipients(excipients, vector):
        for excipient in excipients:
            # Get the cluster that the booster is part of
            cluster = booster_clusters[excipient["smi"]]["cluster"]
            vector[vector_index_mapping[cluster]] = 1.0 # 1 FOR EACH EXCIPIENT/BOOSTER

    def vectorize_vr(vr, vector):
        volumetricRatioNumerator = numeratorFromRatio(vr)
        if max_volumetric_ratio:
            vector[vector_index_mapping["volumetric_ratio"]] = (
                volumetricRatioNumerator / max_volumetric_ratio
            )
        else:
            vector[vector_index_mapping["volumetric_ratio"]] = volumetricRatioNumerator

    def vectorize_ilrna(ilrna, vector):
        ILRNAWtRatioNumerator = numeratorFromRatio(ilrna)
        if max_il_rna_wt_ratio:
            vector[vector_index_mapping["il_rna_wt_ratio"]] = (
                ILRNAWtRatioNumerator / max_il_rna_wt_ratio
            )
        else:
            vector[vector_index_mapping["il_rna_wt_ratio"]] = ILRNAWtRatioNumerator

    def vectorize(entry: dict):
        """
        Vectorizes an LNP entry.
        Args:
            entry (dict): LNP entry.
        Returns:
            list: Vectorized LNP entry.
        """
        vector = [0] * len(vector_index_mapping)

        for feature in features:
            if feature == "components":
                vectorize_components(entry["components"], vector)
            elif excipients and feature == "excipients":
                vectorize_excipients(entry["excipients"], vector)
            elif ratios and feature == "il_rna_wt_ratio":
                vectorize_ilrna(entry["il_rna_wt_ratio"], vector)
            elif ratios and feature == "volumetric_ratio":
                vectorize_vr(entry["volumetric_ratio"], vector)

        return vector

    assert (max_volumetric_ratio == None and max_il_rna_wt_ratio == None) or (
        max_volumetric_ratio != None and max_il_rna_wt_ratio != None
    ), "Both max volumetric ratio and max IL RNA wt ratio must be provided."

    # Join the nested lipid smiles template into one list of names.
    vector_index_mapping = {}
    index = 0

    with open(lipid_smiles_path, "r") as f:
        lipid_smiles = json.load(f)
    with open(booster_clusters_path, "r") as f:
        booster_clusters = json.load(f)
    with open(il_clusters_path, "r") as f:
        il_clusters = json.load(f)

    # Add the booster clusters to the index mapping
    for smi, attrs in booster_clusters.items():
        if attrs["cluster"] not in vector_index_mapping.keys():
            vector_index_mapping[attrs["cluster"]] = index
            index += 1

    # Add the IL clusters to the index mapping
    for smi, attrs in il_clusters.items():
        if attrs["cluster"] not in vector_index_mapping.keys():
            vector_index_mapping[attrs["cluster"]] = index
            index += 1

    # Add HL, CH, and PEG lipid SMILES
    for lipid_type in lipid_smiles.values():
        if lipid_type != "IL": # ILs covered through clustering
            for name in lipid_type.keys():
                vector_index_mapping[name] = index
                index += 1

    if ratios:
        vector_index_mapping.update(
            {
                "volumetric_ratio": index,
                "il_rna_wt_ratio": index + 1,
            }
        )

    vectors = []
    for entry in library.values():
        vectors.append(vectorize(entry))
    vectors = np.array(vectors)

    return vectors

def full_vectorization(
    library: dict,
    lipid_smiles_path,
    pbae_smiles_path,
    lipids_only=False,
    max_volumetric_ratio=None,
    max_il_rna_wt_ratio=None,
    with_pbae=False,
):
    """
    Vectorizes an LNP library.

    3 types of vectorization:
    1. Lipid only: Vectorizes the lipid components only. Requires lipids_only to be specified.
    2. Full: Vectorizes the lipid components, volumetric ratio, and IL RNA wt ratio. Extracts numerator only from volumetric and IL RNA wt ratios. Set lipids_only to False, max_volumetric_ratio to None, and max_il_rna_wt_ratio to None.
    3. Full with normalization: Vectorizes the lipid components, volumetric ratio, and IL RNA wt ratio. Normalizes the volumetric ratio and IL RNA wt ratio by their respective max values. Set lipids_only to False, max_volumetric_ratio to the max volumetric ratio, and max_il_rna_wt_ratio to the max IL RNA wt ratio.

    Args:
        lipids_only:
        library (dict): LNP entry.
        max_volumetric_ratio (float): Max volumetric ratio.
        max_il_rna_wt_ratio (float): Max IL RNA wt ratio.

    Returns:
        ndarray: Vectorized LNP entry.
    """

    def vectorize(entry: dict):
        """
        Vectorizes an LNP entry.
        Args:
            entry (dict): LNP entry.
        Returns:
            list: Vectorized LNP entry.
        """
        vector = [0] * len(vector_index_mapping)
        for key, value in entry.items():
            if key == "components":
                for component in value:
                    vector[vector_index_mapping[component["smi"]]] = component["mol"]
            # TODO: This is a hacky way to check whether we are doing lipid vectorization only or full.
            elif lipids_only:
                continue
            elif key == "volumetric_ratio":
                volumetricRatioNumerator = numeratorFromRatio(value)
                if max_volumetric_ratio:
                    vector[vector_index_mapping[key]] = (
                        volumetricRatioNumerator / max_volumetric_ratio
                    )
                else:
                    vector[vector_index_mapping[key]] = volumetricRatioNumerator
            elif key == "il_rna_wt_ratio":
                ILRNAWtRatioNumerator = numeratorFromRatio(value)
                if max_il_rna_wt_ratio:
                    vector[vector_index_mapping[key]] = (
                        ILRNAWtRatioNumerator / max_il_rna_wt_ratio
                    )
                else:
                    vector[vector_index_mapping[key]] = ILRNAWtRatioNumerator
        return vector

    assert (max_volumetric_ratio == None and max_il_rna_wt_ratio == None) or (
        max_volumetric_ratio != None and max_il_rna_wt_ratio != None
    ), "Both max volumetric ratio and max IL RNA wt ratio must be provided."

    # Use lipid smiles as source of truth for vector index mapping.
    lipid_smiles_file_path = lipid_smiles_path
    pbae_smiles_file_path = pbae_smiles_path

    # Join the nested lipid smiles template into one list of names.
    vector_index_mapping = {}
    index = 0
    if with_pbae:
        with open(pbae_smiles_file_path, "r") as f:
            lipid_smiles = json.load(f)
        for name in lipid_smiles.keys():
            vector_index_mapping[name] = index
            index += 1

    with open(lipid_smiles_file_path, "r") as f:
        lipid_smiles = json.load(f)

    for lipid_type in lipid_smiles.values():
        for name in lipid_type.keys():
            vector_index_mapping[name] = index
            index += 1

    if not lipids_only:
        vector_index_mapping.update(
            {
                "volumetric_ratio": index,
                "il_rna_wt_ratio": index + 1,
            }
        )

    vectors = []
    for entry in library.values():
        vectors.append(vectorize(entry))
    vectors = np.array(vectors)

    return vectors


def loadFractionalLibrary(paths: list, fraction=None, rankingLabel=None, K=None):
    """
    Loads a specified fraction of the entries from the list of files. If ranking label is provided, the fraction of the library is sampled from the top entries.
    Args:
        path (list): List of files to sample entries from.
        fraction (float): Fraction of the entries to load.
        rankingLabel (str): Label to rank the entries by. If None, the entries are sampled randomly.

    Returns:
        dict: Sample of entries.
    """
    # For ranking, both rankingLabel and topK must be provided.
    if rankingLabel is not None or K is not None:
        assert (
            rankingLabel is not None and K is not None
        ), "Both ranking label and topK must be provided."

    virtualLibrary = {}
    for fileNum, file in enumerate(paths):
        with open(file, "r") as f:
            fileEntries = json.load(f)
            for key, value in fileEntries.items():
                if rankingLabel is not None or np.random.random() < fraction:
                    virtualLibrary[key] = value

    if rankingLabel is not None:
        # Sort the entries by the ranking label.
        topKIdx = len(virtualLibrary) * K
        virtualLibrary = dict(
            sorted(
                virtualLibrary.items(),
                key=lambda item: item[1]["labels"][rankingLabel],
                reverse=True,
            )[:topKIdx]
        )

    return virtualLibrary


def hash_from_dict(dict):
    """
    Generates a hash from a dictionary.
    Args:
        dict (dict): Dictionary to hash.

    Returns:
        str: Hash of the dictionary.
    """
    return hashlib.md5(json.dumps(dict).encode(), usedforsecurity=False).hexdigest()


def getMaxVolumetricAndILRNAWtRatio(dataset: dict):
    """
    Gets the max volumetric and IL RNA wt ratio from a dataset containing LNP entries.
    Args:
        dataset (dict): Dictionary containing LNP entries.

    Returns:
        tuple: Max volumetric ratio and max IL RNA wt ratio.
    """
    maxVolumetricRatio = 0
    maxILRNAWTRatio = 0
    for formulation in dataset.values():
        # Strip the numerator.
        volumetricRatio = formulation["volumetric_ratio"]
        volumetricRatioNumerator = numeratorFromRatio(volumetricRatio)
        maxVolumetricRatio = max(maxVolumetricRatio, volumetricRatioNumerator)

        ILRNAWtRatio = formulation["il_rna_wt_ratio"]
        ILRNAWtRatioNumerator = numeratorFromRatio(ILRNAWtRatio)
        maxILRNAWTRatio = max(maxILRNAWTRatio, ILRNAWtRatioNumerator)

    return maxVolumetricRatio, maxILRNAWTRatio


def names_from_entry(entry):
    """
    Unpacks and arranges the component names from a single-IL LNP entry.
    Args:
        entry (dict): LNP entry.

    Returns:
        list: Component names ordered as IL, HL, CH, and PEG.
    """
    # Read the dictionary mapping component type -> smile -> name.
    # Convert it to a smile -> name dictionary.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    smiles_file_path = os.path.join(script_dir, "../misc/lipid_smiles.json")

    with open(smiles_file_path, "r") as f:
        componentTypeToSmileToNameMap = json.load(f)

    smileToNameMap = {}
    for componentDict in componentTypeToSmileToNameMap.values():
        smileToNameMap.update(componentDict)

    component_names = [None for _ in range(4)]
    component_indices = {"IL": 0, "HL": 1, "CH": 2, "PEG": 3}

    for component in entry["components"]:
        component_type = component["component_type"]
        if component_type not in component_indices:
            continue

        component_index = component_indices[component_type]
        if component_names[component_index] is not None:
            raise ValueError(f"Multiple {component_type} components are not supported.")
        component_names[component_index] = smileToNameMap[component["smi"]]

    return component_names


def mols_from_entry(entry):
    """
    Unpacks the total molarity for each lipid type from an LNP entry.

    Historical reference entries with multiple components of the same type are
    aggregated. Newly generated virtual-library entries contain exactly one IL.
    Args:
        entry (dict): LNP entry.

    Returns:
        ndarray: Length 4 list containing the mol percentages.
    """
    molarPercentages = np.zeros(4)
    for component in entry["components"]:
        if component["component_type"] == "IL":
            molarPercentages[0] += component["mol"]
        elif component["component_type"] == "HL":
            molarPercentages[1] += component["mol"]
        elif component["component_type"] == "CH":
            molarPercentages[2] += component["mol"]
        elif component["component_type"] == "PEG":
            molarPercentages[3] += component["mol"]
    return molarPercentages


def smiles_from_entry(entry):
    """Returns component SMILES ordered as IL, HL, CH, and PEG."""
    smiles = [None for _ in range(4)]
    component_indices = {"IL": 0, "HL": 1, "CH": 2, "PEG": 3}

    for component in entry["components"]:
        component_type = component["component_type"]
        if component_type not in component_indices:
            continue

        component_index = component_indices[component_type]
        if smiles[component_index] is not None:
            raise ValueError(f"Multiple {component_type} components are not supported.")
        smiles[component_index] = component["smi"]

    return smiles
