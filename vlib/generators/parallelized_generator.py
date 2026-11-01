# Import libraries
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml
from tqdm import tqdm
import argparse
from pyprojroot import here
sys.path.insert(0, str(here()))

from vlib.utils.analysis_utils import mols_from_entry
from vlib.utils.generator_utils import (
    get_excipient_multiplier,
    get_excipient_smiles,
    get_il_rna_wt_ratios,
    get_lipid_smiles,
    get_molar_percentages_by_ranges,
    get_num_resulting_entries,
    get_volumetric_ratios,
    hamming_distance_checker,
)


def generate_library(
    molarity,
    lipid_smiles,
    il_rna_wt_ratios,
    volumetric_ratios,
    dataset_name,
    output_file_base_name,
    initial_entry_number,
    part_number,
    excipient_smiles,
    excipient_mode,
    excipient_molar_conc,
    random_seed=None,
    metadata=None,
):
    if metadata is None:
        metadata = {}

    def generate_entry(
        il_smile,
        hl_smile,
        ch_smile,
        peg_smile,
        il_molarity,
        hl_molarity,
        ch_molarity,
        peg_molarity,
        volumetric_ratio,
        il_rna_wt_ratio,
        excipient_smile,
    ):
        entry = {
            "components": [
                {
                    "smi": il_smile,
                    "component_type": "IL",
                    "mol": il_molarity,
                },
                {"smi": hl_smile, "component_type": "HL", "mol": hl_molarity},
                {"smi": ch_smile, "component_type": "CH", "mol": ch_molarity},
                {"smi": peg_smile, "component_type": "PEG", "mol": peg_molarity},
            ],
            "volumetric_ratio": volumetric_ratio,
            "il_rna_wt_ratio": il_rna_wt_ratio,
            "dataset_name": dataset_name,
            "excipients": [
                {
                    "smi": excipient_smile,
                    "component_type": "Excipient",
                    "molar_conc": excipient_molar_conc,
                }
            ],
        }
        # Add metadata to entry.
        for key, value in metadata.items():
            if key not in entry:
                entry[key] = value
            elif type(entry[key]) == list:
                entry[key] = value + entry[key]
            else:
                raise ValueError(
                    f"Metadata key {key} already exists in entry and is not a list."
                )

        il_count = sum(
            component["component_type"] == "IL"
            for component in entry["components"]
        )
        if il_count != 1:
            raise ValueError(
                f"Virtual-library entries must contain exactly one IL; found {il_count}."
            )

        if len(entry["excipients"]) != 1:
            raise ValueError("Virtual-library entries must contain exactly one excipient.")

        return entry

    excipient_multiplier = get_excipient_multiplier(
        len(excipient_smiles), excipient_mode
    )
    num_entries = (
        len(lipid_smiles)
        * len(il_rna_wt_ratios)
        * len(volumetric_ratios)
        * excipient_multiplier
    )
    entry_number = initial_entry_number
    final_entry_number = initial_entry_number + num_entries - 1
    output_filename = (
        f"{output_file_base_name}_{initial_entry_number}-{final_entry_number}.json"
    )
    rng = np.random.default_rng(random_seed)

    with open(output_filename, "x") as output_file:
        output_file.write("{")
        is_first_entry = True

        for lipid_smile in lipid_smiles:
            for il_rna_wt_ratio in il_rna_wt_ratios:
                for volumetric_ratio in volumetric_ratios:
                    if excipient_mode == "sample":
                        selected_excipients = [
                            excipient_smiles[rng.integers(len(excipient_smiles))]
                        ]
                    else:
                        selected_excipients = excipient_smiles

                    for excipient_smile in selected_excipients:
                        entry_key = f"part_{part_number}_{int(entry_number)}"
                        entry = generate_entry(
                            lipid_smile[0],
                            lipid_smile[1],
                            lipid_smile[2],
                            lipid_smile[3],
                            molarity[0],
                            molarity[1],
                            molarity[2],
                            molarity[3],
                            volumetric_ratio.item(),
                            il_rna_wt_ratio.item(),
                            excipient_smile,
                        )

                        if not is_first_entry:
                            output_file.write(",")
                        output_file.write("\n  ")
                        json.dump(entry_key, output_file)
                        output_file.write(": ")
                        json.dump(entry, output_file)

                        is_first_entry = False
                        entry_number += 1

        output_file.write("\n}\n")

    if entry_number != final_entry_number + 1:
        raise RuntimeError(
            f"Expected to generate {num_entries} entries, generated "
            f"{entry_number - initial_entry_number}."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["parallelized_generator"]

    dataset_name = config["dataset_name"]
    output_path_and_base_name = config["output_path_and_base_name"]
    part_number = config["part_number"]
    excipient_smiles = get_excipient_smiles(config["excipient_smiles_path"])
    excipient_mode = config["excipient_mode"]
    excipient_molar_conc = config["excipient_molar_conc"]
    random_seed = config.get("random_seed")

    # IL, CH, PEG
    lipid_base_values = config["lipid_base_values"]
    il_molar_config = config["il_molar_config"]
    hl_molar_config = config["hl_molar_config"]
    ch_molar_config = config["ch_molar_config"]
    peg_molar_config = config["peg_molar_config"]

    molarities = get_molar_percentages_by_ranges(
        lipid_base_values,
        il_molar_config,
        ch_molar_config,
        hl_molar_config,
        peg_molar_config,
    )

    # Read the top K percent B16F10 and DC24 formulations.
    real_dataset_path = config["real_dataset_path"]
    with open(real_dataset_path, "r") as f:
        real_library_dict = json.load(f)

    top_k_of_real_dataset = config["top_k_of_real_dataset"]

    # Get the top k fraction of the real dataset.
    top_k_idx = int(len(real_library_dict) * top_k_of_real_dataset)
    top_k_b16_f10_formulations = sorted(
        real_library_dict.values(),
        key=lambda x: x["labels"]["in_house_lnp_B16F10_luc"],
        reverse=True,
    )[:top_k_idx]
    top_k_dc24_formulations = sorted(
        real_library_dict.values(),
        key=lambda x: x["labels"]["in_house_lnp_DC24_luc"],
        reverse=True,
    )[:top_k_idx]

    # Get the molarities of the top K percent formulations.
    b16_f10_molarities = np.array(
        [
            mols_from_entry(entry)
            for entry in top_k_b16_f10_formulations
        ]
    )
    dc24_molarities = np.array(
        [
            mols_from_entry(entry)
            for entry in top_k_dc24_formulations
        ]
    )

    # Determine whether the generated molarities are within the hamming distance of the top K percent formulations.
    hamming_distance = config["hamming_distance"]
    molarities_in_b16_f10_hamming_distance = molarities[
        hamming_distance_checker(molarities, b16_f10_molarities, hamming_distance)
    ]
    molarities_in_dc24_hamming_distance = molarities[
        hamming_distance_checker(molarities, dc24_molarities, hamming_distance)
    ]

    lipid_smiles_path = config["lipid_smiles_path"]
    lipid_smiles = get_lipid_smiles(lipid_smiles_path)

    il_rna_wt_ratio_config = config["il_rna_wt_ratio_config"]
    il_rna_wt_ratios = get_il_rna_wt_ratios(il_rna_wt_ratio_config)

    # Not actually a range, just choosing either 1:1 or 3:1
    volumetric_ratio_config = config["volumetric_ratio_config"]
    volumetric_ratios = get_volumetric_ratios(volumetric_ratio_config)

    # Make output directories.
    #check_if_directory_is_empty(os.path.dirname(output_path_and_base_name))
    os.makedirs(os.path.dirname(output_path_and_base_name), exist_ok=True)

    # Compute the number of entries per process if each process only generates entries for a particular molarity.
    entries_per_process = get_num_resulting_entries(
        [None],
        lipid_smiles,
        il_rna_wt_ratios,
        volumetric_ratios,
        len(excipient_smiles),
        excipient_mode,
    )
    total_entries = get_num_resulting_entries(
        molarities,
        lipid_smiles,
        il_rna_wt_ratios,
        volumetric_ratios,
        len(excipient_smiles),
        excipient_mode,
    )

    print(f"{total_entries} entries will be generated.")

    start = time.time()
    results = []
    with Pool() as pool:
        for i, molarity in enumerate(molarities):
            process_seed = None if random_seed is None else random_seed + i

            results.append(
                pool.apply_async(
                    generate_library,
                    args=(
                        molarity,
                        lipid_smiles,
                        il_rna_wt_ratios,
                        volumetric_ratios,
                        dataset_name,
                        output_path_and_base_name,
                        i * entries_per_process,
                        part_number,
                        excipient_smiles,
                        excipient_mode,
                        excipient_molar_conc,
                        process_seed,
                    ),
                )
            )

        for result in tqdm(results, "Generating entries..."):
            result.get()

        pool.close()
        pool.join()
    stop = time.time()
    print(f"Time elapsed: {stop - start} seconds.")
