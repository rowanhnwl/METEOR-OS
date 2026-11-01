import json
import os
import numpy as np


EXCIPIENT_MODES = {"sample", "full"}


def get_excipient_smiles(path):
    """Loads the non-empty SMILES values from an excipient name-to-SMILES file."""
    with open(path, "r") as f:
        excipient_info = json.load(f)

    if not isinstance(excipient_info, dict) or not excipient_info:
        raise ValueError("Excipient SMILES data must be a non-empty JSON object.")

    excipient_smiles = list(excipient_info.keys())
    if any(not isinstance(smi, str) or not smi.strip() for smi in excipient_smiles):
        raise ValueError("Every excipient must have a non-empty SMILES string.")
    if len(excipient_smiles) != len(set(excipient_smiles)):
        raise ValueError("Excipient SMILES strings must be unique.")

    return excipient_smiles


def get_excipient_multiplier(excipient_count, excipient_mode):
    """Returns how many excipient variants to generate for each base LNP."""
    if excipient_mode not in EXCIPIENT_MODES:
        choices = ", ".join(sorted(EXCIPIENT_MODES))
        raise ValueError(
            f"Invalid excipient_mode {excipient_mode!r}; expected one of: {choices}."
        )
    if excipient_count < 1:
        raise ValueError("At least one excipient SMILES string is required.")

    return 1 if excipient_mode == "sample" else excipient_count


def get_num_resulting_entries(
    molarities,
    lipid_smiles,
    il_rna_wt_ratios,
    volumetric_ratios,
    excipient_count,
    excipient_mode,
):
    """
    Returns the number of entries that will be generated given the parameters.
    """
    return (
        len(molarities)
        * len(lipid_smiles)
        * len(il_rna_wt_ratios)
        * len(volumetric_ratios)
        * get_excipient_multiplier(excipient_count, excipient_mode)
    )


def get_lipid_smiles(path):
    """
    Returns all combinations of lipid SMILES codes from the .json at path.

    Args:
        path (str): Path to .json file containing lipid SMILES codes.

    Returns:
        ndarray: 2-D array of lipid SMILES codes.
    """
    # Load lipid SMILES from .json
    with open(path) as f:
        lipidSmiles = json.load(f)

    ILSmiles = lipidSmiles["IL"]
    HLSmiles = lipidSmiles["HL"]
    CHSmiles = lipidSmiles["CH"]
    PEGSmiles = lipidSmiles["PEG"]

    ILSmiles = np.array(tuple(ILSmiles.keys())).reshape(-1, 1)
    HLSmiles = np.array(tuple(HLSmiles.keys())).reshape(-1, 1)
    CHSmiles = np.array(tuple(CHSmiles.keys())).reshape(-1, 1)
    PEGSmiles = np.array(tuple(PEGSmiles.keys())).reshape(-1, 1)

    # Generate all possible combinations of lipid SMILES
    lipidSmiles = np.array(
        np.meshgrid(ILSmiles, HLSmiles, CHSmiles, PEGSmiles)
    ).T.reshape(-1, 4)

    return lipidSmiles


def getMolarPercentagesByRangesOld(
    lipidBaseValues,
    ILMolarConfig,
    cholMolarConfig,
    HLMolarConfig,
    PEGMolarConfig,
):
    """
    Returns all molar combinations of lipid components that are within the given ranges. Generated values are either with a given step size or a given number of steps.

    Args:
        lipidBaseValues (list): [IL, chol, PEG]. The base values of each lipid component from which the molarities will vary from.
        ILMolarConfig (list): [min, max, stepSize OR numSteps]
        HLMolarConfig (list): [min, max, stepSize OR numSteps]
        cholMolarConfig (list): [min, max, stepSize OR numSteps]
        PEGMolarConfig (list): [min, max, stepSize OR numSteps]

    Returns:
        ndarray: 2-D array of molarities.
    """
    # Compute the number of steps given the step size.
    ILMin, ILMax, ILStepSize = ILMolarConfig
    HLMin, HLMax, HLStepSize = HLMolarConfig
    cholMin, cholMax, cholStepSize = cholMolarConfig
    PEGMin, PEGMax, PEGStepSize = PEGMolarConfig

    ILNumPoints = round((ILMax - ILMin) / ILStepSize) + 1
    HLNumPoints = round((HLMax - HLMin) / HLStepSize) + 1
    cholNumPoints = round((cholMax - cholMin) / cholStepSize) + 1
    PEGNumPoints = round((PEGMax - PEGMin) / PEGStepSize) + 1

    # Generate 1-D arrays of molarities. Round to 2 decimal places.
    ILMolarities = np.round(np.linspace(ILMin, ILMax, num=ILNumPoints), 2).reshape(
        -1, 1
    )
    HLMolarities = np.round(np.linspace(HLMin, HLMax, num=HLNumPoints), 2).reshape(
        -1, 1
    )
    cholMolarities = np.round(
        np.linspace(cholMin, cholMax, num=cholNumPoints), 2
    ).reshape(-1, 1)
    PEGMolarities = np.round(np.linspace(PEGMin, PEGMax, num=PEGNumPoints), 2).reshape(
        -1, 1
    )

    # Generate all possible combinations of molarities
    allMolarityCombinations = np.array(
        np.meshgrid(ILMolarities, HLMolarities, cholMolarities, PEGMolarities)
    ).T.reshape(-1, 4)

    # Filter out molarities that do not sum to 1
    allMolarityCombinations = allMolarityCombinations[
        np.isclose(np.sum(allMolarityCombinations, axis=1), 1)
    ]

    # Filter out molarities that contain negative molar percentages.
    allMolarityCombinations = allMolarityCombinations[
        np.all(allMolarityCombinations > 0, axis=1)
    ]

    # Prevent negative zero.
    allMolarityCombinations = np.abs(allMolarityCombinations)

    # Find indices at which components are equal to the base values.
    ILIsBaseValue = np.isclose(allMolarityCombinations[:, 0], lipidBaseValues[0])
    cholIsBaseValue = np.isclose(allMolarityCombinations[:, 2], lipidBaseValues[1])
    PEGIsBaseValue = np.isclose(allMolarityCombinations[:, 3], lipidBaseValues[2])

    # Choose values where only one component varies/other components are base values.
    ILVaryingMolarities = allMolarityCombinations[
        np.logical_and(cholIsBaseValue, PEGIsBaseValue)
    ]
    cholVaryingMolarities = allMolarityCombinations[
        np.logical_and(ILIsBaseValue, PEGIsBaseValue)
    ]
    PEGVaryingMolarities = allMolarityCombinations[
        np.logical_and(ILIsBaseValue, cholIsBaseValue)
    ]

    # When varying HL, the compensation by all other components must be equal.
    distanceFromBaseValues = allMolarityCombinations[:, [0, 2, 3]] - lipidBaseValues
    HLVaryingMolarities = allMolarityCombinations[
        np.logical_and(
            np.isclose(distanceFromBaseValues[:, 0], distanceFromBaseValues[:, 1]),
            np.isclose(distanceFromBaseValues[:, 0], distanceFromBaseValues[:, 2]),
        )
    ]

    # Vary ionizable lipid and PEG lipid.
    ILPEGVaryingMolarities = allMolarityCombinations[cholIsBaseValue]
    ILCholVaryingMolarities = allMolarityCombinations[PEGIsBaseValue]

    # Combine all valid molarities.
    # Ensure uniqueness by filtering out duplicates.
    validMolarities = np.concatenate(
        (
            ILVaryingMolarities,
            cholVaryingMolarities,
            PEGVaryingMolarities,
            HLVaryingMolarities,
            ILPEGVaryingMolarities,
            ILCholVaryingMolarities,
        )
    )
    validMolarities = np.unique(validMolarities, axis=0)

    return validMolarities


def get_molar_percentages_by_ranges(
    lipidBaseValues,
    ILMolarConfig,
    CHMolarConfig,
    HLMolarConfig,
    PEGMolarConfig,
):
    """
    Returns all molar combinations of lipid components that are within the given ranges. Generated values are either with a given step size or a given number of steps.

    Args:
        lipidBaseValues (list): [IL, HL, CH, PEG]. The base values of each lipid component from which the molarities will vary from.
        ILMolarConfig (list): [min, max, stepSize OR numSteps]
        HLMolarConfig (list): [min, max, stepSize OR numSteps]
        CHMolarConfig (list): [min, max, stepSize OR numSteps]
        PEGMolarConfig (list): [min, max, stepSize OR numSteps]

    Returns:
        ndarray: 2-D array of molarities.
    """
    # Compute the number of steps given the step size.
    ILMin, ILMax, ILStepSize = ILMolarConfig
    HLMin, HLMax, HLStepSize = HLMolarConfig
    CHMin, CHMax, CHStepSize = CHMolarConfig
    PEGMin, PEGMax, PEGStepSize = PEGMolarConfig

    ILNumPoints = round((ILMax - ILMin) / ILStepSize) + 1
    ILBaseValue, HLBaseValue, CHBaseValue, PEGBaseValue = lipidBaseValues

    # Generate 1-D arrays of molarities. Round to 2 decimal places.
    ILMolarities = np.round(np.linspace(ILMin, ILMax, num=ILNumPoints), 3).reshape(
        -1, 1
    )
    # TODO: Temporarily assigning only 4 PEG molarities.
    PEGMolarities = np.array([0.01, 0.025, 0.05, 0.1]).reshape(-1, 1)

    # Vary PEG:
    # 1. Compensate changes in PEG with IL, CH, and HL.
    # 2. Fix the relative amounts of IL, CH, and HL.
    remainingMolarPercentage = 1 - PEGMolarities
    ILRelativeAmount = ILBaseValue / (ILBaseValue + CHBaseValue + HLBaseValue)
    CHRelativeAmount = CHBaseValue / (ILBaseValue + CHBaseValue + HLBaseValue)
    ILCol = np.round(remainingMolarPercentage * ILRelativeAmount, 3)
    CHCol = np.round(remainingMolarPercentage * CHRelativeAmount, 3)
    HLCol = np.round(remainingMolarPercentage - ILCol - CHCol, 3)
    PEGVaryingMolarities = np.hstack((ILCol, HLCol, CHCol, PEGMolarities))

    # For each PEG molarity, generate molarities for all other methods.
    CHVaryingMolarities = []
    ILCholVaryingMolarities = []
    ILVaryingMolarities = []
    for baseValue in PEGVaryingMolarities:
        ILBaseValue, HLBaseValue, CHBaseValue, PEGBaseValue = baseValue

        # Vary IL:
        # 1. Fixed PEG at base value.
        # 2. Let CH and HL vary with a fixed relative amount between the two.
        PEGCol = PEGBaseValue * np.ones(ILMolarities.shape)
        CHHLRatio = CHBaseValue / (HLBaseValue + CHBaseValue)
        remainingMolarPercentage = np.ones(ILMolarities.shape) - ILMolarities - PEGCol
        CHCol = np.round(remainingMolarPercentage * CHHLRatio, 3)
        HLCol = np.round(np.ones(ILMolarities.shape) - ILMolarities - PEGCol - CHCol, 3)
        ILVaryingMolarities.append(np.hstack((ILMolarities, HLCol, CHCol, PEGCol)))

        # Vary CH:
        # 1. Fixed IL and PEG at base values.
        remainingMolarPercentage = 1 - ILBaseValue - PEGBaseValue
        CHStart = max(remainingMolarPercentage - HLMax, CHMin)
        CHEnd = min(remainingMolarPercentage - HLMin, CHMax)
        CHCol = np.arange(CHStart, CHEnd + CHStepSize, CHStepSize).reshape(-1, 1)
        HLCol = np.round(remainingMolarPercentage - CHCol, 3)
        CHVaryingMolarities.append(
            np.hstack(
                (
                    ILBaseValue * np.ones(HLCol.shape),
                    HLCol,
                    CHCol,
                    PEGBaseValue * np.ones(HLCol.shape),
                )
            )
        )

        # Vary IL and CH:
        # 1. Fixed PEG at base value.
        # 2. Sweep through ILs.
        # 3. For each IL, sweep through CH and HLs.
        for formulation in ILVaryingMolarities[-1]:
            # Varying CH and HLs is constrained by the proximity of each to the maxima (1) & minima (0)
            HL, CH = formulation[1], formulation[2]
            amountToVary = min(HL, CH, 1 - HL, 1 - CH)
            total = HL + CH

            if amountToVary == HL:
                HLCol = np.round(np.arange(HL, -HLStepSize, -HLStepSize), 3)
                CHCol = np.round(total - HLCol, 3)
            elif amountToVary == CH:
                CHCol = np.round(np.arange(CH, -CHStepSize, -CHStepSize), 3)
                HLCol = np.round(total - CHCol, 3)
            elif amountToVary == 1 - HL:
                HLCol = np.round(np.arange(HL, 1 + HLStepSize, HLStepSize), 3)
                CHCol = np.round(total - HLCol, 3)
            elif amountToVary == 1 - CH:
                CHCol = np.round(np.arange(CH, 1 + CHStepSize, CHStepSize), 3)
                HLCol = np.round(total - CHCol, 3)

            HLCol = HLCol.reshape(-1, 1)
            CHCol = CHCol.reshape(-1, 1)

            ILCholVaryingMolarities.append(
                np.concatenate(
                    (
                        formulation[0] * np.ones(HLCol.shape),
                        HLCol,
                        CHCol,
                        formulation[3] * np.ones(HLCol.shape),
                    ),
                    axis=1,
                )
            )

    ILVaryingMolarities = np.vstack(ILVaryingMolarities)
    CHVaryingMolarities = np.vstack(CHVaryingMolarities)
    ILCholVaryingMolarities = np.vstack(ILCholVaryingMolarities)

    # Combine all valid molarities.
    # Ensure uniqueness by filtering out duplicates.
    validMolarities = np.concatenate(
        (
            ILVaryingMolarities,
            CHVaryingMolarities,
            PEGVaryingMolarities,
            ILCholVaryingMolarities,
        )
    )

    # Filter out molarities that do not sum to 1
    validMolarities = validMolarities[np.isclose(np.sum(validMolarities, axis=1), 1)]

    # Filter out molarities that contain negative or zero molar percentages.
    validMolarities = validMolarities[np.all(validMolarities > 0, axis=1)]

    # Prevent negative zero.
    validMolarities = np.abs(validMolarities)

    # Ensure uniqueness.
    validMolarities = np.unique(validMolarities, axis=0)

    return validMolarities


def get_molar_percentages_by_hamming_distance(
    lipid_base_values,
    il_molar_config,
    hl_molar_config,
    ch_molar_config,
    peg_molar_config,
    hamming_distance,
    target_molar_percentage=1,
):
    """
    Returns all molar combinations of lipid components that are within the given ranges. Generated values are either with a given step size or a given number of steps.

    Args:
        hamming_distance (float): Maximum hamming distance between lipid ratios and the base values.
        lipid_base_values (list): [IL, HL, chol, PEG]. The base values of each lipid component from which the molarities will vary from.
        il_molar_config (list): [min, max, stepSize OR numSteps]
        hl_molar_config (list): [min, max, stepSize OR numSteps]
        ch_molar_config (list): [min, max, stepSize OR numSteps]
        peg_molar_config (list): [min, max, stepSize OR numSteps]
        target_molar_percentage (float): The target molar percentage of the lipid components.

    Returns:
        ndarray: 2-D array of molarities.
    """
    # Check if the number of steps or the step size is given
    ILMin, ILMax, ILStepSize = il_molar_config
    HLMin, HLMax, HLStepSize = hl_molar_config
    cholMin, cholMax, cholStepSize = ch_molar_config
    PEGMin, PEGMax, PEGStepSize = peg_molar_config

    ILNumPoints = round((ILMax - ILMin) / ILStepSize) + 1
    HLNumPoints = round((HLMax - HLMin) / HLStepSize) + 1
    cholNumPoints = round((cholMax - cholMin) / cholStepSize) + 1
    PEGNumPoints = round((PEGMax - PEGMin) / PEGStepSize) + 1

    # Generate 1-D arrays of molarities. Round to 2 decimal places.
    ILMolarities = np.linspace(
        ILMin, ILMax, num=ILNumPoints, dtype=np.float128
    ).reshape(-1, 1)
    HLMolarities = np.linspace(
        HLMin, HLMax, num=HLNumPoints, dtype=np.float128
    ).reshape(-1, 1)
    cholMolarities = np.linspace(
        cholMin, cholMax, num=cholNumPoints, dtype=np.float128
    ).reshape(-1, 1)
    PEGMolarities = np.linspace(
        PEGMin, PEGMax, num=PEGNumPoints, dtype=np.float128
    ).reshape(-1, 1)

    # Generate all possible combinations of molarities
    allMolarityCombinations = np.array(
        np.meshgrid(ILMolarities, HLMolarities, cholMolarities, PEGMolarities)
    ).T.reshape(-1, 4)

    # Filter out molarities that do not sum to the target molar percentage
    allMolarityCombinations = allMolarityCombinations[
        np.isclose(np.sum(allMolarityCombinations, axis=1), target_molar_percentage)
    ]

    # Ensure that molar percentages are at least 1%.
    allMolarityCombinations = allMolarityCombinations[
        np.all(
            np.logical_or(
                allMolarityCombinations >= 0.01,
                np.isclose(allMolarityCombinations, 0.01),
            ),
            axis=1,
        )
    ]

    # Prevent negative zero.
    allMolarityCombinations = np.abs(allMolarityCombinations)

    # By Hamming distance
    lipid_base_values = np.array(lipid_base_values, dtype=np.float128)
    validMolarities = allMolarityCombinations[
        np.logical_or(
            np.sum(np.abs(allMolarityCombinations - lipid_base_values), axis=1)
            <= hamming_distance,
            np.isclose(
                np.sum(np.abs(allMolarityCombinations - lipid_base_values), axis=1),
                hamming_distance,
            ),
        )
    ]

    validMolarities = np.unique(validMolarities, axis=0)
    validMolarities = validMolarities.astype(dtype=float)

    return validMolarities


def get_il_rna_wt_ratios(ILRNAWtRatioConfig):
    """
    Returns all possible IL:RNA weight ratios given the lower bound, upper bound, and step size.

    Args:
        ILRNAWtRatioConfig: (list) [min, max, stepSize]

    Returns:
        ndarray: 2-D array of IL:RNA weight ratios.
    """
    ILRNAWtRatioMin, ILRNAWtRatioMax, ILRNAWtRatioStepSize = ILRNAWtRatioConfig

    # Determine the number of points to generate.
    ILRNAWtRatioNumPoints = (
        int((ILRNAWtRatioMax - ILRNAWtRatioMin) / ILRNAWtRatioStepSize) + 1
    )

    # Generate the 1-D array of IL part of the weight ratio. Round to the nearest integer.
    ILRNAWtRatios = np.round(
        np.linspace(ILRNAWtRatioMin, ILRNAWtRatioMax, num=ILRNAWtRatioNumPoints)
    )

    # Include the RNA part of the weight ratio.
    ILRNAWtRatios = [str(int(ILWeight)) + ":1" for ILWeight in ILRNAWtRatios]
    ILRNAWtRatios = np.array(ILRNAWtRatios).reshape(-1, 1)

    return ILRNAWtRatios


def get_volumetric_ratios(volumetricRatioConfig):
    """
    Returns all possible volumetric ratios given the lower bound, upper bound, and step size.

    Args:
        volumetricRatioConfig: (list) [min, max, stepSize]

    Returns:
        ndarray: 2-D array of volumetric ratios.
    """
    (
        volumetricRatioMin,
        volumetricRatioMax,
        volumetricRatioStepSize,
    ) = volumetricRatioConfig

    # Determine the number of points to generate.
    volumetricRatioNumPoints = (
        int((volumetricRatioMax - volumetricRatioMin) / volumetricRatioStepSize) + 1
    )

    # Generate the 1-D array of the numerator of the volumetric ratio. Round to the nearest integer.
    volumetricRatios = np.round(
        np.linspace(
            volumetricRatioMin, volumetricRatioMax, num=volumetricRatioNumPoints
        )
    )

    # Include the denominator of the volumetric ratio.
    volumetricRatios = [
        str(int(volumetricRatio)) + ":1" for volumetricRatio in volumetricRatios
    ]
    volumetricRatios = np.array(volumetricRatios).reshape(-1, 1)

    return volumetricRatios


def serializeToFileName(input):
    """
    Converts input into a string suitable for a filename. Replaces periods with underscores.

    Args:
        input (ndarray, string, numerical): 1-D ndarray of molarities.

    Returns:
        str: String representation of molarities with periods replaced with underscores.
    """
    if isinstance(input, str):
        return input.replace(".", "_")
    elif isinstance(input, np.ndarray):
        strMolarity = ""

        for m in input:
            strMolarity += str(m).replace(".", "_") + "_"

        return strMolarity[:-1]
    elif isinstance(input, float):
        return str(input).replace(".", "_")


class process_counter:
    """
    A class that keeps a global counter of completed processes.
    """

    def __init__(self):
        self.current = 1

    def next(self):
        self.current += 1
        return int(self.current - 1)

    def reset(self):
        self.current = 1


def hamming_distance_checker(arr1, arr2, hammingDistance):
    """
    Computes a boolean array representing whether each row in arr1 is within hammingDistance of any row in arr2.
    """
    reshaped = arr2[:, np.newaxis, :]
    diff = np.sum(np.abs(arr1 - reshaped), axis=2) <= hammingDistance
    return np.any(diff, axis=0)
