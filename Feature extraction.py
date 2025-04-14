import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import iFeatureOmegaCLI
import re
import math


def DDE(fastas, **kw):
    """
    DDE feature extraction
    """
    AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6, 'M': 1, 'N': 2,
                'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2}

    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#'] + diPeptides
    encodings.append(header)
    myTM = [(myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61) for pair in diPeptides]
    AADict = {aa: i for i, aa in enumerate(AA)}

    for item in fastas:
        name, sequence = item[0], re.sub('-', '', item[1])
        tmpCode = [0] * 400
        for j in range(len(sequence) - 1):
            if sequence[j] in AADict and sequence[j + 1] in AADict:
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] += 1
        if sum(tmpCode) != 0:
            tmpCode = [x / sum(tmpCode) for x in tmpCode]
        myTV = [(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1)) for j in range(len(myTM))]
        tmpCode = [(tmpCode[j] - myTM[j]) / math.sqrt(myTV[j]) if myTV[j] != 0 else 0 for j in range(len(tmpCode))]
        encodings.append([name] + tmpCode)
    return encodings


def feature_DDE(file_path):
    """
    Read sequences from a FASTA file, extract features using DDE method,
    and return a DataFrame with feature names as DDE1, DDE2, ...
    """
    fasta_list = open(file_path, 'r', encoding='utf-8').readlines()
    aa_feature_list = []
    # Every two lines form one record, the first line is the header, the second line is the sequence
    for flag in range(0, len(fasta_list), 2):
        fasta_str = [[fasta_list[flag].strip(), fasta_list[flag + 1].strip()]]
        dpc_output = DDE(fasta_str)
        # dpc_output[0] is the header, dpc_output[1] is the data, excluding the first item (name)
        dpc_features = dpc_output[1][1:]
        aa_feature_list.append(dpc_features)
    aa_feature_list = pd.DataFrame(aa_feature_list)
    aa_feature_list.columns = [f'DDE{i + 1}' for i in range(aa_feature_list.shape[1])]
    return aa_feature_list


def generate_features(input_txt_path):
    """
    Generate all descriptor features and return the combined DataFrame
    """
    descriptors = [
        "AAC", "PAAC", "DPC type 2", "CTDC", "CTDT", "CTDD",
        "CKSAAGP type 2", "QSOrder",
        "EAAC", "CKSAAP type 1", "CTriad", "ASDC", "GAAC"
    ]
    features = []

    for descriptor in descriptors:
        protein_descriptor = iFeatureOmegaCLI.iProtein(input_txt_path)
        try:
            protein_descriptor.get_descriptor(descriptor)
            protein_descriptor.display_feature_types()
            if protein_descriptor.encodings is None:
                print(f"Failed to extract {descriptor} features.")
            else:
                protein_descriptor.encodings = protein_descriptor.encodings.reset_index(drop=True)
                print(f"{descriptor} feature shape: {protein_descriptor.encodings.shape}")
                features.append(protein_descriptor.encodings)
        except Exception as e:
            print(f"Error extracting {descriptor} features: {e}")

    dde = feature_DDE(input_txt_path).reset_index(drop=True)
    print(f"DDE feature shape: {dde.shape}")
    features.append(dde)

    # Concatenate all features horizontally
    result = pd.concat(features, axis=1)
    print(f"Total combined feature shape: {result.shape}")
    return result


def get_labels(file_path):
    """
    Generate labels based on the header in the FASTA file:
    If the header starts with ">pos", label is 1; otherwise, it's 0.
    """
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Every two lines form one record, header is in even-numbered lines (starting from 0)
    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        if header.lower().startswith('>pos'):
            labels.append(1)
        else:
            labels.append(0)
    return labels


def main(input_txt_path, output_csv_path):
    """
    Extract all features and save them to a CSV file
    """
    features_df = generate_features(input_txt_path)
    # Get labels and add them to the DataFrame
    labels = get_labels(input_txt_path)
    features_df['Label'] = labels
    # Save as CSV file
    output_csv_path = output_csv_path.replace('.csv', '_13.csv')
    features_df.to_csv(output_csv_path, index=False)
    print(f"Feature matrix shape: {features_df.shape}")
    print(f"CSV file saved as: {output_csv_path}")


if __name__ == "__main__":
    input_txt_path = 'All_data.txt'
    output_csv_path = 'Feature.csv'
    main(input_txt_path, output_csv_path)
