# reference: https://github.com/matsunagalab/ColabBTR/blob/main/colabbtr/morphology.py
# mapping atom name to radius in nanometer
# POPE and POPG are radius for phosphorus atom in lipid headgroups
# Referenced Martini2 and Martini3 force field
# URL: https://www.nature.com/articles/s41592-021-01098-3#Sec13
# URL: https://pubs.acs.org/doi/10.1021/jp071097f

Atom2Radius = {
    "H": 0.120,
    "HE": 0.140,
    "B": 0.192,
    "C": 0.170,
    "CA": 0.170,
    "CB": 0.170,
    "CG": 0.170,
    "CG1": 0.170,
    "CG2": 0.170,
    "CG3": 0.170,
    "CD": 0.170,
    "CD1": 0.170,
    "CD2": 0.170,
    "CD3": 0.170,
    "CZ": 0.170,
    "CZ1": 0.170,
    "CZ2": 0.170,
    "CZ3": 0.170,
    "CE": 0.170,
    "CE1": 0.170,
    "CE2": 0.170,
    "CE3": 0.170,
    "CH": 0.170,
    "CH1": 0.170,
    "CH2": 0.170,
    "CH3": 0.170,
    "N": 0.155,
    "NE": 0.155,
    "NZ": 0.155,
    "ND1": 0.155,
    "ND2": 0.155,
    "NE1": 0.155,
    "NE2": 0.155,
    "NH1": 0.155,
    "NH2": 0.155,
    "O": 0.152,
    "OH": 0.152,
    "OG": 0.152,
    "OE1": 0.152,
    "OE2": 0.152,
    "OG1": 0.152,
    "OG2": 0.152,
    "OD1": 0.152,
    "OD2": 0.152,
    "OXT": 0.152,
    "F": 0.147,
    "MG": 0.173,
    "AL": 0.184,
    "SI": 0.210,
    "P": 0.180,
    "S": 0.180,
    "SD": 0.180,
    "SG": 0.180,
    "CL": 0.175,
    "AR": 0.188,
}

Residue2Radius = {
    "POPE": 0.235, # added
    "POPG": 0.235, # added
    "POP": 0.235, # added
    "K": 0.275,
    "CYS": 0.275,
    "PHE": 0.32,
    "LEU": 0.31,
    "TRP": 0.34,
    "VAL": 0.295,
    "ILE": 0.31,
    "MET": 0.31,
    "HIS": 0.305,
    "HSD": 0.305,
    "HSE": 0.305, # added
    "HSP": 0.305, # added
    "TYR": 0.325,
    "ALA": 0.25,
    "GLY": 0.225,
    "PRO": 0.28,
    "ASN": 0.285,
    "THR": 0.28,
    "SER": 0.26,
    "ARG": 0.33,
    "GLN": 0.30,
    "ASP": 0.28,
    "LYS": 0.32,
    "GLU": 0.295
}