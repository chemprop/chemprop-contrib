# data files "AqSolDBc.csv" and "OChemUnseen.csv" should be downloaded from:
# https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CZVZIA

import pandas as pd
from rdkit.Chem import CanonSmiles, MolFromSmiles


def _f(smiles):
    try:
        assert MolFromSmiles(smiles) is not None
        return True
    except:
        return False


if __name__ == "__main__":
    train = pd.read_csv("AqSolDBc.csv")
    train = train[train["SmilesCurated"].map(_f)]
    test = pd.read_csv("OChemUnseen.csv")
    test = test[test["SMILES"].map(_f)]
    train["canon_smiles"] = train["SMILES"].map(CanonSmiles)
    test["canon_smiles"] = test["SMILES"].map(CanonSmiles)
    train = train[~train["canon_smiles"].isin(set(test["canon_smiles"]))]
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
