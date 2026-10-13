from rdkit import Chem

from chemprop.utils import make_mol


def make_polymer_mol(
    smi: str,
    keep_h: bool = False,
    add_h: bool = False,
    ignore_stereo: bool = False,
    reorder_atoms: bool = False,
) -> Chem.Mol:
    """
    Builds an RDKit molecule from a SMILES string containing multiple components (e.g. monomers).

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool, optional
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
        Default is False.
    add_h : bool, optional
        whether to add hydrogens to the molecule. Default is False.
    ignore_stereo : bool, optional
        If True, ignores stereochemical information (R/S and Cis/Trans) when constructing the molecule. Default is False.
    reorder_atoms : bool, optional
        whether to reorder the atoms in the molecule by their atom map numbers. This is useful when
        the order of atoms in the SMILES string does not match the atom mapping, e.g. '[F:2][Cl:1]'.
        Default is False. NOTE: This does not reorder the bonds.

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    # Create one molecule object per fragment and combine the fragments into
    # a single molecule object
    mols = []
    for s in smi.split("."):
        m = make_mol(s, keep_h, add_h, ignore_stereo, reorder_atoms)
        mols.append(m)
    # Combine all the mols into a single mol object
    mol = mols.pop(0)
    while len(mols) > 0:
        m2 = mols.pop(0)
        mol = Chem.CombineMols(mol, m2)

    return mol


def remove_wildcard_atoms(rwmol: Chem.Mol) -> Chem.Mol:
    """
    removes wildcard atoms from a RDKit Mol
    """
    indicies = [a.GetIdx() for a in rwmol.GetAtoms() if "*" in a.GetSmarts()]
    while len(indicies) > 0:
        rwmol.RemoveAtom(indicies[0])
        indicies = [a.GetIdx() for a in rwmol.GetAtoms() if "*" in a.GetSmarts()]
    Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)

    return rwmol
