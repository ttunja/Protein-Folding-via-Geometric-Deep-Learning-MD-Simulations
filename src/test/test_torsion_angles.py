import numpy as np

def calc_dihedral(u1, u2, u3, u4):
    """ Calculate dihedral angle method. From bioPython.PDB
    (adapted to np.array)
    Calculate the dihedral angle between 4 vectors
    representing 4 connected points. The angle is in
    degrees.
    """

    a1 = u2 - u1
    a2 = u3 - u2
    a3 = u4 - u3

    v1 = np.cross(a1, a2)
    v1 = v1 / (v1 * v1).sum(-1)**0.5
    v2 = np.cross(a2, a3)
    v2 = v2 / (v2 * v2).sum(-1)**0.5
    porm = np.sign((v1 * a3).sum(-1))
    rad = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)
    if not porm == 0:
        rad = rad * porm

    return np.degrees(rad)  # Convert radians to degrees


def calculate_torsion_angle(atom1, atom2, atom3, atom4):
    """
    Calculate the torsion angle formed by four atoms.

    Args:
    atom1, atom2, atom3, atom4 (np.array): The coordinates of the four atoms.

    Returns:
    float: Torsion angle in degrees.
    """

    # Vectors between the atoms
    b1 = atom2 - atom1
    b2 = atom3 - atom2
    b3 = atom4 - atom3

    # Normal vectors to the planes formed by the first three and last three atoms
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 = n1.astype(np.float64) / np.linalg.norm(n1)
    n2 = n2.astype(np.float64) / np.linalg.norm(n2)

    # Calculate the torsion angle
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.degrees(np.arctan2(y, x))

def get_torsion_angle_numpy(atom1, atom2, atom3, atom4):
    # Vector calculations using numpy
    b1 = atom2 - atom1
    b2 = atom3 - atom2
    b3 = atom4 - atom3

    # Normalize b2
    b2_norm = b2 / np.linalg.norm(b2)

    # Calculate projections of c and d onto the plane defined by a, b
    c_project_ab = atom1 + np.dot(atom3 - atom1, b2_norm) * b2_norm
    d_project_ab = atom1 + np.dot(atom4 - atom1, b2_norm) * b2_norm

    # Shift d to be in the plane of c
    dshifted = atom4 - d_project_ab + c_project_ab

    # Calculate vectors for angle calculation
    vector1 = dshifted - c_project_ab
    vector2 = atom3 - c_project_ab

    # Cosine of the angle
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure the value is in the valid range for arccos

    # Calculate angle and adjust sign
    angle = np.arccos(cos_angle)
    sign = np.sign(np.dot(np.cross(vector1, vector2), b2))
    torsion_angle = np.degrees(angle) * sign

    return torsion_angle



################################################################## INVERSE ##############################################################

from scipy.spatial.transform import Rotation as R

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def adjust_torsion_angle(atom1, atom2, atom3, atom4, current_angle, desired_angle):
    """
    Adjust the torsion angle of a set of four atoms.
    :param atoms: Coordinates of the four atoms (list of 3D tuples).
    :param current_angle: Current torsion angle in degrees.
    :param desired_angle: Desired torsion angle in degrees.
    :return: New coordinates of the four atoms.
    """

    # Calculate the rotation axis (bond between atom2 and atom3)
    rotation_axis = atom3 - atom2

    # Convert angles to radians
    current_angle_rad = np.radians(current_angle)
    desired_angle_rad = np.radians(desired_angle)

    # Calculate the rotation angle required
    rotation_angle_rad = desired_angle_rad - current_angle_rad

    # Create rotation matrix
    rot_matrix = rotation_matrix(rotation_axis, rotation_angle_rad)

    # Adjust position of atom4
    atom4_rotated = np.dot(rot_matrix, atom4 - atom3) + atom3

    return atom4_rotated

def calculate_centroid(atoms):
    return np.mean(atoms, axis=0)

def translate_atoms(atoms, translation_vector):
    return atoms + translation_vector

def calculate_rmsd(atoms_new, atoms_ref):
    diff = atoms_new - atoms_ref
    return np.sqrt(np.mean(np.square(diff)))

def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom2 - atom1)

def scale_and_translate(atom1_ref, atom2_ref, atom3_ref, atom4_ref, distances):
    # Scale each segment
    atom2_ref = atom1_ref + distances[0] * (atom2_ref - atom1_ref) / calculate_distance(atom1_ref, atom2_ref)
    atom3_ref = atom2_ref + distances[1] * (atom3_ref - atom2_ref) / calculate_distance(atom2_ref, atom3_ref)
    atom4_ref = atom3_ref + distances[2] * (atom4_ref - atom3_ref) / calculate_distance(atom3_ref, atom4_ref)
    
    return atom1_ref, atom2_ref, atom3_ref, atom4_ref



if __name__=="__main__":
    import ipdb; ipdb.set_trace()
    atom1_ref = np.array([21.350, 31.325, 22.681])
    atom2_ref = np.array([22.409, 31.286, 21.483])
    atom3_ref = np.array([22.840, 29.751, 21.498])
    atom4_ref = np.array([23.543, 29.175, 22.594])
    torsion_angle1 = calc_dihedral(atom1_ref, atom2_ref, atom3_ref, atom4_ref)
    torsion_angle2 = calculate_torsion_angle(atom1_ref, atom2_ref, atom3_ref, atom4_ref)
    torsion_angle3 = get_torsion_angle_numpy(atom1_ref, atom2_ref, atom3_ref, atom4_ref)
    
    # Example usage
    desired_angle = -65.609  # new desired torsion angle in degrees
    atom1 = np.array([1, 0, 0])
    atom2 = np.array([0, 1, 0])
    atom3 = np.array([0, 0, 1])
    atom4 = np.array([1, 0, 1])

    current_angle = calc_dihedral(atom1, atom2, atom3, atom4)

    # Calculate original distances
    distances = [
        calculate_distance(atom1, atom2),
        calculate_distance(atom2, atom3),
        calculate_distance(atom3, atom4)
    ]

    # Scale and translate atom_ref set
    atom1_ref, atom2_ref, atom3_ref, atom4_ref = scale_and_translate(atom1_ref, atom2_ref, atom3_ref, atom4_ref, distances)
    atoms_ref = np.array([atom1_ref, atom2_ref, atom3_ref, atom4_ref])

    # TODO when I scale and translate the torsion angle changes
    # TODO wrong function used, it should be used calc_dihedral I think
    # TODO when I calc_dehidral od atoms_translated and atoms_new, I get different angle completely (4.92)
    # Adjust the position of atom4 
    # [ 0.86300247, -0.35723014,  0.64276986]
    # TODO ovo si sigurno ne radi kako bi trebalo, jer nakon ovoga bi trebao dobit desired angle
    atom4_new = adjust_torsion_angle(atom1, atom2, atom3, atom4, current_angle, desired_angle)

    atom4_new_new = adjust_torsion_angle(atom1, atom2, atom3, atom4_new, desired_angle, current_angle)

    atoms_new = np.array([atom1, atom2, atom3, atom4_new])

    # Calculate centroids
    centroid_new = calculate_centroid(atoms_new)
    centroid_ref = calculate_centroid(atoms_ref)

    # Translate the new atoms to superimpose
    translation_vector = centroid_ref - centroid_new
    atoms_translated = translate_atoms(atoms_new, translation_vector)
    # Calculate RMSD
    rmsd = calculate_rmsd(atoms_translated, atoms_ref)
    rmsd
    # NE MOZE BIT ISTO KAD NISU UDALJENI JEDNAKO BONDOVI
    print("Tin")
