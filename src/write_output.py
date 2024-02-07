from typing import TextIO

from src.detector_features import get_maxEnergy_sm_mM
from src.fem_handler import FEMBase


def write_txt_toNN(
    det_event: list[list],
    FEM_instance: FEMBase,
    sm_mM_map: dict,
    local_coord_dict: dict,
    file_stream: TextIO,
) -> None:
    """
    Write the event data to a text file.

    Parameters:
    det_event (list): The event data.
    FEM_instance (FEMBase): The FEM instance.
    sm_mM_map (dict): The mapping of the channels to the mod and mM.
    local_coord_dict (dict): The local coordinates of the channels.
    file_path (str): The path to the text file to write to.

    Returns:
    None. The function writes the event data to a text file.
    """
    # Find the maximum energy mM and sm in the event
    max_mM, max_sm, _ = get_maxEnergy_sm_mM(det_event, sm_mM_map)

    # Get the position of the rows and cols according to the local coordinates
    num_rows_cols = 8
    x_pos = {
        FEM_instance.x_pitch * (FEM_instance.x_pitch / 2 + i - 1): i
        for i in range(num_rows_cols)
    }
    y_pos = {
        FEM_instance.y_pitch * (FEM_instance.y_pitch / 2 + i - 1): i
        for i in range(num_rows_cols)
    }

    # Sum rows and columns energy for the max_mM
    rows_energy = [0] * 8
    cols_energy = [0] * 8
    for ch in det_event:
        if sm_mM_map[ch[2]][1] == max_mM:
            row_pos = x_pos[local_coord_dict[ch[2]][0]]
            col_pos = y_pos[local_coord_dict[ch[2]][1]]
            rows_energy[row_pos] += ch[1]
            cols_energy[col_pos] += ch[1]

    file_stream.write(
        "\t".join(f"{item:.2f}" for item in cols_energy)
        + "\t"
        + "\t".join(f"{item:.2f}" for item in rows_energy)
        + "\t"
        + f"{max_mM}\t{max_sm}\n"
    )
