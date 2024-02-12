from typing import Callable, TextIO

from src.utils import get_maxEnergy_sm_mM
from src.fem_handler import FEMBase


def write_txt_toNN(
    FEM_instance: FEMBase,
    sm_mM_map: dict,
    local_coord_dict: dict,
    chtype_map: dict,
    file_stream: TextIO,
) -> Callable:
    """
    Write the event data to a text file.

    Parameters:
    FEM_instance (FEMBase): The FEM instance.

    Returns:
    Callable: The function to write the event data to a text file.
    """

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

    def _write_txt_toNN(
        det_list: list[list],
    ):
        """
        Write the event data to a text file.

        Parameters:
        det_event (list): The event data.
        sm_mM_map (dict): The mapping of the channels to the mod and mM.
        local_coord_dict (dict): The local coordinates of the channels.
        file_stream (TextIO): The file stream to write to.
        """
        # Find the maximum energy mM and sm in the event
        max_mm_list, _ = get_maxEnergy_sm_mM(det_list, sm_mM_map, chtype_map)

        max_mM = sm_mM_map[max_mm_list[0][2]][1]
        max_sm = sm_mM_map[max_mm_list[0][2]][0]

        # Sum rows and columns energy for the max_mM
        rows_energy = [0] * 8
        cols_energy = [0] * 8
        for ch in det_list:
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

    return _write_txt_toNN
