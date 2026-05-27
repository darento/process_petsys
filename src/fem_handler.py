class FEMBase:
    """
    This class represents the base class for the FEM instances.

    Parameters:
        - x_pitch (float): The pitch of the x-axis.
        - y_pitch (float): The pitch of the y-axis.
        - sum_rows_cols (bool): A boolean indicating whether to sum the rows and columns.
        - channels (int): The number of channels.
        - mM_channels (int): The number of mM channels.
        - num_ASICS (int): The number of ASICS.
        - sum_row_offset (int): The top row index used to flip loc_y in sum_rows_cols mode.
    """

    def __init__(
        self,
        x_pitch: float,
        y_pitch: float,
        sum_rows_cols: bool,
        channels: int,
        mM_channels: int,
        num_ASICS: int,
        sum_row_offset: int,
    ):
        self.x_pitch = x_pitch
        self.y_pitch = y_pitch
        self.sum_rows_cols = sum_rows_cols
        self.channels = channels
        self.mM_channels = mM_channels
        self.num_ASICS = num_ASICS
        self.sum_row_offset = sum_row_offset

    def get_coordinates(self, channel_pos: int) -> tuple:
        """
        Returns the coordinates of a channel.

        The grid divisors are derived from the total number of channels, so the
        same implementation serves every FEM type:
            - FEM128 (channels=128): grid_dim=8, span=16
            - FEM256 (channels=256): grid_dim=16, span=32

        Parameters:
            - channel_pos (int): The position of the channel.

        Returns:
        tuple: The coordinates of the channel.
        """
        if not self.sum_rows_cols:
            # Number of columns in the (square-ish) channel grid: channels // 16.
            grid_dim = self.channels // 16
            row = channel_pos // grid_dim
            col = channel_pos % grid_dim
            loc_x = round((col + 0.5) * self.x_pitch, 2)
            loc_y = round((row + 0.5) * self.y_pitch, 2)
        else:
            # Channel span of a summed row/column layout: channels // 8.
            span = self.channels // 8
            loc_x = round(self.x_pitch / 2 + self.x_pitch * (channel_pos % span), 2)
            loc_y = round(
                self.y_pitch / 2
                + self.y_pitch * (self.sum_row_offset - channel_pos % span),
                2,
            )
        return (loc_x, loc_y)


class FEM128(FEMBase):
    """
    This class represents the FEM128 instance.

    Parameters:
        - x_pitch (float): The pitch of the x-axis.
        - y_pitch (float): The pitch of the y-axis.
        - mM_channels (int): The number of mM channels.
        - sum_rows_cols (bool): A boolean indicating whether to sum the rows and columns.
        - channels (int): The number of channels (typically 128).
    """

    def __init__(
        self,
        x_pitch: float,
        y_pitch: float,
        mM_channels: int,
        sum_rows_cols: bool,
        channels: int,
    ):
        super().__init__(
            x_pitch,
            y_pitch,
            sum_rows_cols,
            channels,
            mM_channels,
            2,
            sum_row_offset=channels // 16 - 1,
        )


class FEM256(FEMBase):
    """
    This class represents the FEM256 instance.

    Parameters:
        - x_pitch (float): The pitch of the x-axis.
        - y_pitch (float): The pitch of the y-axis.
        - mM_channels (int): The number of mM channels.
        - sum_rows_cols (bool): A boolean indicating whether to sum the rows and columns.
        - channels (int): The number of channels (typically 256).

    """

    def __init__(
        self,
        x_pitch: float,
        y_pitch: float,
        mM_channels: int,
        sum_rows_cols: bool,
        channels: int,
    ):
        super().__init__(
            x_pitch,
            y_pitch,
            sum_rows_cols,
            channels,
            mM_channels,
            4,
            sum_row_offset=channels // 8 - 1,
        )


def get_FEM_instance(
    FEM_type: str,
    x_pitch: float,
    y_pitch: float,
    mM_channels: int,
    sum_rows_cols: bool,
    channels: int,
) -> FEMBase:
    """
    Returns the FEM instance.

    Parameters:
        - FEM_type (str): The type of FEM.
        - x_pitch (float): The pitch of the x-axis.
        - y_pitch (float): The pitch of the y-axis.
        - mM_channels (int): The number of mM channels.
        - sum_rows_cols (bool): A boolean indicating whether to sum the rows and columns.
        - channels (int): The number of channels.

    Returns:
    FEMBase: The FEM instance.
    """
    if FEM_type == "FEM128":
        return FEM128(x_pitch, y_pitch, mM_channels, sum_rows_cols, channels)
    elif FEM_type == "FEM256":
        return FEM256(x_pitch, y_pitch, mM_channels, sum_rows_cols, channels)
    else:
        raise ValueError("Unsupported FEM type")
