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
    """

    def __init__(
        self,
        x_pitch: float,
        y_pitch: float,
        sum_rows_cols: bool,
        channels: int,
        mM_channels: int,
        num_ASICS: int,
    ):
        self.x_pitch = x_pitch
        self.y_pitch = y_pitch
        self.sum_rows_cols = sum_rows_cols
        self.channels = channels
        self.mM_channels = mM_channels
        self.num_ASICS = num_ASICS

    def get_coordinates(self, channel_pos: int) -> tuple:
        """
        Returns the coordinates of a channel.

        Parameters:
            - channel_pos (int): The position of the channel.

        Returns:
        tuple: The coordinates of the channel.
        """
        raise NotImplementedError("Subclass must implement abstract method")


class FEM128(FEMBase):
    """
    This class represents the FEM128 instance.

    Parameters:
        - x_pitch (float): The pitch of the x-axis.
        - y_pitch (float): The pitch of the y-axis.
        - mM_channels (int): The number of mM channels.
        - sum_rows_cols (bool): A boolean indicating whether to sum the rows and columns.
    """

    def __init__(
        self, x_pitch: float, y_pitch: float, mM_channels: int, sum_rows_cols: bool
    ):
        super().__init__(x_pitch, y_pitch, sum_rows_cols, 128, mM_channels, 2)

    def get_coordinates(self, channel_pos: int) -> tuple:
        """
        Returns the coordinates of a channel.

        Parameters:
            - channel_pos (int): The position of the channel.

        Returns:
        tuple: The coordinates of the channel.
        """
        if not self.sum_rows_cols:
            row = channel_pos // 8
            col = channel_pos % 8
            loc_x = (col + 0.5) * self.x_pitch
            loc_y = (row + 0.5) * self.y_pitch
        else:
            loc_x = self.x_pitch / 2 + self.x_pitch * (channel_pos % 16)
            loc_y = self.y_pitch / 2 + self.y_pitch * (7 - channel_pos % 16)
        return (loc_x, loc_y)


class FEM256(FEMBase):
    """
    This class represents the FEM256 instance.

    Parameters:
        - x_pitch (float): The pitch of the x-axis.
        - y_pitch (float): The pitch of the y-axis.
        - mM_channels (int): The number of mM channels.
        - sum_rows_cols (bool): A boolean indicating whether to sum the rows and columns.
    """

    def __init__(
        self, x_pitch: float, y_pitch: float, mM_channels: int, sum_rows_cols: bool
    ):
        super().__init__(x_pitch, y_pitch, sum_rows_cols, 256, mM_channels, 4)

    def get_coordinates(self, channel_pos: int) -> tuple:
        """
        Returns the coordinates of a channel.

        Parameters:
            - channel_pos (int): The position of the channel.

        Returns:
        tuple: The coordinates of the channel.
        """
        if not self.sum_rows_cols:
            row = channel_pos // 16
            col = channel_pos % 16
            loc_x = round((col + 0.5) * self.x_pitch, 2)
            loc_y = round((row + 0.5) * self.y_pitch, 2)
        else:
            loc_x = round(self.x_pitch / 2 + self.x_pitch * (channel_pos % 32), 2)
            loc_y = round(self.y_pitch / 2 + self.y_pitch * (31 - channel_pos % 32), 2)
        return (loc_x, loc_y)


def get_FEM_instance(
    FEM_type: str, x_pitch: float, y_pitch: float, mM_channels: int, sum_rows_cols: bool
) -> FEMBase:
    """
    Returns the FEM instance.

    Parameters:
        - FEM_type (str): The type of FEM.
        - x_pitch (float): The pitch of the x-axis.
        - y_pitch (float): The pitch of the y-axis.
        - mM_channels (int): The number of mM channels.
        - sum_rows_cols (bool): A boolean indicating whether to sum the rows and columns.

    Returns:
    FEMBase: The FEM instance.
    """
    if FEM_type == "FEM128":
        return FEM128(x_pitch, y_pitch, mM_channels, sum_rows_cols)
    elif FEM_type == "FEM256":
        return FEM256(x_pitch, y_pitch, mM_channels, sum_rows_cols)
    else:
        raise ValueError("Unsupported FEM type")
