class FEMBase:
    def __init__(self, x_pitch: float, y_pitch: float, channels: int):
        self.x_pitch = x_pitch
        self.y_pitch = y_pitch
        self.channels = channels

    def get_coordinates(self, channel_pos: int) -> tuple:
        raise NotImplementedError("Subclass must implement abstract method")


class FEM128(FEMBase):
    def __init__(self, x_pitch: float, y_pitch: float):
        super().__init__(x_pitch, y_pitch, 128)

    def get_coordinates(self, channel_pos: int) -> tuple:
        row = channel_pos // 8
        col = channel_pos % 8
        loc_x = (col + 0.5) * self.x_pitch
        loc_y = (row + 0.5) * self.y_pitch
        return (loc_x, loc_y)


class FEM256(FEMBase):
    def __init__(self, x_pitch: float, y_pitch: float):
        super().__init__(x_pitch, y_pitch, 256)

    def get_coordinates(self, channel_pos: int) -> tuple:
        # Implement FEM256 logic
        pass


def get_FEM_instance(FEM_type: str, x_pitch: float, y_pitch: float):
    if FEM_type == "FEM128":
        return FEM128(x_pitch, y_pitch)
    elif FEM_type == "FEM256":
        return FEM256(x_pitch, y_pitch)
    else:
        raise ValueError("Unsupported FEM type")
