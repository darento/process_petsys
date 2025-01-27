# TODO: module to generate LM file from binary file.


from ctypes import (
    Structure,
    c_byte,
    c_char,
    c_double,
    c_float,
    c_int,
    c_short,
    c_ushort,
)


class LMHeader(Structure):
    _pack_ = 4
    _fields_ = [
        ("identifier", c_char * 16),
        ("rawCounts", c_double),
        ("acqTime", c_double),
        ("activity", c_double),
        ("isotope", c_char * 16),
        ("detectorSizeX", c_double),
        ("detectorSizeY", c_double),
        ("startTime", c_double),
        ("measurementTime", c_double),
        ("moduleNumber", c_int),
        ("ringNumber", c_int),
        ("ringDistance", c_double),
        ("detectorDistance", c_double),
        ("isotopeHalfLife", c_double),
        ("weight", c_float),
        ("maxTemp", c_float),
        ("percentLoss", c_float),
        ("detectorPixelSizeX", c_float),
        ("detectorPixelSizeY", c_float),
        ("reserved", c_float * 3),
        ("version", c_byte * 2),
        ("breast", c_char),
        ("unused_1", c_char),
        ("gatePeriod", c_double),
        ("DOILayer", c_short),
        ("method", c_short),
        ("StudyId", c_short),
        ("detectorPixelsX", c_byte),
        ("detectorPixelsY", c_byte),
        ("unused_2", c_char * 4),
    ]


class CoincidenceV3(Structure):
    _fields_ = [
        ("time", c_float),
        ("energy1", c_ushort),
        ("energy2", c_ushort),
        ("amount", c_float),
        ("xPosition1", c_byte),
        ("yPosition1", c_byte),
        ("xPosition2", c_byte),
        ("yPosition2", c_byte),
        ("pair", c_ushort),
    ]


class CoincidenceV4(Structure):
    _fields_ = [
        ("time", c_float),
        ("energy1", c_ushort),
        ("energy2", c_ushort),
        ("amount", c_float),
        ("xPosition1", c_byte),
        ("yPosition1", c_byte),
        ("xPosition2", c_byte),
        ("yPosition2", c_byte),
        ("pair", c_ushort),
        ("dt", c_short),
    ]


class CoincidenceV5(Structure):
    _fields_ = [
        ("time", c_float),
        ("energy1", c_ushort),
        ("energy2", c_ushort),
        ("amount", c_float),
        ("xPosition1", c_byte),
        ("yPosition1", c_byte),
        ("zPosition1", c_byte),
        ("xPosition2", c_byte),
        ("yPosition2", c_byte),
        ("zPosition2", c_byte),
        ("pair", c_ushort),
        ("dt", c_short),
    ]
