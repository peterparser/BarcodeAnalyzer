from enum import IntEnum

class Grade(IntEnum):
    '''
    Class which represents the grades assigned to a scanline or a barcode, it has several static methods to compute the grade
    for each parameter of interest
    
    '''
    A = 4
    B = 3
    C = 2
    D = 1
    F = 0

    @staticmethod
    def get_min_reflectance_grade(score, rmax):
        value = (255 * score) / 100
        if value <= 0.5 * rmax:
            return Grade.A
        else:
            return Grade.F

    @staticmethod
    def get_min_edge_contrast_grade(score):
        if score >= 15:
            return Grade.A
        else:
            return Grade.F


    @staticmethod
    def get_symbol_contrast_grade(score):
        if score >= 70:
            return Grade.A
        elif score >= 55:
            return Grade.B
        elif score >= 40:
            return Grade.C
        elif score >= 20:
            return Grade.D
        else:
            return Grade.F

    @staticmethod
    def get_modulation_grade(score):
        if score >= 0.7:
            return Grade.A
        elif score >= 0.6:
            return Grade.B
        elif score >= 0.5:
            return Grade.C
        elif score >= 0.4:
            return Grade.D
        else:
            return Grade.F

    @staticmethod
    def get_defects_grade(score):
        if score <= 0.15:
            return Grade.A
        elif score <= 0.2:
            return Grade.B
        elif score <= 0.25:
            return Grade.C
        elif score <= 0.3:
            return Grade.D
        else:
            return Grade.F

    @staticmethod
    def get_decodability_grade(score):
        return Grade.A

    @staticmethod
    def get_decode_grade(score):
        return Grade.A

    @staticmethod
    def get_barcode_grade(score):
        if score < 0.5:
            return Grade.F
        elif score < 1.5:
            return Grade.D
        elif score < 2.5:
            return Grade.C
        elif score < 3.5:
            return Grade.B
        else:
            return Grade.A
