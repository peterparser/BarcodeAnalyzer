class ScanlineFeatures:
    """
        A class representing the parameters of interest of a barcode's scanline
        :param row_index: the index of the row of the barcode where we took the scanline
        :param line_values: the list of pixels' intensities of the scanline
        :param edges: the coordinates of the edges of the scanline
        
    """
    def __init__(self):
        self.row_index = None
        self.line_values = None
        self.edges = None
        self.n_edge = None
        self.min_reflectance = None
        self.max_reflectance = None
        self.min_edge_contrast = None
        self.symbol_contrast = None
        self.modulation = None
        self.defects = None
        self.decode = "PASS"
        self.decodability = 0.62
        self.grades = {}
        self.grade = None
