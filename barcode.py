import cv2
import math
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from scanline_features import ScanlineFeatures
from grade import Grade

plt.rcParams['figure.figsize'] = [5, 5]

class Barcode:
    """
        A class representing all the images and features of interest of a barcode image
        :param bars: the boxes' coordinates which enclose the bars of the barcode
        :param scanlines: the 10 scanline we take to analyse several parameters of the barcode
        :param grade: the grade which describes the quality of the barcode
        :param left_gap: the gap we want at the left of the barcode
        :param right_gap: the gap we want at the right of the barcode
        :param min_width: the width of the thinnest bar
        :param min_height: the height of the shortest bar
        :coord_roi: a tuple (x, y, w, h) where x and y are the top-left coordinate of the roi
        
    """
    def __init__(self):
        self.image_name = None
        self.image = None
        self.roi = None
        self.bin_image = None
        self.bin_adaptive_roi = None
        self.bin_otsu_roi = None
        self.coord_roi = None
        self.rotation_angle = 0
        self.min_width = None
        self.min_height = None
        self.bars = None
        self.l_to_r_widths = []
        self.left_gap = None
        self.right_gap = None
        self.scanlines = None
        self.grade = None

    def is_barcode_horizontal(self):
        """
            Checks if the given barcode is horizontal (with vertical bars with 0 degrees of inclination) or not

        """

        #First we find the edges of the binarized roi with Canny, then we find the lines of the barcode with HoughLines
        edges = cv2.Canny(self.bin_adaptive_roi, 100, 150, apertureSize = 3)
        lines = cv2.HoughLines(edges, 1, np.pi / 2, 70) 
        #We cycle the lines found and store their angles in a list
        angles = []
        for line in lines:
            rho, theta = line[0]
            angles.append(int(math.degrees(theta)))

        #The rotation angle of the barcode becomes the mode value of the angles list
        return abs(stats.mode(angles)[0]) != 90
    
    def main_set_roi(self):
        """
            Finds the coordinates of the Region Of Interest of the given barcode within the cropped image and the binarized roi too

        """

        '''MORPHOLOGY FOR ROI DETECTION'''
        #We construct the kernel which will be used for morphology operations
        hit_miss_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 10))
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        lateral_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))

        hit_miss = cv2.morphologyEx(self.bin_image, cv2.MORPH_HITMISS, hit_miss_kernel)
        dilated = cv2.dilate(hit_miss, dilation_kernel, iterations = 4)
        eroded = cv2.erode(dilated, erosion_kernel, iterations = 1)
        final = cv2.dilate(eroded, dilation_kernel, iterations = 1)
        final = cv2.morphologyEx(final, cv2.MORPH_OPEN, open_kernel)


        final = cv2.dilate(final, lateral_dilation, iterations=2)

        '''FINDING BIGGEST BLOB'''
        #Finds the countours of connected components and picks the one with maximum area,
        #then the minimum area rectangle which encloses it
        cnts = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area_contour = max(cnts[1], key = cv2.contourArea)
        rect = cv2.minAreaRect(max_area_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #We find x and y which are the coordinates of the top-left corner of the roi, then w and h which are the width and height of the roi
        x, y, w, h = cv2.boundingRect(box)
        self.roi = self.image[y : y + h, x : x + w]

        try:
            self.bin_adaptive_roi = cv2.adaptiveThreshold(self.roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 11)
        except:
            return False

        self.coord_roi = (x, y, w, h)
        return


    def set_roi(self):
        """
            Uses the __set_roi__ method and reuses it after having rotated the image 90 degrees clockwise in case 
            the barcode orientation wasn't horizontal

        """
        self.main_set_roi()
        if not self.is_barcode_horizontal():
            print("Rotating the image 90° clockwise...")
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.rotation_angle +=90
            self.bin_image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 11)
            self.main_set_roi()
            
    
    def verticalize_roi(self):
        """
            Modifies the inclination of the barcode image and roi if the mode of bars' angles isn't 0 degrees

        """

        edges = cv2.Canny(self.bin_adaptive_roi, 100, 200, apertureSize=3)
        votes_thresh = 170
        lines = cv2.HoughLines(edges, 1, np.pi / 180, votes_thresh)

        #If the lines found are less than 40 we retry with HoughLines decreasing the votes threshold
        while lines is None or len(lines) < 40:
            votes_thresh -= 60
            lines = cv2.HoughLines(edges, 1, np.pi / 180, votes_thresh)

        angles = []
        for line in lines:
            rho, theta = line[0]
            angles.append(math.degrees(theta))

        frequent_angle = stats.mode(angles)[0][0]
        #print("frequent_angle: ", frequent_angle)

        #We don't want to turn the image upside-down but only to rotate it for the few degrees of inclination it has
        if frequent_angle > 90:
            frequent_angle -= 180

        self.rotation_angle -= frequent_angle    
        #print('The image will be rotated %3.2f degrees' % frequent_angle)

        #We find the coordinates of the center of the image and the center of the roi
        (h_roi, w_roi) = self.roi.shape[:2]
        (h_image, w_image) = self.image.shape[:2]
        center_roi = (w_roi // 2, h_roi // 2)
        center_image = (w_image // 2, h_image // 2)
        #print("Rotation angle: %3.2f" % self.rotation_angle)

        #We find the rotation matrix for the roi and the image
        if frequent_angle != 0:
            rot_mat_roi = cv2.getRotationMatrix2D(center_roi, frequent_angle, 1)
            rot_mat_image = cv2.getRotationMatrix2D(center_image, frequent_angle, 1)
            #The borderValue is the color of the border outside the box, the default value is 0 which is black
            mean_value = np.mean(self.roi[0])
            self.image = cv2.warpAffine(self.image, rot_mat_image, (w_image, h_image), borderValue = mean_value)
            self.roi = cv2.warpAffine(self.roi, rot_mat_roi, (w_roi, h_roi), borderValue = mean_value)
            
    def set_minimum_width_height_bars(self):
        """
            Finds the minimum width and height of a bar in the given barcode within all the bars founded enclosed by a box

        """

        contours = cv2.findContours(self.bin_otsu_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        widths = []
        heights = []
        bars = []

        #We consider only the bars which are higher or equal than the 60% of their barcode's image height
        min_height = 0.6 * self.roi.shape[0]
        colored_roi = cv2.cvtColor(self.roi.copy(), cv2.COLOR_GRAY2RGB)
        # Returns the minimum rectangular area which encloses the point of a contour.
        rects = map(lambda x: cv2.minAreaRect(x), contours)
        sorted_rects = sorted(rects, key=lambda x:x[0][0])
        for rect in sorted_rects:


            # (x,y) = center
            # (width, height) = dimensions
            # angle = rotation of the rectangle clockwise (if positive)
            (real_x, real_y), (real_width, real_height), angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #We find x and y which are the coordinates of the top-left corner of the roi, then w and h which are the width and height of the roi
            x, y, width, height = cv2.boundingRect(box)

            #Depending on the bar's angle sometimes height and width are inverted, so we reinvert them
            if real_width > real_height:
                temp_width = real_width
                real_width = real_height
                real_height = temp_width

            if height > min_height and (abs(angle) < 3 or abs(angle) > 87):
                widths.append(real_width)
                heights.append(real_height)
                bar = cv2.boxPoints(rect)
                bar = np.int0(bar)
                bars.append(bar)

                cv2.drawContours(colored_roi, [bar], 0, (255,0,0), 2)

        self.min_width = min(widths)
        self.min_height = min(heights)
        self.bars = bars
        #We draw the roi with the found bars in red
        plt.title('BARS OF THE BARCODE', pad=10)
        plt.imshow(colored_roi)
        plt.show(block = True)
    
    def resize_roi(self):
        """
            Draw a bounding box enclosing the barcode plus a surrounding background area (quite zone) of size (quite zone):
                - X dimension, above and below the code
                - 10*X before the first bar, 10*X following the last bar

        """

        #X is the min_width of a bar of the given barcode
        X = self.min_width

        #We find the min_x, max_x, min_y, max_y
        min_x = np.min(self.bars[0][:,0])
        max_x = np.max(self.bars[0][:,0])
        min_y = np.min(self.bars[0][:,1])
        max_y = np.max(self.bars[0][:,1])
        for bar in self.bars[1:]:
            min_temp_x = np.min(bar[:,0])
            max_temp_x = np.max(bar[:,0])
            min_temp_y = np.min(bar[:,1])
            max_temp_y = np.max(bar[:,1])
            min_x = min(min_x, min_temp_x)
            max_x = max(max_x, max_temp_x)
            min_y = min(min_y, min_temp_y)
            max_y = max(max_y, max_temp_y)

        #The gaps we have to add to the left, right, bottom
        self.left_gap = 10 * math.ceil(X) - min_x
        self.right_gap = max_x + math.ceil(X) * 10

        above_gap = math.ceil(X)-min_y
        bottom_gap = math.ceil(X)+max_y

        x, y, w, h = self.coord_roi

        #We resize the roi following the given rules, we do that reusing the initial coord_roi despite the rotation we had in the verticalize_roi method
        self.roi = self.image[y - above_gap : y + bottom_gap, x - self.left_gap : x + self.right_gap] 

        
    def cut_roi(self):
        """
            According to the minimum height of a vertical bar, text elements such as numbers are removed so to create a
            smaller box that would not include them.
        """
        x, y = self.coord_roi[0:2]

        max_min_y = np.min(self.bars[0][:,1])
        for bar in self.bars[1:]:
            min_temp_y = np.min(bar[:,1])
            max_min_y = max(max_min_y, min_temp_y)

        y_up = max_min_y
        y_bottom = y_up+round(self.min_height)

        self.roi = self.image[y + y_up : y + y_bottom, x - self.left_gap : x + self.right_gap]
        self.coord_roi = (x - self.left_gap, y + y_up, self.right_gap + self.left_gap, y_bottom - y_up)
        
    def set_scanlines(self):
        """
            Takes 10 evenly spaces scan lines (horizontal lines) within the smaller box including the barcode 
            but not the text elements, 

        """
        even_space = self.roi.shape[0]//11
        lines = list(range(even_space, even_space * 11, even_space))
        line_values = self.roi[even_space : even_space * 11 : even_space, :]

        self.scanlines = []
        for i in range(10):
            self.scanlines.append(ScanlineFeatures())
            self.scanlines[i].row_index = lines[i]
            self.scanlines[i].line_values = line_values[i]
            #print('In line %d the intensities are' % (lines[i], line_values[i]))

        '''PLOT OF SCANLINES'''
        colored_roi_scanlines = cv2.cvtColor(self.roi.copy(), cv2.COLOR_GRAY2RGB)
        lines
        for scan_line in lines:
            cv2.line(colored_roi_scanlines, (0, scan_line), (self.roi.shape[1], scan_line), (255,255,255), thickness=4)

        plt.title("SCANLINES", pad = 10)
        plt.imshow(colored_roi_scanlines)
        plt.show(block = True)

    def set_scanlines_min_reflectance(self):
        for scanline in self.scanlines:
            scanline.min_reflectance = min(scanline.line_values)
            #print('In line %d the minimum reflectance is %d' % (scanline.row_index, scanline.min_reflectance))
            
    def set_scanlines_max_reflectance(self):
        for scanline in self.scanlines:
            scanline.max_reflectance = max(scanline.line_values)
            #print('In line %d the maximum reflectance is %d' % (scanline.row_index, scanline.max_reflectance))
    
    def set_scanlines_symbol_contrast(self):
        for scanline in self.scanlines:
            scanline.symbol_contrast = 100*(scanline.max_reflectance - scanline.min_reflectance)/255.0
            #print('In line %d the symbol contrast is %3.2f%%' % (scanline.row_index, scanline.symbol_contrast))
            
    def set_scanlines_edges(self):   
        for scanline in self.scanlines:
            global_threshold = scanline.min_reflectance + (255*(scanline.symbol_contrast/100)/2)
            normalized_scanline = scanline.line_values - global_threshold

            #The number of edges is the number of times that the threshold is crossed
            asign = np.sign(normalized_scanline)

            #With np.roll we shift asign of one position than subtract from there the original asign array and check where the value is zero.
            #A value different from 0 would mean that the value at the right were different due to the threshold
            #The condition would transform the int array in a boolean array but we use astype(int) so True is 1 and False is 0
            signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
            signchange[0] = 0
            scanline.n_edge = signchange.sum()        
            scanline.edges = signchange

            #print('In line %d the symbol contrast is %3.2f%%' % (scanline.row_index, scanline.n_edges))
            
    def set_left_to_right_widths(self):
        edges = self.scanlines[0].edges
        temp_width = 0
        for value in edges:
            if value == 1:
                self.l_to_r_widths.append(round(temp_width/self.min_width))
                temp_width = 0
            else:
                temp_width += 1

        self.l_to_r_widths.append(round(temp_width/self.min_width))

        #print('Widths of spaces and bars from left to right:', self.l_to_r_widths)

    def set_scalines_minimum_edge_contrast(self):
        # For each scanline
        for scanline in self.scanlines:
            #Maximum edge contrast, better than this doesn't exist
            min_edge_contrast = 100
            #Here i find the index of the edges
            #np.where with only the condition is equivalent to np.nonzero, which returns 2 arrays (rows and columns of the matrix)
            #with the index of the values different from zero, we take only the row index, our matrix's shape is len(scanline.edges) X 1
            edges_index = np.where(scanline.edges == 1)[0]

            # edge pos is the index in edges_index array
            # index is the index of the edge in the scanline.line_values array
            for edge_pos, index in enumerate(edges_index):

                # Here we need to take the left part wrt. the edge and the right part
                # Particular case is when we are at the first iteration and the last
                if edge_pos == 0:
                    left_scan = scanline.line_values[:index+1]
                else:
                    left_scan = scanline.line_values[edges_index[edge_pos-1]: index+1]

                if edge_pos == len(edges_index)-1:
                    right_scan = scanline.line_values[index:]
                else:
                    right_scan = scanline.line_values[index:edges_index[edge_pos+1]]

                # Now we need to check if is a white to black edge or black to white
                # In order to search for max and min in the appropriate direction
                if scanline.line_values[index-1] > scanline.line_values[index+1]: # WHITE TO BLACK EDGE
                    max_white = np.max(left_scan)
                    max_black = np.min(right_scan)
                
                else: # BLACK TO WHITE
                    max_white = np.max(right_scan)
                    max_black = np.min(left_scan)

                # Compute the contrast in %
                edge_contrast = (100*(max_white - max_black))/255            
                # Take the minimum
                min_edge_contrast = min(min_edge_contrast, edge_contrast)
            scanline.min_edge_contrast = min_edge_contrast  
            
    def set_scanline_modulation(self):
        for scanline in self.scanlines:
            scanline.modulation = scanline.min_edge_contrast / scanline.symbol_contrast
            #print('In line %d the symbol contrast is %3.2f%%' % (scanline.row_index, scanline.modulation))
            
    def set_scanline_defects(self):
        for scanline in self.scanlines:
            #Questo potremmo metterlo in una variabile dato che lo abbiamo già calcolato sopra vedremo
            edges_index = np.where(scanline.edges == 1)[0]
            ERNMax = 0
            for edge_pos, index in enumerate(edges_index):

                # Problem in the first and in the second case, the edge determination doesn't hold.
                # We need to treat this two cases as quite zone so as a space
                # In the other case is ok to use the edge determination
                if edge_pos == 0:
                    # Need to fix quiet zones
                    to_scan = scanline.line_values[:index]
                    continue
                elif edge_pos == len(edges_index)-1:
                    # Need to fix quiet zones
                    to_scan = scanline.line_values[index:]
                    continue
                else:
                    to_scan = scanline.line_values[index:edges_index[edge_pos+1]]


                #White to black, so we are analyzing a bar
                if scanline.line_values[edge_pos-1] > scanline.line_values[edge_pos+1]:
                    local_minima = argrelextrema(to_scan, np.less)[0]
                    for i in range(len(local_minima)-1):
                        maximum = np.max(to_scan[local_minima[i]:local_minima[i+1]])
                        ERNMax = max(ERNMax, max(maximum - to_scan[local_minima[i]], maximum- to_scan[local_minima[i+1]]))


                #We are analyzing a space
                else:
                    local_maximum = argrelextrema(to_scan, np.greater)[0]
                    for i in range(len(local_maximum)-1):
                        minimum = np.min(to_scan[local_maximum[i]:local_maximum[i+1]])
                        ERNMax = max(ERNMax, max(to_scan[local_maximum[i]]-minimum, to_scan[local_maximum[i+1]]-minimum))

                #x = range(len(to_scan))
                #plt.plot(x, to_scan, "-o")
                #plt.show(block = True)
            scanline.defects = ERNMax/(scanline.max_reflectance - scanline.min_reflectance)
            #print('In line %d the defect score is %3.2f%%' % (scanline.row_index, ERNMax/(scanline.max_reflectance - scanline.min_reflectance)))
            
    def grade_scanlines(self):
        for scanline in self.scanlines:
            scanline.grades['min_reflectance'] = Grade.get_min_reflectance_grade(scanline.min_reflectance, max(scanline.line_values))
            scanline.grades['min_edge_contrast'] = Grade.get_min_edge_contrast_grade(scanline.min_edge_contrast)
            scanline.grades['symbol_contrast'] = Grade.get_symbol_contrast_grade(scanline.symbol_contrast)
            scanline.grades['modulation'] = Grade.get_modulation_grade(scanline.modulation)
            scanline.grades['defects'] = Grade.get_defects_grade(scanline.defects)
            scanline.grades['decodability'] = Grade.get_decodability_grade(scanline.decodability)
            scanline.grades['decode'] = Grade.get_decode_grade(scanline.decode)
            scanline.grade = min(scanline.grades.values())
            
    def grade_barcode(self):
        avg_grade = np.mean(list(map(lambda x: x.grade, self.scanlines)))
        self.grade = Grade.get_barcode_grade(avg_grade)
        #print('Grade of the barcode: 'self.grade.name)
        
    def write_row_lists(self):
        rows = []
        rows.append(self.image_name)
        rows.append(self.grade.name)
        # X-dimension
        rows.append("{:3.2f}" .format(self.min_width))

        # height
        rows.append("{:3.2f}" .format(self.min_height))

        #BB
        vertexes = [(self.coord_roi[0], self.coord_roi[1]),
                    (self.coord_roi[0]+self.coord_roi[2], self.coord_roi[1]), 
                    (self.coord_roi[0], self.coord_roi[1]+self.coord_roi[3])]
        vertexes_str = repr(vertexes).replace(",",";")
        center = (self.coord_roi[2] // 2, self.coord_roi[3] // 2)
        center_str = repr(center).replace(",",";")
        rows.append("v:{} c:{}".format(vertexes_str, center_str))

        # orientation

        rows.append("%d" %(self.rotation_angle))

        # number of edges in each scanline
        # quality parameter values computed in each of the 10 scanlines
        features = []
        for scanline in self.scanlines:
            features.append(" min ref: {:3.2f}" .format(scanline.min_reflectance))
            features.append(" max ref: {:3.2f}" .format(scanline.max_reflectance))
            features.append(" min edge contrast: {:3.2f}" .format(scanline.min_edge_contrast))
            features.append(" simbol contrast: {:3.2f}" .format(scanline.symbol_contrast))
            features.append(" modulation: {:3.2f}" .format(scanline.modulation))
            features.append(" defects: {:3.2f}" .format(scanline.defects))
            features.append(" decodability: {:3.2f}" .format(scanline.decodability))
            features.append(" decode: " + scanline.decode)
            features_str = ';'.join(features)
            rows.append("{}; num_edges:{}".format(features_str, scanline.n_edge))
            features = []

        # Sequence (from left to right) of the sizes of the found bars and spaces, in units given by X dimension.
        sizes = str(self.l_to_r_widths).replace(",",";")
        rows.append(sizes)
        rows.append("\n")
        return rows
    
    
