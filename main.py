import cv2
import glob
import matplotlib.pyplot as plt
from colorama import Fore, Style
from barcode import Barcode

title = r"""██████╗  █████╗ ██████╗  ██████╗ ██████╗ ██████╗ ███████╗     █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗███████╗██████╗ 
██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝╚══███╔╝██╔════╝██╔══██╗
██████╔╝███████║██████╔╝██║     ██║   ██║██║  ██║█████╗      ███████║██╔██╗ ██║███████║██║   ╚████╔╝   ███╔╝ █████╗  ██████╔╝
██╔══██╗██╔══██║██╔══██╗██║     ██║   ██║██║  ██║██╔══╝      ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝   ███╔╝  ██╔══╝  ██╔══██╗
██████╔╝██║  ██║██║  ██║╚██████╗╚██████╔╝██████╔╝███████╗    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████╗███████╗██║  ██║
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝
                                                                                                                             """

def print_title():
    print(Fore.YELLOW + title)
    print(Style.RESET_ALL)
    
def main(barcode_directory):

    column_names = ["IMAGE NAME", "BARCODE GRADE", "X-DIMENSION", "HEIGHT", "BOUNDING BOX", "ROTATION ANGLE",
                        "SCANLINE-1","SCANLINE-2", "SCANLINE-3", "SCANLINE-4", "SCANLINE-5", 
                        "SCANLINE-6", "SCANLINE-7", "SCANLINE-8", "SCANLINE-9", "SCANLINE-10", "SIZE BARS-SPACES\n"]
    header = ','.join(column_names)
    f = open('barcode_features.csv', 'w')
    f.write(header)

    #Cycle over the given images in the ./resources folder
    for file in glob.glob(barcode_directory + "/I25-MASTER GRADE IMGB.BMP"):

        #print()
        #print('|' * 80)
        #print()

        #Print the image_name and the image itself  
        barcode = Barcode()
        barcode.image_name = file
        #print('image_name: ', barcode.image_name)

        plt.title('ORIGINAL IMAGE', pad=10)
        barcode.image = cv2.imread(barcode.image_name, cv2.IMREAD_GRAYSCALE)
        plt.imshow(barcode.image, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show(block = True)

        #Show the roi and print its coordinates
        barcode.bin_image = cv2.adaptiveThreshold(barcode.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 11)
        barcode.set_roi()
        plt.title('ORIGINAL ROI', pad=10)
        plt.imshow(barcode.roi, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show(block = True)
        #print('coord_roi: ', barcode.coord_roi)

        #Show the roi after the verticalization
        plt.title('VERTICALIZED ROI', pad=10)
        barcode.verticalize_roi()
        plt.imshow(barcode.roi, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show(block = True)

        #Show the highlighted bars in red and print the minimum width of a bar and the minimum_height of a bar
        barcode.bin_otsu_roi = cv2.threshold(barcode.roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        barcode.set_minimum_width_height_bars()
        #print('min_width: %3.2f' % barcode.min_width)
        #print('min_height: ', barcode.min_height)

        #Show the roi resized with the following rule: bounding box enclosing the barcode plus a surrounding background area
        #(quite zone) of size (quite zone): X dimension, above and below the code, 10*X before the first bar, 10* X following the last bar
        plt.title('RESIZED ROI', pad=10)
        barcode.resize_roi()
        #Show the roi resized with the following rule
        plt.imshow(barcode.roi, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show(block = True)

        #Numbers and text elements are remmoved so to create a smaller box that would not include them.
        plt.title('CUT ROI', pad=10)
        barcode.cut_roi()
        plt.imshow(barcode.roi, cmap = 'gray', vmin = 0, vmax = 255)
        plt.show(block = True)

        #Set the scanlines we have to analyse and plot them    
        barcode.set_scanlines()

        #Methods which sets the scanlines' parameters we have to find
        barcode.set_scanlines_min_reflectance()

        barcode.set_scanlines_max_reflectance()

        barcode.set_scanlines_symbol_contrast()

        barcode.set_scanlines_edges()

        barcode.set_scalines_minimum_edge_contrast()

        barcode.set_scanline_defects()

        barcode.set_scanline_modulation()

        barcode.set_left_to_right_widths()

        #Set the grade of each scanline
        barcode.grade_scanlines()

        #Set the grade of the barcode
        barcode.grade_barcode()

        row = ','.join(barcode.write_row_lists())

        f.write(row)
    f.close()
    
if __name__ == "__main__":
    print_title()
    barcode_directory = input('Insert the name of the directory you want to analyse:\t')
    main(barcode_directory)
    print("\nEnd of analysis")


