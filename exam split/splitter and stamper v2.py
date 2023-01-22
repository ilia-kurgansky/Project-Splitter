#!/usr/bin/env python3

"""
split_qr_exam
=============

This script splits a pdf composed of scanned exams. The input is a pdf file containing multiple
scanned exames. On each exam cover there should be a QR code identifying the student writing this
exam.
For each part of the exam a folder will be created and for each student a pdf will be created in
that folder to allow for parallel evaluation. To ensure pseudonymity the files are named with the
sha256 hash of the student id.

For this to work, you have to specify all the parts of the exam with the corresponding amount of
pages.
"""
from __future__ import print_function
import logging
import qrcode
import pandas as pd

import shutil

import numpy as np
from PIL import Image

import fitz
import cv2

import os
from filecmp import dircmp
import subprocess

c = 0
blank = "Blank"
raw = "Raw"
stamped = "Stamped"

completed = "Completed"
soft_files = "1 Digital submissions"
unsplit_scans = "2 Unsplit scans"
split_scans = "Sorted files"

marked = "Marked"
portfolios = "Portfolios"

# invalid = "_Invalid files"

AA_stamp = "EO"
absolute_cloud = "C:\\Users\\KIO\\Rugby School\\Project Splitter - Documents"
absolute_workdir = "D:\\Programming\\QR code splitting\\exam split\\Processing"

data = []

unmarked_log = []
marked_log = [[], []]


class NoQRCodeFoundException(Exception):
    """Exception thrown if no QR code is found on a page.

    Should only be thrown, if multiple attempts at multiple thresholds were attempted."""


class NoCoverFoundException(Exception):
    """Exception thrown if no page with a QR code is found.

    Should only be thrown, if all pages of an exam are tested."""


class QRScanner:
    """Class that handles the scanning of QR codes

    Intended use is creating an instance of the class once, and using it to process all neccessary
    images.

    Attributes:
        threshold: A value between 0 and 255, indicating what amount of grey will be interpreted
                   as black or white. Higher values lead to more black, lower values lead to more
                   white. No threshold will be applied, if the value is None.
        shrink: Shinks the image to this width in the detection phase.
        crop: A tuple of fraction coordinates, representing the area to crop the image to before
              scanning for QR codes. The first element represents the upper left corner of the area
              the second element represents the lower right corner.
              E.g.: ((0,0),(0.5,0.5)) would take the top left quartal of the image,
                    ((0,0.5),(0,1)) would take the left half of the image,
                    ((0.25,0.25), (0.75, 0.75)) would take the center of the image.
              If any value is greater than 1 or less than 0, it will be set to 1 or 0 respectively.

    Example use:
        qr_scanner = QRScanner(127, 600)
        try:
            qr_string = qr_scanner.get_qr_string(image)
            print(qr_string)
        except NoQRCodeFoundException:
            print("Could not find any QR codes :(")
    """

    def __init__(self, threshold, shrink, crop):
        """Initializes all attributes of the object"""

        self.threshold = threshold
        if self.threshold is not None and self.threshold > 255:
            logging.warning("Threshold above 255, setting to 255")
            self.threshold = 255
        elif self.threshold is not None and self.threshold < 0:
            logging.warning("Threshold below 0, setting to 0")
            self.threshold = 0
        self.shrink = shrink
        self._qr_instance = cv2.QRCodeDetector()
        if crop is not None:
            if crop[0][0] > 1:
                crop[0][0] = 1
            if crop[0][0] < 0:
                crop[0][0] = 0
            if crop[0][1] > 1:
                crop[0][1] = 1
            if crop[0][1] < 0:
                crop[0][1] = 0
            if crop[1][0] > 1:
                crop[1][0] = 1
            if crop[1][0] < 0:
                crop[1][0] = 0
            if crop[1][1] > 1:
                crop[1][1] = 1
            if crop[1][1] < 0:
                crop[1][1] = 0
        self.crop = crop

    def get_qr_string(self, image):
        """Decodes a QR code on page `pagenr` and returns the string.

        Args:
            image: The image to extract the string from.

        The image is first cropped, shrunk down and cleaned up, to improve performance and accuracy.
        If no code can be found after using these two operations, a second attempt is made for each
        threshold between 0 and 255, but not shrinking down the image.

        Returns:
            The string encoded in the QR code in the image

        Raises:
            NoQRCodeFoundException: If no QR code can be found
        """

        cropped_image = self.crop_image(image)
        qr_coordinates = self.detect_qr_code(cropped_image)
        if qr_coordinates is None:
            # # qr_coordinates = self.detect_qr_code_slow(cropped_image)  # modified
            # if qr_coordinates is None:
            raise NoQRCodeFoundException()
        return self.decode_qr_code(cropped_image, qr_coordinates)

    def crop_image(self, image):
        """Croppes the image

        This operation does not create a new image. It returns a slice view of the original image.
        Relevant attributes for this method are: self.crop

        Args:
            image: The image to slice

        Returns:
            A cropped portion of the image. Or the original image, if self.crop is None.
        """

        if self.crop is None:
            return image
        width, height, _ = image.shape
        x_start, new_width = int(self.crop[0][0] * width), int(self.crop[1][0] * width)
        y_start, new_height = int(self.crop[0][1] * height), int(self.crop[1][1] * height)
        # img = Image.fromarray(image[x_start:new_width, y_start:new_height],'RGB')
        # img.show()
        # input()
        return image[x_start:new_width, y_start:new_height]

    def shrink_image(self, image):
        """Resizes an image to a width of `self.shrink`

        This function creates a new, resized image with width `self.shrink`. This operation
        preserves the aspect ratio of the input image.

        Args:
            image: The image to shrink

        Returns:
            A tuple consisting of
              * The input image shrunk to the width of `self.shrink`
              * The floating point scaling factor used to shrink the image
        """

        factor = image.shape[0] / self.shrink
        returns = cv2.resize(image, None, fx=(1 / factor), fy=(1 / factor)), factor
        return returns

    def clean_up(self, image, overwrite_threshold=None):
        """Cleans up a greyscale image by removing the grey parts

        The higher the threshold, the more grey is interpreted as black. The lower the threshold,
        the more grey is interpreted as white.

        Args:
            image: The image to clean up
            overwrite_threshold: If set, use this value instead of `self.threshold`.

        Returns:
            A new monochrome image.
        """

        threshold = self.threshold if overwrite_threshold is None else overwrite_threshold

        return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

    def detect_qr_code(self, image, no_shrink=False, overwrite_threshold=None):
        """Finds the location of a QR code in an image using an optimized method.

        First a shrinking operation is performed. Since the mere detection of a QR code does not
        need all the details of the image, this operation does not impact the accuracy of the
        detection severly, but helps greatly to improve the performance.
        Then a clean up operation is performed. This converts the greyscale input image to a black
        and white image, thereby cleaning up the QR code and improve the accuracy of the scan.

        Args:
            image: The image to process
            no_shrink: If True, skips the shrinking operation
            overwrite_threshold: If set, uses this value instead of `self.threshold` in the
                                 clean_up phase

        Returns:
            A numpy float array of coordinates if a QR code is found, otherwise it returns None.
            If more than one QR code is found... that would be undefined behaviour.
        """

        mini_image, factor = self.shrink_image(image) \
            if self.shrink is not None and not no_shrink else (image, 1)

        mini_cleaned_up = self.clean_up(
            mini_image, overwrite_threshold=overwrite_threshold
        ) if self.threshold is not None else image
        # img = Image.fromarray(mini_cleaned_up, 'RGB')
        # img.show()
        try:
            found, coordinates = self._qr_instance.detect(mini_cleaned_up)
        except:

            img = Image.fromarray(mini_cleaned_up, 'RGB')
            img.show()
            return None
            # input()
        if not found:
            # img.show()
            return None
        # else:
        #     global c
        #     c += 1
        #     img = Image.fromarray(mini_cleaned_up, 'RGB')
        #     img.save(f"{c}.png")

        return coordinates * factor

    def detect_qr_code_slow(self, image):
        """Finds a QR code in in an image

        This should only be called, if a call to `detect_qr_code` was unsuccessful.

        Instead of shrinking and cleaning the image up, this method only tries to clean the image
        up, thereby maximizing the accuracy. Also, instead of only using a fix threshold, all
        possible thresholds, starting with `self.threshold` will be tried, until one run yields a
        result.

        Args:
            image: The image to detect a QR code in


        Returns:
            A numpy float array of coordinates if a QR code is found, otherwise it returns None.
        """

        thresh = self.threshold
        coordinates = None
        while coordinates is None:
            # print(f"({thresh})", end="")
            coordinates = self.detect_qr_code(image, no_shrink=True, overwrite_threshold=thresh)
            thresh += 1
            if thresh >= 256:
                # return None
                thresh = 0
            if thresh == self.threshold - 1:
                # print()
                return None
        img = Image.fromarray(image, 'RGB')
        #     img.save(f"{c}.png")
        img.show()
        return coordinates

    def decode_qr_code(self, image, coordinates):
        """Given an image and the coordinates of a QR code, returns the string of the QR code.

        Args:
            image: An image with a QR code
            coordinates: Coordinates found with `self.detect_qr_code()` or
                         `self.detect_qr_code_slow()`

        Returns:
            The string encoded in the QR code, or an empty string, if no string could be decoded.
        """
        try:
            return self._qr_instance.decode(image, coordinates)[0]
        except cv2.error:
            print("WTF", end=" ")
            img = Image.fromarray(image, 'RGB')
            img.show()
            return ""


class Exam_Splitter:
    """Processeses one single pdf file containing all the scanned exams.

    For each student, a folder is created containing a pdf file for each
    part defined. A student is identified by a QR code in the top left
    part of the cover sheet.

    If a student cannot be identified, the exam will be placed in a seperate folder
    and a warning will be issued. This exam needs to be processed manually.

    Some errors, that can occur in the scanning process will be fixed however. If
    an exam has fewer pages than defined, because the scanner skipped a page for some reason,
    the rest of the exams wont be aligned to the total number of pages per exam. In such a case
    an offset is calculated going forward and the problematic exam will be put in a seperate folder
    for manual processing.

    Attributes:
        pdf_file: A pymupdf-object, containing all the exams in sequence
        destpath: The output folder. If it does not exist, it will be created.
        parts: A list of tupels describing the parts of the exam together
               with the first and last page of the part. E.g.
               [("Cover", 0, 2), ("Task 1", 2, 4), ("Bonuspage_1", 4, 6), ("Bonuspage_2", 6, 8)]
        qr_scanner: An instance of `QRScanner`
    """

    def __init__(self, filepath, qr_scanner):
        """Initializes the object.

        Opens the file `filename` as a pymupdf-object. If the number of pages in the input document
        is not divisible by the number of pages per exam (given by the `parts` list), a warning is
        issued. Manual intervention may be needed.

        Args:
            filename: The name of the file to open
            destpath: see corresponding Attribute
            parts: see corresponding Attribute
            qr_scanner: see corresponding Attribute

        Raises:
            RuntimeError, if filename does not exist or is not readable
        """
        self.qr_scanner = qr_scanner
        self.filepath = filepath
        if "zzz" not in filepath:
            self.pdf_file = fitz.open(filepath)
        else:
            print(f"\nAlready processed {filepath}.\n")
            self.pdf_file = None

        self.dest_folder = ""
        self.QRfound = False

        self._chunks = []

    def process_exams(self, filetype):
        # filetype is marked or unmarked

        if self.QRfound:  # if file actually contains a QR code - then it is ready for splitting and writing into folders
            return self.write_files(filetype)  # write_files returns number of files written
        else:
            # if file does not contain a QR code - it could be a digital work submission, return False to deal with...
            return False

    def identify_papers(self, thorough_mode=False):
        # if thorough_mode:
        #     print(f"File is {self.filepath}")
        #     thresh = int(input(f"Current threshold is {self.qr_scanner.threshold}, enter new threshold:"))
        #     self.qr_scanner.threshold = thresh
        if self.pdf_file is None:
            self.QRfound = False
            return False

        last_cover_index = -1
        last_id = -1

        print(f"\nOpening {self.filepath}")
        print("Found QR on pages:", end=" ")

        threshs = range(20, 180, 10)  # try all sorts of thresholds.... slow, but faster than manual trial and error
        shrinks = [300, 250, 200]  # took out 200
        for page in range(self.pdf_file.pageCount):
            something_found_flag = False
            for shrink in shrinks:  # turns out some QR codes work better with different shrink values...
                if something_found_flag:
                    break
                for thresh in threshs:
                    self.qr_scanner.threshold = thresh
                    self.qr_scanner.shrink = shrink
                    try:
                        image = self.get_image_for_page(page)
                        student_id = self.qr_scanner.get_qr_string(image).split("_")
                        # print(student_id)
                        # 7690_Zheng_Peter_Art_1_DA/AR_DB is split
                        if len(student_id) > 1:
                            print(f"{page}({thresh},{shrink})", end=" ")
                            if last_id == -1:
                                last_id = student_id
                                last_cover_index = page
                            else:
                                self._chunks.append((last_id, last_cover_index, page - 1))
                                last_id = student_id
                                last_cover_index = page
                            something_found_flag = True
                            break
                    except NoQRCodeFoundException:
                        # print(page, "contains no QR code")
                        pass
        if last_cover_index > -1:  # found at least one QR code
            self._chunks.append((last_id, last_cover_index, self.pdf_file.pageCount - 1))
            print(f"\nFound {len(self._chunks)} papers.")
            if len(self._chunks[0][0]) < 2:  # false recognition of a pattern, so not actually a QR code.
                print("Looks like QR code is invalid:", self._chunks[0])
                self.QRfound = False
                self.pdf_file.close()
                return False
            self.QRfound = True
            return True
        else:
            print("NONE\n")
            self.QRfound = False
            self.pdf_file.close()
            return False

        # print(self._chunks)

    def write_files(self, file_type):  # I think this is done now...
        # file_type is either "marked" or "unmarked"
        dest = ""
        outcome = [0, 0]  # new, existing
        for chunk in self._chunks:
            # each chunk[0] is a list containing 7690_Zheng_Peter_Art_1_DA/AR_DB split by _

            # the below is very flaky, consistency with "make_directories" or it will not work...
            candidate_details = f"{chunk[0][1]} {chunk[0][2]} ({chunk[0][0]})"  # Zheng Peter (7690)
            yeargroup = chunk[0][-1]  # DB
            subject = chunk[0][3]
            ass_number = chunk[0][4]
            setcode = chunk[0][5].replace("/", "-")

            start = chunk[1]
            end = chunk[2]

            if file_type == "marked":
                self.dest_folder = os.path.join("Processing", marked, portfolios)
                dest = os.path.join(self.dest_folder, subject, yeargroup, setcode, candidate_details)
                # dest = os.path.join(self.dest_folder, subject, yeargroup, candidate_details)
            elif file_type == "unmarked":
                self.dest_folder = os.path.join("Processing", completed, subject)
                dest = os.path.join(self.dest_folder, yeargroup, split_scans, setcode)
            os.makedirs(dest, exist_ok=True)

            file_path = os.path.join(dest, f"{chunk[0][0]}_{ass_number}_{chunk[0][1]} {chunk[0][2]}.pdf")
            file_pdf = fitz.open()
            file_pdf.insertPDF(self.pdf_file, from_page=start, to_page=end)
            if os.path.exists(file_path):
                print(f"File already exists, {file_path} {file_pdf.pageCount}")
                outcome[1] += 1
            else:
                print(f"Wrote {file_path} with {file_pdf.pageCount} pages.")
                outcome[0] += 1

                file_pdf.save(file_path)

            file_pdf.close()
        self.pdf_file.close()
        print()

        # want to avoid renaming files from now on as scans are coming in
        # results in too many updates and reuploads... Need to rely on logs to keep track of what was processed.
        # filename = self.filepath.split("\\")[-1]
        # new_filename = "zzz" + filename
        # new_filepath = self.filepath.replace(filename, new_filename)
        # os.rename(self.filepath, new_filepath)
        return outcome

    def get_image_for_page(self, pagenr):
        """Renders the page `pagenr` and returns a numpy array containing the imagedata"""

        pix = self.pdf_file[pagenr].getPixmap()
        image = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        return np.ascontiguousarray(image[..., [2, 1, 0]])  # rgb to bgr


def parse_filename(filename):
    #  HODs to tell students to name their digital submissions as:
    #  CandidateNo_AssignmentNo.*, e.g. 7590_1.pdf or 2536_3.docx
    #  Must also place the digital submissions into subject-specific folder - this is how assignment subject is known
    filename = filename.strip().split(".")  # if all is well: ["7590_1","pdf"]
    filename[0] = filename[0].replace("-", "_")  # many mistake "-" for "_", so might as well address...
    info = filename[0].strip().split("_")  # if all is well: ["7590","1"]
    if len(info) >= 2:  # if the are two pieces (or more if secondary sorting)...
        info[0] = info[0].replace(" ", "")
        info[1] = info[1].replace(" ", "")
        if info[0].isdigit() and info[1].isdigit():  # ...and both are numbers...
            # doesn't mean the filename is right, but at least it matches expectation
            # print(filename)
            return [info[0]] + [info[1]] + [filename[1]]  # need to also grab the file extension for correct resaving
    # failing that, definitely wrong name so return None
    return None


def look_up_student(candidate_num, subject):
    for student in data:  # linear search *because you're worth it*
        if student[0] == candidate_num and student[4] == subject:
            return student
    return None


def move_by_filename(full_file_path, subject, filename, filetype):  # returns True if file copied, False if file exists
    # if "zzz" in filename:  # no longer need this as using a log to keep track of previously updated files.
    #     return False

    # filetype is either "marked" or "unmarked"
    portfolio_path = ""

    #  subject is provided by the path where the filename is found
    file_info = parse_filename(filename)  # filename must be like 7590_1.pdf, candidateNo and assNo
    if file_info is not None:
        # if filename parsed well...
        student = look_up_student(file_info[0], subject)  # find student details by candidate number
        if student is not None:
            # if a student is found...
            sname = f"{student[1]} {student[3]} ({student[0]})"
            setcode = student[5].replace("/", "-")

            if filetype == "marked":
                portfolio_path = os.path.join("Processing", marked, portfolios, subject, student[-1], setcode, sname)
            elif filetype == "unmarked":
                portfolio_path = os.path.join("Processing", completed, subject, student[-1], split_scans, setcode)

            fname = f"{student[0]}_{file_info[1]}_{student[1]} {student[3]}.{file_info[2]}"  # e.g. 7352_1_Jimmy Bobbins.docx -
            # in case need to sort again later saving file as if it is a digital submission all over again.

            os.makedirs(portfolio_path, exist_ok=True)

            full_out_path = os.path.join(portfolio_path, fname)
            if os.path.isfile(full_out_path):  # this should only run if an update has been made to the original.
                # still will not allow overwriting to avoid mistakes
                print(f"{subject} {student[-1]}: {filename} already exists, worth checking.")
                return False
            else:
                print(f"Copied {filename} to {full_out_path}")
                shutil.copy(full_file_path, full_out_path)
                return True
        else:
            # if a student is not found, but filename parsing was ok...
            print(f"Student with candidate number {file_info[0]} for {subject} not found.")
            raise ValueError(f"Student with candidate number {file_info[0]} for {subject} not found.")

    else:
        # if filename completely wrong...
        print(f"Filename {filename} is completely dodgy...")
        return "ERROR"
        # raise ValueError(f"Filename {filename} is completely dodgy...")


def parse_cropping(crop_string, no_crop):
    """Parses a string formatted like "x1,y1:x2,y2" into the tuple ((x1,y1), (x2,y2))
    where x1, x2, y1, y2 are floating point numbers."""

    if no_crop:
        return None
    coord_x, coord_y = crop_string.split(":")
    start_x, end_x = coord_x.split(",")
    start_y, end_y = coord_y.split(",")
    return ((float(start_x), float(end_x)), (float(start_y), float(end_y)))


def pdf_stamp_QR(newdoc, paper_index, qr_info, qr_img):  # tidy up the details that appear on the cover
    cover = newdoc[0]
    cover_img = cover.getPixmap()
    dimensions = [cover_img.width, cover_img.height]

    qr_topleft = (dimensions[0] * 0.66, 0)
    qr_bottomright = (dimensions[0], dimensions[0] - qr_topleft[0])
    qr_rect = fitz.Rect(qr_topleft, qr_bottomright)  # places the QR code in the top-right third of the page
    pict = fitz.Pixmap(qr_img)
    cover.insertImage(qr_rect, pixmap=pict, overlay=True)

    set_code_rect = fitz.Rect(dimensions[0] // 10, dimensions[1] // 15, 2 * dimensions[0] // 3, dimensions[1] // 6)
    if qr_info[6] != "nan":
        set_code = f"AA - {qr_info[5]}"
    else:
        set_code = f"{qr_info[5]}"

    if qr_info[7] != "nan":
        set_code_rect2 = fitz.Rect(0, 0, 1.7 * dimensions[0] // 10, 1.7 * dimensions[1] // 15)
        shape = cover.new_shape()
        shape.drawRect(set_code_rect2)
        shape.finish(color=0.5, fill=0.5, width=2)
        shape.commit()
        shape = cover.new_shape()
        shape.drawRect(set_code_rect)
        shape.finish(color=1, fill=1, width=5)
        shape.commit()

    sc = cover.insertTextbox(set_code_rect, set_code, fontsize=40, align=fitz.TEXT_ALIGN_LEFT, color=0, fill=0,
                             fill_opacity=1, stroke_opacity=1)

    subj_rect = fitz.Rect(dimensions[0] // 10, dimensions[1] // 8, 2 * dimensions[0] // 3, dimensions[1] // 3)
    subj = f"{qr_info[4]} - {paper_index}"
    if qr_info[7] != "nan":
        subj += " - Exams Office"
    su = cover.insertTextbox(subj_rect, subj, fontsize=20, align=fitz.TEXT_ALIGN_LEFT)

    warn_rect = fitz.Rect(dimensions[0] // 10, dimensions[1] // 3, 9 * dimensions[0] // 10, 3 * dimensions[1] // 7)
    warn = "*** DO NOT WRITE ON THIS PAGE ***"
    wa = cover.insertTextbox(warn_rect, warn, fontsize=20, align=fitz.TEXT_ALIGN_CENTER)

    details_rect = fitz.Rect(dimensions[0] // 10, 3 * dimensions[1] // 7, 9 * dimensions[0] // 10,
                             5 * dimensions[1] // 7)

    if qr_info[6] != "nan":
        details = f"\n    Name:   {qr_info[1]} {qr_info[3]}  ({qr_info[0]}) \n    AA:   {qr_info[6]}"
    else:
        details = f"\n    Name:   {qr_info[1]} {qr_info[3]}  ({qr_info[0]})"
    details += f"\n\n    If submitting digitally, save your file as '{qr_info[0]}_{paper_index}'"
    cover.drawRect(details_rect, color=(0, 0, 0))
    de = cover.insertTextbox(details_rect, details, fontsize=20)

    if min(de, wa, su, sc) < 0:
        print(qr_info, paper_index, "TOO BIG")

    return newdoc


def pdf_stamp_page_info(newdoc, paper_index, student):
    for i, page in enumerate(newdoc):
        if i > 0:
            page_img = page.getPixmap()
            dimensions = [page_img.width, page_img.height]

            details_rect = fitz.Rect(dimensions[0] // 10, 14 * dimensions[1] // 15, 9 * dimensions[0] // 10,
                                     dimensions[1])
            details = f"{student[0]}-{student[4]} {paper_index}-page {i}"
            de = page.insertTextbox(details_rect, details, fontsize=8, align=fitz.TEXT_ALIGN_RIGHT)
    # newdoc.save("stamped.pdf")
    return newdoc


def pdf_save_stamped_single(newdoc, paper_index, student):
    subject = student[4]
    examcode = f"{student[0]}"
    yeargroup = student[-1]
    setcode = student[5]
    setcode = setcode.replace("/", "-")
    dest = os.path.join("Processing", blank, stamped, subject, yeargroup, "Assignment " + str(paper_index), setcode)
    os.makedirs(dest, exist_ok=True)
    if student[7] != "nan":
        filename = f"_{AA_stamp} {examcode} {subject} {paper_index}.pdf"
    else:
        filename = f"{examcode} {subject} {paper_index}.pdf"

    file_path = os.path.join(dest, filename)
    newdoc.save(file_path)
    print("Saved", filename)


def pdf_prepare_for_stamping(dir_path, doc, filename):
    newdoc = fitz.open()
    # newdoc.insertPDF(doc)
    if "blank" in filename.lower():
        newdoc.insert_page(0)
        return newdoc
    newdoc.insert_pdf(doc)

    newdoc.insert_page(0)
    newdoc.insert_page(0)  # insert two blank pages. One for cover, one for overleaf blank for two-sided printing.

    return newdoc


def compress_n_get_doc(dir_path, filename, depth=0):
    # if PDF is over approx 1.6 MB in size - compress it:
    # - first rename original to zzzFileName.pdf
    # - then compress and resave as FileName.pdf
    full_path = os.path.join(dir_path, filename)
    fsize = os.path.getsize(full_path)
    exceptions = ["Business_2_big.pdf"]
    if fsize > 1600000 and filename not in exceptions:
        new_name = "zzz" + "z" * depth + filename
        new_full_path = os.path.join(dir_path, new_name)
        os.rename(full_path, new_full_path)
        pdf_compress_raw(new_full_path, full_path)
        return compress_n_get_doc(dir_path, filename, depth + 1)
    print(f'\n......Fetched {filename} at {fsize // 1024} KB from {dir_path}')
    doc = fitz.open(full_path)
    return doc


def pdf_compress_raw(inp_path, out_path):
    filename = inp_path
    print("\nCompressing", filename)
    arg1 = '-sOutputFile=' + out_path
    # subprocess.call(['C:/Program Files/gs/gs9.53.3/bin/gswin64c.exe',
    #                       '-sDEVICE=pdfwrite',
    #                       '-dCompatibilityLevel=1.7',
    #                       '-dPDFSETTINGS=/ebook', '-dNOPAUSE',
    #                       '-dBATCH', '-dQUIET', str(arg1), filename],
    #                      stdout=subprocess.PIPE)
    subprocess.call(['C:/Program Files/gs/gs9.53.3/bin/gswin64c.exe',
                     '''''' '-dSimulateOverprint = true',
                     '-sDEVICE=pdfwrite',
                     '-dCompatibilityLevel=1.7',
                     '-dPDFSETTINGS=/screen',
                     '-dAutoRotatePages=/None',
                     '-dColorImageDownsampleType=/Bicubic',
                     '-dColorImageResolution=120',
                     '-dDownsampleColorImages=true',
                     '-dGrayImageDownsampleType=/Bicubic',
                     '-dGrayImageResolution=120',
                     '-dDownsampleGrayImages=true',
                     '-dMonoImageResolution=120',
                     '-dDownsampleMonoImages=true',
                     '-dNOPAUSE', '-dBATCH', '-dQUIET', str(arg1), filename])
    print(f'Compressed {filename}')


def read_data(data_path):
    try:
        # candidate no, surname, first, preferred, subj, set, access arrangements
        XX_data = pd.read_excel(data_path, sheet_name="XX", usecols="A:H").values
        DB_data = pd.read_excel(data_path, sheet_name="DB", usecols="A:H").values
        LXX_data = pd.read_excel(data_path, sheet_name="LXX", usecols="A:H").values
        EB_data = pd.read_excel(data_path, sheet_name="EB", usecols="A:H").values
        FB_data = pd.read_excel(data_path, sheet_name="FB", usecols="A:H").values

        XX_data = [list(_) + ["XX"] for _ in
                   XX_data]  # now: candidate no, surname, first, preferred, subj, set, AAs, yeargroup
        DB_data = [list(_) + ["DB"] for _ in DB_data]
        LXX_data = [list(_) + ["LXX"] for _ in LXX_data]
        EB_data = [list(_) + ["EB"] for _ in EB_data]
        FB_data = [list(_) + ["FB"] for _ in FB_data]
        data = XX_data + DB_data + LXX_data + EB_data + FB_data

        # quick cleanup to turn everything to strings and clear whitespace:
        for s, student in enumerate(data):
            for i, detail in enumerate(student):
                data[s][i] = str(detail).strip()

        data = sorted(data, key=lambda student: student[6])
        # print(data)
        # data = [DB_data,XX_data]
        return data
    except FileNotFoundError:
        raise FileNotFoundError("I am not seeing a file called 'data.xlsx' and it is the only thing I understand...")


def make_qr(details_list, paper_index):
    # encodes candidate no, name, subject, yeargroup and assignment number
    # this is way too much information to encode, could just do candidate no and subject...
    # ...but trying to avoid having to look-up on decoding, so oh well.
    details = details_list[0:2] + [details_list[3]]  # candidate number, surname and preferred name
    details += [details_list[4]] + [str(paper_index)]  # subject name and assignment number
    details += [details_list[5]] + [details_list[-1]]  # setcode and yeargroup

    details = "_".join(details)
    img = qrcode.make(details)
    img.save('qr.png')  # 7690_Zheng_Peter_Art_1_DA/AR_DB


def locate_raw_assignments(student_info):
    subject = student_info[4]
    yeargroup = student_info[-1]
    set = student_info[5]
    # Don't need below special circumstances anymore.
    # if subject == "Politics":
    #     if "X1" in set:
    #         path = os.path.join("Processing", blank, raw, subject, yeargroup, "X1")
    #     else:
    #         path = os.path.join("Processing", blank, raw, subject, yeargroup, "others")
    # else:
    #     path = os.path.join("Processing", blank, raw, subject, yeargroup)
    # print(yeargroup)
    path = os.path.join("Processing", blank, raw, subject, yeargroup)
    try:
        assignments = {f: [os.path.join(path), get_assignment_number(f)] for f in os.listdir(path) if
                       os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".pdf") and f.lower()[0:3] != "zzz"}
    except FileNotFoundError:
        # print(subject, yeargroup)
        assignments = {}
    return assignments


def get_assignment_number(filename):
    # assuming HODs stick to naming convention, but not too reliably
    # so, geography_1 and geo_1 and Geography-1 should all mean "assignment number 1"
    for char in filename:
        if char.isdigit():
            return int(char)
    return


def make_directories(data):
    for student in data:
        # make dirs for blank unstamped assignments to go into for each department
        raw_path = os.path.join("Processing", blank, raw, student[4], student[-1])
        os.makedirs(raw_path, exist_ok=True)

        raw_path = os.path.join("Processing", blank, raw, student[4], student[-1], "For remote students")
        os.makedirs(raw_path, exist_ok=True)

        raw_path = os.path.join("Processing", completed, student[4], student[-1], soft_files)
        os.makedirs(raw_path, exist_ok=True)

        raw_path = os.path.join("Processing", completed, student[4], student[-1], split_scans)
        os.makedirs(raw_path, exist_ok=True)

        raw_path = os.path.join("Processing", completed, student[4], student[-1], unsplit_scans)
        os.makedirs(raw_path, exist_ok=True)

        # make dirs for departments to dump marked papers into
        raw_path = os.path.join("Processing", marked, raw, student[4])  # don't need yeargroup for this one
        os.makedirs(raw_path, exist_ok=True)

        # make dirs for student portfolio evidence to go into
        name = f"{student[1]} {student[3]} ({student[0]})"
        setcode = student[5]
        setcode = setcode.replace("/", "-")
        portfolio_path = os.path.join("Processing", marked, portfolios, student[4], student[-1], setcode, name)
        os.makedirs(portfolio_path, exist_ok=True)


def get_subfolders(rawpath):
    subj_folders = [subj for subj in os.listdir(rawpath) if not os.path.isfile(os.path.join(rawpath, subj))]
    return subj_folders


def get_full_paths_marked(rawpath, subj_folders):
    files = {}
    for subj in subj_folders:
        folder_path = os.path.join(rawpath, subj)
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                      os.path.isfile(os.path.join(folder_path, f))]
        files[subj] = file_paths
    return files


def get_full_paths_unmarked(rawpath, subj_folders):
    files = {}
    for subj in subj_folders:
        files[subj] = []
        for yg in ["DB", "XX", "LXX", "EB", "FB"]:
            for folder in [unsplit_scans, soft_files]:
                folder_path = os.path.join(rawpath, subj, yg, folder)
                try:
                    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                                  os.path.isfile(os.path.join(folder_path, f))]
                    files[subj] += file_paths
                except FileNotFoundError:
                    pass
    return files


def check_file_exists(paper_index, student):
    subject = student[4]
    examcode = f"{student[0]}"
    yeargroup = student[-1]
    setcode = student[5]
    setcode = setcode.replace("/", "-")
    dest = os.path.join("Processing", blank, stamped, subject, yeargroup, "Assignment " + str(paper_index), setcode)
    file_path_noAA = os.path.join(dest, f"{examcode} {subject} {paper_index}.pdf")
    file_path_AA = os.path.join(dest, f"_{AA_stamp} {examcode} {subject} {paper_index}.pdf")
    return os.path.isfile(file_path_noAA) or os.path.isfile(file_path_AA)


def print_summary(summary):
    print()
    for filename in sorted(summary.keys()):
        print(
            f"{filename} {'-' * (35 - len(filename))} {summary[filename][0]} new, {summary[filename][1]} existing files.")


def print_sort_summary(summary):
    print()
    for ygsubj in sorted(summary.keys()):
        print(
            f"{ygsubj} {'-' * (35 - len(ygsubj))} QR: {summary[ygsubj][0]}, name: {summary[ygsubj][1]} {'-' * (5 - len(str(summary[ygsubj][2])))} {summary[ygsubj][2]} existing files.")


def stamp_n_save_individual(data):
    # there are cases when students in same set don't sit same assignments, e.g. Further Maths GCSE
    setsubj = group_students_by_subjectset(data)  # so group by subject+set combo key
    summary = {}  # a dictionary to store a summary of updates
    for key in sorted(setsubj.keys()):  # for every unique set+subj combo...
        # assume that all students in this set+subj will sit the same assignments...
        student0 = setsubj[key][0]  # ...so get the first student
        assignments = locate_raw_assignments(student0)  # and find all assignments for this student for this subject
        # outer loop over the assignments (for every assignment...)
        for filename in assignments.keys():
            sum_key = f"{student0[-1]}_{filename}"
            summary[sum_key] = summary.get(sum_key, [0, 0])
            assignment_number = assignments[filename][1]
            dir_path = assignments[filename][0]
            raw_doc = compress_n_get_doc(dir_path, filename)  # this is the assignment that will get stamped for each
            count = 0
            for student in setsubj[key]:  # ...for every student in this subject-set
                if not check_file_exists(assignment_number, student):
                    count += 1
                    summary[sum_key][0] += 1
                    make_qr(student, assignment_number)  # qr code is saved locally and overwritten each time
                    # incremental processing of the paper:
                    newpaper = pdf_prepare_for_stamping(dir_path, raw_doc, filename)  # add blank cover
                    qr_paper = pdf_stamp_QR(newpaper, assignment_number, student, "qr.png")  # add info to front cover
                    stamped_paper = pdf_stamp_page_info(qr_paper, assignment_number, student)  # add info to page footer
                    pdf_save_stamped_single(stamped_paper, assignment_number, student)
                else:
                    summary[sum_key][1] += 1
                    # print(f"{student[0]} {student[1:3]} already has {filename}.")
            if count > 0:
                print(f"Saved {count} new files for {key}")
            raw_doc.close()
    print_summary(summary)


def group_students_by_subjectset(data):
    setsubj = {}
    for student in data:
        key = student[4] + student[5]
        setsubj[key] = setsubj.get(key, []) + [student]

    return setsubj


def sort_marked_files(qr_scanner):
    # scanning for qr and splitting scanned-in papers
    #   Look into "Marked Assignments/Raw" to find list of subject subfolders
    #   For each subfolder, look inside to find list of files...
    #   For each file, if it is PDF - do the exsplit thing with QR scanner and everything...
    #   ...if that fails - do move by filename
    #   If file is not PDF - do move by filename straight away
    #   There is a "marked_log.txt" that tracks which files have been processed already.
    #   The marked log also tracks how many papers were in each pdf.
    update_existing_marked_log()
    summary = {}
    marked_raw_path = os.path.join("Processing", marked, raw)
    subj_folders = get_subfolders(marked_raw_path)
    full_paths = get_full_paths_marked(marked_raw_path, subj_folders)
    for subj in full_paths.keys():
        for full_path in full_paths[subj]:
            sum_key = subj  # no access to year group information at this stage, not important.
            # summary will contain [new QR, new by name, existing]
            summary[sum_key] = summary.get(sum_key, [0, 0, 0])

            if file_already_processed(full_path, "marked"):
                # print(f"{full_path} processed already.")
                num = marked_log[1][marked_log[0].index(make_file_hash(full_path))]
                summary[sum_key][2] += num
                continue

            if full_path.lower().endswith(".pdf"):
                filename = full_path.split("\\")[-1]
                file_copied = move_by_filename(full_path, subj, filename, "marked")
                if file_copied == "ERROR":
                    exsplit = Exam_Splitter(full_path, qr_scanner)
                    qr_found = exsplit.identify_papers()
                    # exsplit = Exam_Splitter(full_path, qr_scanner)
                    # qr_found = exsplit.process_exams("marked")
                    if qr_found:
                        files_written = exsplit.process_exams("marked")
                        summary[sum_key][0] += files_written[0]
                        summary[sum_key][2] += files_written[1]

                        add_to_marked_log(full_path, sum(files_written))  # record total number of documents
                    else:
                        raise ValueError("Wrong filename and no QR codes...")
                else:
                    # success = True

                    add_to_marked_log(full_path, 1)
                    if file_copied:
                        summary[sum_key][1] += 1
                    else:
                        summary[sum_key][2] += 1

                ### Changed the order, below used to do QR code check first, followed by filename move...
                ### The above is more suitable in reality: filename move attempt first, followed by QR scan.
                # exsplit = Exam_Splitter(full_path, qr_scanner)
                # qr_found = exsplit.identify_papers()
                # # exsplit = Exam_Splitter(full_path, qr_scanner)
                # # qr_found = exsplit.process_exams("marked")
                # if qr_found:
                #     files_written = exsplit.process_exams("marked")
                #     summary[sum_key][0] += files_written[0]
                #     summary[sum_key][2] += files_written[1]
                #
                #     add_to_marked_log(full_path, sum(files_written))  # record total number of documents
                # else:
                #     filename = full_path.split("\\")[-1]
                #     file_copied = move_by_filename(full_path, subj, filename, "marked")
                #     if file_copied == "ERROR":
                #         # thorough = True
                #         raise ValueError("Dodgy filename... ")
                #     else:
                #         # success = True
                #
                #         add_to_marked_log(full_path, 1)
                #         if file_copied:
                #             summary[sum_key][1] += 1
                #         else:
                #             summary[sum_key][2] += 1
            else:
                filename = full_path.split("\\")[-1]
                file_copied = move_by_filename(full_path, subj, filename, "marked")

                if file_copied == "ERROR":
                    raise ValueError("Dodgy filename... ")
                else:
                    add_to_marked_log(full_path, 1)
                    if file_copied:
                        summary[sum_key][1] += 1
                    else:
                        summary[sum_key][2] += 1
            write_marked_log_changes()  # moved this in so that the marked log is written after every file
    print_sort_summary(summary)


def update_existing_unmarked_log():
    log = open("unmarked_log.txt", "r")
    for line in log:
        unmarked_log.append(line.strip())
    log.close()


def update_existing_marked_log():
    log = open("marked_log.txt", "r")
    for line in log:
        entry_str = line.strip()
        entry = entry_str.split(",")  # the log keeps "filenamesizetime,7" e.g., where 7 is number of papers in the file
        marked_log[0].append(entry[0])  # the internal list has shape [[filehash,filehash..],[number of papers, num...]]
        marked_log[1].append(int(entry[1]))
    log.close()


def make_file_hash(fullpath):
    entry = fullpath
    fsize = str(os.path.getsize(fullpath))
    mtime = str(os.path.getmtime(fullpath))
    entry += fsize + mtime
    return entry


def add_to_unmarked_log(fullpath):
    entry = make_file_hash(fullpath)
    unmarked_log.append(entry)


def add_to_marked_log(fullpath, num):
    entry = make_file_hash(fullpath)
    marked_log[0].append(entry)
    marked_log[1].append(num)


def write_unmarked_log_changes():
    to_write = list(set(unmarked_log))
    to_write.sort()
    log = open("unmarked_log.txt", "w")
    for thing in to_write:
        log.write(thing + "\n")


def write_marked_log_changes():
    if len(marked_log[0]) != len(marked_log[1]):
        raise ValueError("Marked log length mismatch!")

    to_write = []
    for i in range(len(marked_log[0])):
        entry = f"{marked_log[0][i]},{marked_log[1][i]}"
        if entry not in to_write:
            to_write.append(entry)
    to_write.sort()
    log = open("marked_log.txt", "w")
    for thing in to_write:
        log.write(thing + "\n")


def file_already_processed(fullpath, filetype):
    # check either the unmarked log or the marked log to see if already done
    entry = make_file_hash(fullpath)

    if filetype == "unmarked":
        if entry in unmarked_log:
            return True
    elif filetype == "marked":
        if len(marked_log) > 0 and entry in marked_log[0]:
            # marked_log list has two halves [[filehashes],[contained papers numbers]]
            return True
    return False


def sort_unmarked_files(qr_scanner):
    update_existing_unmarked_log()
    summary = {}
    unmarked_path = os.path.join("Processing", completed)
    subj_folders = get_subfolders(unmarked_path)
    full_paths = get_full_paths_unmarked(unmarked_path, subj_folders)
    for subj in full_paths.keys():
        for full_path in full_paths[subj]:
            if "\\XX\\" in full_path:
                yg = "XX"
            elif "\\DB\\" in full_path:
                yg = "DB"
            elif "\\EB\\" in full_path:
                yg = "EB"
            elif "\\FB\\" in full_path:
                yg = "FB"
            elif "\\LXX\\" in full_path:
                yg = "LXX"
            else:
                yg = "ERROR"
            sum_key = f"{yg} {subj}"
            # summary will contain [new QR, new by name, existing]
            summary[sum_key] = summary.get(sum_key, [0, 0, 0])

            if file_already_processed(full_path, "unmarked"):
                # print(f"{full_path} processed already.")
                summary[sum_key][2] += 1
                continue

            if full_path.lower().endswith(".pdf"):
                exsplit = Exam_Splitter(full_path, qr_scanner)
                qr_found = exsplit.identify_papers()
                if qr_found:
                    files_written = exsplit.process_exams("unmarked")
                    summary[sum_key][0] += files_written[0]
                    summary[sum_key][2] += files_written[1]
                    add_to_unmarked_log(full_path)  # by this point a file is definitely processed already

                else:
                    filename = full_path.split("\\")[-1]
                    file_copied = move_by_filename(full_path, subj, filename, "unmarked")
                    # the line above raises error if problem

                    # regardless of whether the file was copied anew, or already existed:
                    # make a primitive hash of the file parameters
                    # and add to the log to indicate that this combination does not have to be processed again.

                    # this point is never reached for already logged files and...
                    # this automatically updates log for existing but unlogged files.

                    if file_copied == "ERROR":
                        # thorough = True
                        raise ValueError("Dodgy filename... ")
                    else:
                        # success = True

                        add_to_unmarked_log(full_path)
                        if file_copied:
                            summary[sum_key][1] += 1
                        else:
                            summary[sum_key][2] += 1

                    # add_to_unmarked_log(full_path)  # by this point a file is definitely processed already
                    # if file_copied:
                    #     summary[sum_key][1] += 1
                    # else:
                    #     summary[sum_key][2] += 1
            else:
                filename = full_path.split("\\")[-1]
                file_copied = move_by_filename(full_path, subj, filename, "unmarked")

                if file_copied == "ERROR":
                    raise ValueError("Dodgy filename... ")
                else:
                    add_to_unmarked_log(full_path)
                    if file_copied:
                        summary[sum_key][1] += 1
                    else:
                        summary[sum_key][2] += 1
                # add_to_unmarked_log(full_path)
                # if file_copied:
                #     summary[sum_key][1] += 1
                # else:
                #     summary[sum_key][2] += 1
    write_unmarked_log_changes()
    print_sort_summary(summary)


def shorten_path(long_path):
    long_path = long_path.split("\\")
    useful_part_start = 0
    for upper_level in [blank, completed, marked]:
        try:
            useful_part_start = long_path.index(upper_level)
            valid_path = True
            break
        except ValueError:
            valid_path = False
    if valid_path:
        useful_part = long_path[useful_part_start:]
        short_path = f"...\\{os.path.join(*useful_part)}"
    else:
        return "INCORRECT PATH SOMEHOW"
    return short_path


def get_dir_differences(dcmp, differences, upper_level):
    for name in dcmp.left_only:
        # print(f"Cloud: {name} is new at {shorten_path(dcmp.left)}")
        full_path = os.path.join(dcmp.left, name)
        differences["Cloud"][upper_level].append(full_path)

    for name in dcmp.right_only:
        # print(f"Local: {name} is new at {dcmp.right}")
        full_path = os.path.join(dcmp.right, name)
        differences["Local"][upper_level].append(full_path)

    for name in dcmp.diff_files:
        # print(f"Difference: {name} is different for {12}")
        full_path = os.path.join(dcmp.right, name)
        differences["Difference"][upper_level].append(full_path)

    for sub_dcmp in dcmp.subdirs.values():
        get_dir_differences(sub_dcmp, differences, upper_level)
    return differences


def display_difference_detail(differences):
    buffer = 12
    for key in differences:
        print()
        for inner_key in differences[key]:
            for file_path in differences[key][inner_key]:
                print(f"{key} {'─' * (buffer - len(key))} {inner_key}: {shorten_path(file_path)}")
    print()


def display_difference_summary(differences):
    buffer = 25
    cloud = differences["Cloud"]
    label = "Cloud only"
    print(f"{label} {'-' * (buffer - len(label))} {', '.join([f'{k}: {len(cloud[k])}' for k in cloud])}")

    local = differences["Local"]
    label = "Local only"
    print(f"{label} {'-' * (buffer - len(label))} {', '.join([f'{k}: {len(local[k])}' for k in local])}")

    diff = differences["Difference"]
    label = "Different files"
    print(f"{label} {'-' * (buffer - len(label))} {', '.join([f'{k}: {len(diff[k])}' for k in diff])}")


def copy_from_cloud(differences):
    cloud = differences["Cloud"]
    for upper_level in cloud:
        for input_path in cloud[upper_level]:
            output_path = shorten_path(input_path)[4:]
            output_path = os.path.join(absolute_workdir, output_path)
            shutil.copy(input_path, output_path)
            print(f"Copied {output_path}.")


def copy_from_local(differences):
    local = differences["Local"]
    for upper_level in local:
        for input_path in local[upper_level]:
            output_path = shorten_path(input_path)[4:]
            output_path = os.path.join(absolute_cloud, output_path)
            shutil.copy(input_path, output_path)
            print(f"Copied {output_path}.")


def push_pull_updates(differences):
    choice = input(
        """\nChoose an additive update:
    1. To COPY updates FROM CLOUD into local. 
    2. To COPY updates FROM LOCAL into cloud.
    3. To QUIT.\n""")  # no validation whatsoever...
    if choice == "1":
        copy_from_cloud(differences)
    elif choice == "2":
        copy_from_local(differences)
    elif choice == "3":
        quit()
    push_pull_updates(differences)


def check_cloud_for_differences():
    levels = [blank, completed, marked]
    update_types = ["Cloud", "Local", "Difference"]
    differences = {ku: {kl: [] for kl in levels} for ku in update_types}

    for upper_level in levels:
        cloud = os.path.join(absolute_cloud, upper_level)
        local = os.path.join(absolute_workdir, upper_level)
        dir_comparison = dircmp(cloud, local)
        differences = get_dir_differences(dir_comparison, differences, upper_level)

    display_difference_detail(differences)
    display_difference_summary(differences)

    push_pull_updates(differences)


def tidy_columns(filename, df_index):
    file = pd.ExcelFile(filename)
    dfs = pd.read_excel(file, sheet_name=None, usecols="A:Z", header=None)

    df_index.loc[-1] = "Click on subject name"
    df_index.index = df_index.index + 1
    df_index.sort_index(inplace=True)

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    bold = workbook.add_format({'bold': True})

    for sheetname, df in dfs.items():  # loop through `dict` of dataframes

        df[0] = df_index["Click subject"]
        df.to_excel(writer, sheet_name=sheetname, index=False, header=False)  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object

        for idx, col in enumerate(df):  # loop through all columns
            if idx == 0:
                worksheet.set_column(idx, idx, 25, bold)  # set column width
                continue
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
            )) + 3  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width

    writer.save()


def make_portfolio_spreadsheet():
    # Walks through the portfolio folders for each subject and counts the number of assignments for each student.
    # Outputs a single spreadsheet with a tab for each subject, with students sorted by set.
    # Knows expected numbers of assignments: 6 for XX, 3 for DB, shows percentage completion of the expectation.

    # Upon further research, the below is an awful approach...
    # Apparently, .append() on DataFrames is very slow and it is better to create lists first and then convert to DF.
    # Having spent two hours on the below, I am not touching it again.
    # If it ain't broke...

    portfolio_data = {}
    port_path = os.path.join("Processing", marked, portfolios)
    blank = pd.DataFrame([["", "", ""]], columns=["Set", "Student", "Listing"])
    for subject in os.listdir(port_path):
        subj_level = os.path.join(port_path, subject)
        stripped_subj = subject.replace(" ", "").replace("&", "And")
        portfolio_data[stripped_subj] = portfolio_data.get(stripped_subj,
                                                           [None, None, None, None, None])  # a None for every yeargroup...
        for yg in os.listdir(subj_level):
            yg_level = os.path.join(subj_level, yg)
            # print(yg_level)
            try:
                for set_num, setcode in enumerate(os.listdir(yg_level)):
                    set_level = os.path.join(yg_level, setcode)
                    try:
                        for stu_num, student_dir in enumerate(os.listdir(set_level)):
                            student_path = os.path.join(set_level, student_dir)
                            if os.path.isfile(student_path):
                                print(student_path, "is not a directory")
                                continue

                            # Need to do better than below since a count is not enough anymore.
                            # count = len([1 for _ in list(os.scandir(student_path)) if _.is_file()])
                            #
                            # data_row = pd.DataFrame([[setcode, student_dir, count]], columns=["Set", "Student", "Count"])

                            listing = ""
                            for file in list(os.scandir(student_path)):
                                if file.is_file():
                                    try:
                                        assignment_number = parse_filename(file.name)[1]  # gives assignment number only
                                        listing += str(assignment_number) + " "
                                    except TypeError:
                                        listing += "0 "

                            data_row = pd.DataFrame([[setcode, student_dir, listing]],
                                                    columns=["Set", "Student", "Listing"])

                            ind = None
                            if yg == "DB":
                                ind = 0
                            elif yg == "XX":
                                ind = 1
                            elif yg == "LXX":
                                ind = 2
                            elif yg == "EB":
                                ind = 3
                            elif yg == "FB":
                                ind = 4
                            # print(portfolio_data[stripped_subj], ind)
                            if portfolio_data[stripped_subj][ind] is None:
                                portfolio_data[stripped_subj][ind] = data_row
                            else:
                                if stu_num == 0 and set_num > 0:
                                    data_row = blank.append(data_row, ignore_index=True)
                                portfolio_data[stripped_subj][ind] = portfolio_data[stripped_subj][ind].append(data_row,
                                                                                                               ignore_index=True)
                    except NotADirectoryError:
                        print(set_level, "is not a directory")
            except NotADirectoryError:
                print(yg_level, "is not a directory")

    with pd.ExcelWriter("Portfolio tracking.xlsx") as writer:
        spreadsheet_index = ['=HYPERLINK("' + f'#{subj}!A1"' + f',"{subj}")' for subj in sorted(portfolio_data.keys())]
        df_index = pd.DataFrame(spreadsheet_index, columns=["Click subject"])

        for subj in sorted(portfolio_data.keys()):
            df_index.to_excel(writer, sheet_name=subj, index=False)
            for ind in [0, 1, 2, 3, 4]:
                if portfolio_data[subj][ind] is not None:
                    portfolio_data[subj][ind].to_excel(writer, sheet_name=subj, index=False, startcol=ind * 5 + 3)

    tidy_columns("Portfolio tracking.xlsx", df_index)


def main():
    # setting up the parameters
    global data  # yes, yes... has to be done for student name lookup for parsing by filename... not reading file again
    data = read_data("data.xlsx")  # yup, hardcoded...

    choice = input(
        """Choose:
    1. To preset the directories (only needs to be done once usually)
    2. Generate stamped papers from raw blanks
    3. Sort marked papers into portfolio folders
    4. Check for file updates both ways.
    5. Generate portfolio completion spreadsheet.\n""")  # no validation whatsoever...

    if choice == "1":
        make_directories(data)
    elif choice == "2":
        stamp_n_save_individual(data)

    elif choice == "3":
        config = {"shrink": 300,  # Shrinking more seems to have improved recognition! 200 is the new winner.
                  "crop": parse_cropping("0,0.6:0.3,1.0", False),  # Tweaked to look at smaller area.
                  "threshold": 120}  # Different values for different documents it seems...
        # Physics : approx 110
        # French : approx 160
        # German : 120¬160 (SCR scanner is inconsistent)
        qr_scanner = QRScanner(**config)  # the qr magic-ator
        choice_sorting = input(
            """Choose:
        1. Sort completed(unmarked) papers
        2. Sort the marked papers\n""")  # no validation whatsoever...

        if choice_sorting == "1":
            # looking for blanks, creating and stamping individual papers
            sort_unmarked_files(qr_scanner)
        elif choice_sorting == "2":
            sort_marked_files(qr_scanner)
    elif choice == "4":
        check_cloud_for_differences()
    elif choice == "5":
        make_portfolio_spreadsheet()


if __name__ == "__main__":
    main()
