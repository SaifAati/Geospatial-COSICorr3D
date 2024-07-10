"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import fnmatch
import os
import shutil
from shutil import copyfile, rmtree
from typing import List, Optional


def get_files_based_on_extension(dir, filter="*.tif", disp=False):
    import glob
    os.chdir(dir)
    files_list = []

    for file in glob.glob(filter):
        files_list.append(os.path.join(dir, file))
    files_list.sort()
    if disp:
        print("The list of ", filter, " files are:", files_list, " Ntot=", len(files_list))

    return files_list


def get_files_based_on_extensions(dir, filter_list: Optional[List] = None):
    import glob
    if filter_list is None:
        filter_list = ["*.tif", "*.vrt"]
    os.chdir(dir)
    files_list = []

    for filter in filter_list:
        for file in glob.glob(filter):
            files_list.append(os.path.join(dir, file))
    files_list.sort()

    return files_list


def FilesInDirectory(path, exclusionFilter:Optional[List]=None, displayFile=False):

    if exclusionFilter is None:
        exclusionFilter = []
    files = os.listdir(path)
    files.sort()
    if exclusionFilter:
        oldFiles = files
        files = []
        for file_ in oldFiles:
            if len(exclusionFilter) > 0:
                check = any(item in file_ for item in exclusionFilter)
                # print(check)
                if not check:
                    # print(file_)
                    files.append(file_)

    if displayFile == True:
        print("The list of  files are:", files, " Ntot=", len(files))
    else:
        print(" Ntot=", len(files))
    files_ = []
    for file_ in files:
        files_.append(os.path.join(path, file_))
    return files_


def GetOnlyTifImages(path):
    """

    Args:
        path:

    Returns:

    """
    imgs = FilesInDirectory(path)
    imgsPath = []
    for index, img_ in enumerate(imgs):
        if ".tif" in img_ and ".hdr" not in img_ and ".aux.xml" not in img_:
            imgsPath.append(os.path.join(path, img_))
    return imgsPath


def DirectoryAsEmpty(directoryPath):
    """

    Args:
        directoryPath:

    Returns:

    """
    files = FilesInDirectory(directoryPath)
    for i in range(len(files)):
        filePath = directoryPath + files[i]
        os.remove(filePath)
    return


def CreateDirectory(directoryPath, folderName, cal=None):
    """
    Create a Folder in the directory.
    Before that this function verify if the name of the folder exist in the directory
    if the folder exist the user will choose either to create new one or delete it
    Args:
        directoryPath:
        folderName:
        cal:

    Returns:

    """

    # define the name of the directory to be created

    path = os.path.join(directoryPath, folderName)

    if os.path.exists(path):
        if cal == None:
            print("<< %s >> folder already exist type y to delete and recreate new one or type n :  " % folderName)
            cal = input("Type y/n:")
            if cal == "y":
                try:
                    rmtree(path)
                    os.makedirs(path, exist_ok=True)
                except OSError:
                    print("Creation of the directory %s failed " % path)
                else:
                    print("Successfully created the directory <<%s>> " % path)
                    return (path)

            if cal == "n":
                return (path)
        if cal == "y":
            try:
                rmtree(path)
                os.makedirs(path, exist_ok=True)
            except OSError:
                print("Creation of the directory %s failed " % path)
            else:
                print("Successfully created the directory <<%s>> " % path)
                return (path)
        if cal == "n":
            return (path)
    else:
        try:

            os.makedirs(path, exist_ok=True)

        except OSError:
            print("Creation of the directory %s failed " % path)

        else:
            print("Successfully created the directory << %s >> " % path)
            return (path)


def CreateTxtOfFiles(inputFolder, outputFileName="ListofImgs.txt"):
    """

    Args:
        inputFolder:
        outputFileName:

    Returns:

    """
    files = FilesInDirectory(path=inputFolder)
    with open(os.path.join(inputFolder, outputFileName), 'w') as f:
        for file_ in files:
            f.write("%s\n" % file_)

    return


def DeleteSubsets(inputFilePath, refTxtFile):
    """
    This function will delete the subset that does not exit in the refFile
    Args:
        inputFilePath:
        refTxtFile:

    Returns:

    """

    ## Read all imgs in subset folder
    files = FilesInDirectory(path=inputFilePath)

    for file_ in files:
        if ".tif" in file_ and file_ not in open(refTxtFile).read():
            print(False)
            print(file_)
            os.remove(inputFilePath + file_)


def CompareTwoFiles(file1, file2):
    """

    Args:
        file1:
        file2:

    Returns:

    """
    lineList1 = [line.rstrip('\n') for line in open(file1)]
    lineList2 = [line.rstrip('\n') for line in open(file2)]
    for f in lineList1:
        if f not in lineList2:
            print(f)
    return


def UncompressFile(compressedFilePath, output=None):
    """

    Args:
        compressedFilePath:
        output:

    Returns:

    """
    import tarfile
    import zipfile
    if tarfile.is_tarfile(compressedFilePath):
        file = tarfile.open(compressedFilePath)

        file.list()
        print(file.getmembers())
        print(file.getnames())
        file.close()

    if zipfile.is_zipfile(compressedFilePath):
        with zipfile.ZipFile(compressedFilePath, 'r') as zip_ref:
            directory_to_extract_to = output
            zip_ref.extractall(directory_to_extract_to)
    return


def UncompressBatch(directoryInput, directoryOutput):
    """

    Args:
        directoryInput:
        directoryOutput:

    Returns:

    """
    filesList = get_files_based_on_extension(dir=directoryInput, filter="*.zip")
    nbTot = len(filesList)
    for index, file_ in enumerate(filesList):
        baseName = os.path.basename(file_)[0:-4]
        print("status :", index + 1, "/", nbTot)
        UncompressFile(compressedFilePath=file_, output=os.path.join(directoryOutput, baseName))

    return


def CopyFile(inputFilePath, outputFolder, overWrite=True):
    """

    Args:
        inputFilePath:
        outputFolder:
        overWrite:

    Returns:

    """
    outputFilePath = os.path.join(outputFolder, os.path.basename(inputFilePath))
    files = os.listdir(outputFolder)
    if os.path.basename(inputFilePath) not in files:
        copyfile(src=inputFilePath, dst=outputFilePath)
    else:
        # print(os.path.basename(inputFilePath), ": exist in destination folder!!")
        if overWrite:
            # print("-- replacing with the new file !! --")
            copyfile(src=inputFilePath, dst=outputFilePath)
        # else:
        #     print("-- Keeping the old one !!--")

    return outputFilePath


def Copyfiles(inputdirectory, destinationDirectory, filter=".NTF"):
    """

    Args:
        inputdirectory:
        destinationDirectory:
        filter:

    Returns:

    """
    for root, dirs, files in os.walk(inputdirectory):
        for name in files:
            if name.endswith((filter)):
                ntfFile = os.path.join(root, name)
                print(ntfFile)
                copyfile(src=ntfFile, dst=os.path.join(destinationDirectory, name))

    return


def LocateFile(pattern, root=os.curdir):
    """
    Locate all files matching supplied filename pattern in and below
    supplied root directory.
    Args:
        pattern:
        root:

    Returns:

    """

    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)


def ExtractSubfiles(inputdirectory, fileExtension:Optional[List]=None, disp=False):
    """

    Args:
        inputdirectory:
        fileExtension:
        disp:

    Returns:

    """
    if fileExtension is None:
        fileExtension = ['.NTF']
    filesList = []
    for root, dirs, files in os.walk(inputdirectory):
        for name in files:
            if any(name.endswith(ele) for ele in fileExtension):
                file = os.path.join(root, name)
                filesList.append(file)
    if disp:
        print("Subfiles:", filesList, "NbTot=", len(filesList))
    return filesList


def ContentFolderDelete(folder, exception=None):
    """

    Args:
        folder:
        exception:

    Returns:

    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename != exception:
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    return


def ContentOfFolder(folderPath):
    """

    Args:
        folderPath:

    Returns:
        fileList
        dirPathList
    References:
        https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/

    """
    fileListNames = []
    dirPathList = []
    dirNamesList = []

    for (dirpath, dirnames, filenames) in os.walk(folderPath):
        fileListNames.append(filenames)
        dirPathList.append(dirpath)
        # print(dirnames,dirpath,filenames)
        # print(dirnames)

    fileList = []
    for dirPath_, list_ in zip(dirPathList, fileListNames):
        for name_ in list_:
            fileList.append(os.path.join(dirPath_, name_))
    #
    print("===> #files:", len(fileList))

    return fileList, dirPathList

