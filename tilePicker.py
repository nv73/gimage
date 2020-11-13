# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:14:27 2020

@author: nick.viner
"""

"""
Side project in an attempt to allow easier access to machine learning methods for 
the classification of geo-imagery with limited sample sizes.
"""

"""
Currently focused on pixel based classification of hyperspectral imagery.
"""
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageQt
from sys import argv
import tilePicker_ui
import gdal
import numpy as np
import osr
from os.path import basename
import pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from skimage.feature import greycomatrix, greycoprops
import pandas as pd

class tilePicker_Form(QtWidgets.QMainWindow, tilePicker_ui.Ui_imageTilePicker):
    
    def __init__(self, parent = None):
        
        super().__init__()
        
        self.setupUi(self)
        
        #Set up the layout for interacting with and viewing geoImages
        self.canvas = geoCanvas()
        self.canvasView.addWidget(self.canvas)
        
        #Set up the layout for interacting with loaded image files
        self.loadedImagesTable = tableWidget(deleteEnabled=False)
        self.loadedImagesTableLayout.addWidget(self.loadedImagesTable)
        self.loadedImagesTable.setColumnCount(1)
        self.loadedImagesTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        #Set up the layout for interacting with selected tiles
        self.selectedTilesTable = tableWidget()
        self.selectedTilesTableLayout.addWidget(self.selectedTilesTable)
        self.selectedTilesTable.setColumnCount(1)
        self.selectedTilesTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.selectedTilesTable.setHorizontalHeaderLabels(["Index, X, Y, Class, Filename, Sample Datapoint"])
        
        # Containers for GUI variables
        self.tileSize = self.tileSizeEdit.text()
        
        #Signals
        self.tileSizeEdit.textChanged.connect(self.tileSizeChanged)
        self.actionLoad_Images.triggered.connect(self.loadImage)
        self.loadedImagesTable.itemDoubleClicked.connect(self.changeActiveImage)
        self.tileSelectRadio.toggled.connect(self.tileSelectRadioUpdate)
        self.spectrumSelectRadio.toggled.connect(self.spectrumSelectRadioUpdate)
        self.bandSelectCombo.currentTextChanged.connect(self.updateActiveBand)
        self.bandSelectCombo.currentTextChanged.connect(self.canvas.displayGeoImage)
        self.classNameCombo.currentTextChanged.connect(self.updateActiveClass)
        self.canvas.tileAdded.connect(self.updateSelectedTiles)
        self.selectedTilesTable.cellDeleted.connect(self.updateTileIndexOnDelete)
        self.actionAdd_New_Object_Type.triggered.connect(self.addNewObjectClass)
        self.actionExport_Objects_to_File.triggered.connect(self.canvas.exportTilesToFile)
        self.actionImportMultiDetect.triggered.connect(self.canvas.import_Multidetect)
        self.actionClassifyVegetation.triggered.connect(self.canvas.veggiPy)
        
        #Train
        self.actionAdaBoostTrain.triggered.connect(self.canvas.trainAdaBoost)
        self.actionRfTrain.triggered.connect(self.canvas.trainRandomForest)
        
        #Classify
        self.actionAdaBoostClassify.triggered.connect(lambda i: self.canvas.classify_all("adaBoost"))
        self.actionRfClassify.triggered.connect(lambda i: self.canvas.classify_all("randomForest"))
        
        #Test button
        #self.pushButton.clicked.connect(self.canvas.classify_all)
        #self.pushButton.clicked.connect(lambda i: self.test("i"))
        
        #Add default object classes
        defaultClasses = ["Vegetation","Tree", "Snow", "Water", "Asphalt", "Soil", "Rock"]
        
        for i in defaultClasses:
            
            self.classNameCombo.addItem(i)
    
    def test(self, inputv):
        
        print(inputv)
        
    def addNewObjectClass(self):
        
        newClass = QtWidgets.QInputDialog.getText(self, "Input New Class", "Input New Class")

        self.classNameCombo.addItem(newClass[0])
    
    def updateTileIndexOnDelete(self):
        
        valueToRemove = self.selectedTilesTable.lastDeletedCell.split(",")[0]
        
        self.canvas.tiles[int(valueToRemove)].cleared = True
    
    def updateSelectedTiles(self):
        
        x = self.canvas.tiles[self.canvas.tile_index].x
        y = self.canvas.tiles[self.canvas.tile_index].y
        z = int(self.canvas.activeBand) - 1
        
        tileInfo = "%i, %i, %i, %s, %s, %i" % (self.canvas.tile_index, self.canvas.tiles[self.canvas.tile_index].x, 
                                   self.canvas.tiles[self.canvas.tile_index].y,
                                   self.canvas.tiles[self.canvas.tile_index].classification,
                                   basename(self.canvas.activeGeoImagePath),
                                   self.canvas.activeGeoArray[z, y, x]
                                   )
        
        if self.canvas.tiles[self.canvas.tile_index].classification not in self.canvas.orderedTileClasses:
            
            self.canvas.orderedTileClasses.append(self.canvas.tiles[self.canvas.tile_index].classification)
            
            if self.canvas.tiles[self.canvas.tile_index].classification == "Vegetation":
            
                self.canvas.vegetationLabelIndex = self.canvas.orderedTileClasses.index("Vegetation")
        
        self.selectedTilesTable.addToNext(tileInfo)
        
    def tileSelectRadioUpdate(self):
        
        #Prevent user from using multiple selection types when choosing tiles
        if len(self.canvas.tiles) > 0:
            
            self.spectrumSelectRadio.setEnabled(False)
            
        else:
        
            self.tileSizeEdit.setReadOnly(False)
            
            self.spectrumSelectRadio.setChecked(False)
            
            self.canvas.isSpectrum = False
        
    def spectrumSelectRadioUpdate(self):
        
        #Prevent user from using multiple selection types when choosing tiles
        if len(self.canvas.tiles) > 0:
            
            self.tileSelectRadio.setEnabled(False)
            
        else:
        
            self.tileSizeEdit.setReadOnly(True)
            
            self.tileSelectRadio.setChecked(False)
            
            self.canvas.isSpectrum = True
            
    def updateActiveBand(self):
        
        self.canvas.activeBand = str(self.bandSelectCombo.currentText())
        
    def updateActiveClass(self):
        
        self.canvas.activeClass = str(self.classNameCombo.currentText())
    
    #Runs whenever the lineedit for tile size is updated
    #This changes the dimensions of the array created when shift+clicking the geoCanvas
    def tileSizeChanged(self):
        
        try:
            
            self.canvas.patchSize = int(self.tileSizeEdit.text())
            
        except:
            
            self.canvas.patchSize = 10
    
    #Loads a tif image
    def loadImage(self):
        
        filePath = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image File...", "", ".tif(*.tif)")
        
        self.canvas.importGeoImage(str(filePath[0]))
        
        self.canvas.activeGeoImagePath = str(filePath[0])
        
        self.loadedImagesTable.addToNext(str(filePath[0]))

        self.updateBandCombo(self.canvas.geoImage[str(filePath[0])].bands)
        
        if self.canvas.activeGeoArray.shape[0] == 1:
            
            self.tileSelectRadio.setCheckable(True)
            
            self.tileSelectRadio.setChecked(True)
            
            self.tileSizeEdit.setReadOnly(False)
                        
        else:
            
            self.tileSelectRadio.setCheckable(True)
            
            self.spectrumSelectRadio.setCheckable(True)
            
            self.spectrumSelectRadio.setChecked(True)
            
        self.canvas.activeBand = str(self.bandSelectCombo.currentText())
    
    #Changes which image is being interacted with
    def changeActiveImage(self):
        
        activeFile = self.loadedImagesTable.currentItem().text()

        self.canvas.changeGeoImage(activeFile)

        self.updateBandCombo(self.canvas.geoImage[activeFile].bands)
    
    #updates the combobox containing the band numbers to corrected reflect
    #the currently active image
    def updateBandCombo(self, bands):
        
        self.bandSelectCombo.clear()
        
        self.canvas.total_bands = bands
        
        for f in range(1, bands + 1):
            
            self.bandSelectCombo.addItem(str(f))

class tableWidget(QtWidgets.QTableWidget):
    
    cellDeleted = QtCore.pyqtSignal()
    
    def __init__(self, deleteEnabled = True):
        
        super(tableWidget, self).__init__()
        
        self.totalRows = 0
        
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
        if deleteEnabled == True:
            
            self.itemDoubleClicked.connect(self.deleteCell)
        
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        
        self.lastDeletedCell = None
        
    def deleteCell(self):
        
        cleared = QtWidgets.QTableWidgetItem("Cleared")
        
        self.lastDeletedCell = self.currentItem().text()
        
        self.setItem(self.currentItem().row(), self.currentItem().column(), cleared)
        
        self.cellDeleted.emit()
        
    def addToNext(self, inVal, column = 0):
        
        newItem = QtWidgets.QTableWidgetItem(inVal)
        
        rows = self.rowCount()
        
        if rows == 0:
            
            self.setRowCount(1)
            
            self.setItem(0,column, newItem)
        
        else:
            
            self.setRowCount(rows + 1)
            
            for i in range(rows + 1):
                
                self.setCurrentCell(i, column)
                
                if self.currentItem():
            
                    i+= 1
                    
                else:
                    
                    self.setItem(i, column, newItem)
                    
                    i+= 1
        
        self.resizeColumnsToContents()
              
class geoCanvas(QtWidgets.QGraphicsView):
    
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)
    
    tileAdded = QtCore.pyqtSignal()
    
    spectrumAdded = QtCore.pyqtSignal()
    
    def __init__(self):

        super(geoCanvas, self).__init__()
        
        # Initialize the scene for the graphics view
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self._QtImage = QtWidgets.QGraphicsPixmapItem()

        # QGraphicsView properties
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setMinimumSize(500, 400)
        
        # some helper variables
        self._empty = True
        self._zoom = 0

        # Holder for geoImages
        self.geoImage = {}
        self.geoImage_index = 0
        
        #Tile selection variables
        self.patchSize = int(10)
        self.isSpectrum = False
        self.activeTileShape = QtWidgets.QGraphicsRectItem(0,0, self.patchSize, self.patchSize)
        self.tiles = {}
        self.tile_index = 0
        self.total_bands = 0

        # Coordinate values for the mouse cursor
        self.mouse_coordinates = None
        self.selected_coordinates = None

        # Graphical coordinate indicators
        self.pointSize = 20
        self.displayed_coordinates = QtWidgets.QGraphicsTextItem()
        self.displayed_coordinates.setTransformOriginPoint(self.displayed_coordinates.boundingRect().topLeft())
        self.displayed_coordinates_font = self.displayed_coordinates.font()
        self.displayed_coordinates_font.setPointSize(self.pointSize)
        self.displayed_coordinates.setFont(self.displayed_coordinates_font)
        self.displayed_coordinates_scale = 1

        # Add the initialized image and text to the graphics view
        self.scene.addItem(self._QtImage)
        self.scene.addItem(self.displayed_coordinates)
        
        #Active geo image information
        self.activeGeoImagePath = None
        self.activeGeoArray = None
        self.normal_activeGeoArray = None
        self.activeBand = 0
        self.activeClass = "Vegetation"
        self.multiDetect = []
        
        #Machine Learning stuff
        self.labels = None
        self.testFeatures = None
        self.models = {}
        self.classified_image = None
        self.orderedTileClasses = []
        self.vegetationLabelIndex = 0

    # Taken from https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsvie
    def setQtImage(self, pixmap=None):

        self._zoom = 0

        if pixmap and not pixmap.isNull():

            self._empty = False

            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

            self.viewport().setCursor(QtCore.Qt.CrossCursor)

            self._QtImage.setPixmap(pixmap)

        else:

            self._empty = True

            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

            self._QtImage.setPixmap(QtGui.QPixmap())

        self.fitInView()

    def hasQtImage(self):

        return not self._empty

    # Taken from https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview
    def fitInView(self, scale=True):

        rect = QtCore.QRectF(self._QtImage.pixmap().rect())

        if not rect.isNull():

            self.setSceneRect(rect)

            if self.hasQtImage():
                
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))

                self.scale(1 / unity.width(), 1 / unity.height())

                viewrect = self.viewport().rect()

                scenerect = self.transform().mapRect(rect)

                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())

                self.scale(factor, factor)

            self._zoom = 0

    def wheelEvent(self, event):

        if not self._empty:

            if event.angleDelta().y() > 0:

                factor = 1.25

                self._zoom += 1

                self.displayed_coordinates.setScale(self.displayed_coordinates_scale)

                self.displayed_coordinates_scale = self.displayed_coordinates_scale * 0.8

            else:

                if self.displayed_coordinates_scale < 1.0:

                    self.displayed_coordinates_scale = self.displayed_coordinates_scale * 1.25

                factor = 0.8

                self._zoom -= 1

                self.displayed_coordinates.setScale(self.displayed_coordinates_scale)

            if self._zoom > 0:

                self.scale(factor, factor)

            elif self._zoom == 0:

                self.fitInView()

            else:

                self._zoom = 0

    def enterEvent(self, event):

        self.viewport().setCursor(QtCore.Qt.CrossCursor)

        super(geoCanvas, self).enterEvent(event)

    def mousePressEvent(self, event):
        
        if self._QtImage.isUnderMouse():
            
            self.photoClicked.emit(QtCore.QPoint(event.pos()))

        selected_coordinates = self.mapToScene(event.x(), event.y())

        self.viewport().setCursor(QtCore.Qt.CrossCursor)
        
        #Draw rectangle on a shift click input
        if QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
            
            #Add a try statement in here to fix any tiles that leave the image borders
            if self.isSpectrum == False:
                
                tileCorner = (int(selected_coordinates.x() - (self.patchSize / 2)), int(selected_coordinates.y() - (self.patchSize / 2)))
            
                rect = QtWidgets.QGraphicsRectItem(0,0, self.patchSize, self.patchSize)
            
                rect.setPos(tileCorner[0], tileCorner[1])
                
                self.scene.addItem(rect)
                newTile = tile(self.activeGeoArray[:,tileCorner[1]:tileCorner[1] + self.patchSize, tileCorner[0]:tileCorner[0] + self.patchSize],
                               tileCorner[0], tileCorner[1], self.activeClass, self.patchSize)
                
                if len(self.multiDetect) != 0:
                    
                    newTile.multi = self.multiDetect[tileCorner[1]:tileCorner[1] + self.patchSize, tileCorner[0]:tileCorner[0] + self.patchSize]
                    
                    if np.nanmax(newTile.multi) > 0:
                        
                        newTile.isMulti = 1
                        
                    newTile.multiPercent = newTile.get_multi_percent()

                self.tiles[self.tile_index] = newTile
                
                self.tileAdded.emit()
                
                print(self.orderedTileClasses)
                print(self.vegetationLabelIndex)
                
            else:
                
                x = int(selected_coordinates.x())
                y = int(selected_coordinates.y())
                
                lineH = QtWidgets.QGraphicsLineItem(x - 5, y, x + 5, y)
                lineV = QtWidgets.QGraphicsLineItem(x, y + 5, x, y - 5)
                
                self.scene.addItem(lineH)
                self.scene.addItem(lineV)

                newSpectrum = spectrum(self.activeGeoArray[:, y, x], x, y, self.activeClass, len(self.activeGeoArray[:,y,x]))

                self.tiles[self.tile_index] = newSpectrum
                
                self.tileAdded.emit()
            
            self.tile_index += 1
            
        super(geoCanvas, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):

        super(geoCanvas, self).mouseReleaseEvent(event)

        self.viewport().setCursor(QtCore.Qt.CrossCursor)
        
    def keyPressEvent(self, event):
        
        if event.key() == 61:
            
            self.displayed_coordinates_scale = self.displayed_coordinates_scale * 0.8
            self.displayed_coordinates.setScale(self.displayed_coordinates_scale)
            
        if event.key() == 45:
            
            if self.pointSize == 1:
                
                pass
            
            else:
                
                self.displayed_coordinates_scale = self.displayed_coordinates_scale * 1.25
                self.displayed_coordinates.setScale(self.displayed_coordinates_scale)
        
        super(geoCanvas, self).keyPressEvent(event)

    def mouseMoveEvent(self, event):
        
        if not self._empty:
            
            mouse_coords = self.mapToScene(event.x(), event.y())
        
            self.mouse_coordinates = (mouse_coords.x(), mouse_coords.y())
        
            x = int(mouse_coords.x())
            y = int(mouse_coords.y())
                    
            try:
                
                image_value = self.activeGeoArray[0][y][x]

                self.displayed_coordinates.setPlainText("X: %i, Y: %i, Z: %i" % (x, y, image_value))

                self.displayed_coordinates.setPos(x, y)
                
            except Exception as e:
                
                self.displayed_coordinates.setPlainText("X: -, Y: -, Z: -")
                
                self.displayed_coordinates.setPos(x,y)

            super(geoCanvas, self).mouseMoveEvent(event)

    def toggleDragMode(self):

        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:

            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        elif not self.QtImage.pixmap().isNull():

            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    # Convert a PIL image to a QPixmap
    def imageToQPixmap(self, image):

        imgQt = QtGui.QImage(ImageQt.ImageQt(image))

        qPixMap = QtGui.QPixmap.fromImage(imgQt)

        return (qPixMap)

    # Display a 2d array within the viewport (no georeferencing)
    def displayArray(self, arr):
        
        input_array = abs(np.array(arr))
        
        try:
            
            input_array[abs(input_array) > 50000.00] = np.nan
            
        except Exception as e:
            
            print(e)
        
        if len(arr.shape) != 2:

            print("Input must be 2-dimensional array.")

        else:
            
            img = Image.fromarray(((input_array / np.nanmax(input_array)) * 255).astype('uint8'), mode='L')
                
            imgPixMap = self.imageToQPixmap(img)

            self.setQtImage(imgPixMap)

    # Load a geoimage in gdal format into a dictionary
    # Using the dictionary allows for multiple rasters to be loaded at once
    # Note that the gdal object is deleted on load, saving only the file  path and metadata
    def importGeoImage(self, geoImagePath=None):

        if geoImagePath == None:

            geoImagePath = QtWidgets.QFileDialog.getOpenFileName(self, "Import GeoImage", "", ".tif(*.tif)")
            
            geoImagePath = str(geoImagePath)
            
            self.geoImage[geoImagePath] = geoImageReference(geoImagePath)

        else:
            
            self.geoImage[geoImagePath] = geoImageReference(geoImagePath)
        
        self.activeGeoImagePath = geoImagePath
        
        self.activeGeoArray = self.readAs3DArray(geoImagePath)
        
        self.displayGeoImage()
        
    def import_Multidetect(self):
        
        multiImagePath = QtWidgets.QFileDialog.getOpenFileName(self, "Import GeoImage", "", ".tif(*.tif)")
        
        multiImage = gdal.Open(multiImagePath[0])
        
        x = multiImage.RasterXSize
        y = multiImage.RasterYSize
        
        multiArr = multiImage.GetRasterBand(1).ReadAsArray(0,0,x,y).astype("uint8")
        
        self.multiDetect = multiArr
        
        
    #This is a bit redundant but more intuitive
    def changeGeoImage(self, geoImagePath):
        
        self.activeGeoImagePath = geoImagePath
        
        self.activeGeoArray = self.readAs3DArray(geoImagePath)
        
        self.displayGeoImage(band=self.activeBand)
    
    #Extends the DisplayArray function, allowing 3D arrays to be loaded
    def displayGeoImage(self, band=0):
        
        try:
            
            self.displayArray(self.activeGeoArray[int(band)])
        
        except IndexError as error:
            
            self.displayArray(self.activeGeoArray[0])
    
    #Returns key georeferencing information from a GDAL object
    def getGeodetics(self, inputGeoImage):
        
        geoTransform = inputGeoImage.GetGeoTransform()
        
        projection = inputGeoImage.GetProjection()
        
        return(geoTransform, projection)
    
    #Takes in a GDAL object and returns a 3D Array of that object
    def readAs3DArray(self, inRasterPath):

        gimage = gdal.Open(inRasterPath)
        
        bands = gimage.RasterCount
        
        x = gimage.RasterXSize
        y = gimage.RasterYSize
        
        arr = np.zeros((bands, y, x))

        for band in range(bands):
            
            arr[band] = gimage.GetRasterBand(band + 1).ReadAsArray(0,0,x,y)
            
            arr[band] = ((arr[band] - arr[band].min()) / (arr[band].max() - arr[band].min()) * 255).astype("uint8")
        
        return(arr)
    
    def exportTilesToFile(self):
        
        fname = QtWidgets.QFileDialog.getSaveFileName(self, "Save Tiles","",".til(*.til)")
        
        pickle.dump(self.tiles, open(fname[1], 'wb'))
        
    def classifiedToTif(self):
        
        fname = "test.tiff"
        
        im_shape = self.classified_image.shape
        
        outRaster = gdal.GetDriverByName("GTiff").Create(fname,im_shape[1], im_shape[0], 1, gdal.GDT_UInt16)
        
        geoTransform = self.geoImage[self.activeGeoImagePath].geoTransform
        
        outRaster.SetGeoTransform(geoTransform)
                
        srs = self.geoImage[self.activeGeoImagePath].spatialReferenceSystem
        
        outRaster.SetProjection(srs.ExportToWkt())
        
        outRaster.GetRasterBand(1).WriteArray(self.classified_image)
        
        outRaster.FlushCache()
        

    #########################################################
    #
    #         Here be Classification Tools.
    #
    #
    #########################################################
    
    #This currently creates a duplicate data structure
    #of all selected spectra. Maybe change it to simply refer
    #to the class variable which already holds that info
    def spectralToDataframe(self):
        
        data = self.tiles
        
        spectrumHeader = [str(x) for x in range(self.total_bands)]
        spectrumHeader.append("class")
        
        spectrumDf = pd.DataFrame(columns=spectrumHeader)
        spectrumDf["id"] = [x for x in range(len(data))]
        
        #Band iterator
        b = 0
        
        #For each spectrum object
        for i in data:
            
            #Name of classification
            c = data[i].classification
            
            #Array containing an intensity value for each band
            spec = data[i].data
            
            #for each band in the spectrum
            for band in range(len(spec)):
                
                #assign the b'th value of the associated band column to it's respective
                #intensity value
                spectrumDf[str(band)][b] = spec[band]
                spectrumDf["class"][b] = c 
                
            #increase the iterator
            b+=1

        return(spectrumDf)
    
    #generates a dataframe containing the GLCM information of all selected tiles
    def tileToDataFrame(self):
        
        glcm_prop_headers = ["contrast","dissimilarity","homogeneity","asm","energy","correlation", "class", "average","multi"]
        
        tileDf = pd.DataFrame(columns=glcm_prop_headers)
        
        tileDf["id"] = [x for x in range(len(self.tiles))]
        
        b = 0
                    
        for i in self.tiles:
            
            tileDf["class"][b] = self.tiles[i].classification
            tileDf["contrast"][b] = self.tiles[i].glcm_props["contrast"]
            tileDf["dissimilarity"][b] = self.tiles[i].glcm_props["dissimilarity"]
            tileDf["homogeneity"][b] = self.tiles[i].glcm_props["homogeneity"]
            tileDf["asm"][b] = self.tiles[i].glcm_props["asm"]
            tileDf["energy"][b] = self.tiles[i].glcm_props["energy"]
            tileDf["correlation"][b] = self.tiles[i].glcm_props["correlation"]
            tileDf["average"][b] = self.tiles[i].glcm_props["average"]
            
            if len(self.multiDetect) != 0:
                
                tileDf["multi"][b] = self.tiles[i].glcm_props["multi"]
            
            b += 1
        
        return(tileDf)
            
    #Encode the labels from ascii text into integer values for ML functions
    #Assigns a numeric value to each unique label in the column
    def encode_labels(self, df, column):
        
        uniqueLabels = df[column].unique()
        
        encoded_labels = {}
        
        i = 0
        
        for label in uniqueLabels:
            
            encoded_labels[label] = i
            
            i+=1
            
        labels = df
            
        labels["%s_encoded" % column] = [encoded_labels[x] for x in df[column]]
            
        self.labels = encoded_labels
        
        return(labels)
    
    #Convert spectral data into a format useable by Sklearn
    def prepSpectrum(self):

        tileDataFrame = self.spectralToDataframe()
        
        features = np.array(tileDataFrame.drop(["class", "id"], axis=1))
        
        labels = self.encode_labels(tileDataFrame, "class")
            
        labels = np.array(labels["class_encoded"])
            
        return(features, labels)
    
    #Convert selected tiles into format useable by sklearn    
    def prepTiles(self):
        
        tileDataFrame = self.tileToDataFrame()
        
        features = np.array(tileDataFrame.drop(["class","id"], axis=1))
        
        labels = self.encode_labels(tileDataFrame, "class")
        
        labels = np.array(labels["class_encoded"])
        
        return(features, labels)

    def trainAdaBoost(self):
        
        if self.isSpectrum == True:
            
            features, labels = self.prepSpectrum()
            
        else:
            
            features, labels = self.prepTiles()
            
        train_Features, test_Features, train_Labels, test_Labels = train_test_split(features, labels, test_size=0.3)
                
        adaboost = AdaBoostClassifier(n_estimators = 50, learning_rate=1)
        
        model = adaboost.fit(train_Features, train_Labels)

        self.models["adaBoost"] = model
        
        prediction = model.predict(test_Features)
        
        print("---------Training Results----------")
        print("Accuracy", metrics.accuracy_score(test_Labels, prediction))
        print("\n")
        
    def trainRandomForest(self):
        
        if self.isSpectrum == True:
            
            features, labels = self.prepSpectrum()

            train_Features, test_Features, train_Labels, test_Labels = train_test_split(
                features, labels, test_size=0.3)
        
            rfClass = RandomForestClassifier()
        
            model = rfClass.fit(train_Features, train_Labels)

            self.models["randomForest"] = model

            prediction = model.predict(test_Features)
            print("Accuracy", metrics.accuracy_score(test_Labels, prediction))
            
        else:
            
            pass
        

        
    def veggiPy(self):
        
        if self.isSpectrum == True:
            
            pass
        
        else:
            
            features, labels = self.prepTiles()
            
        train_Features, test_Features, train_Labels, test_Labels = train_test_split(features, labels, test_size=0.3)
        
        adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        
        fit_count = 0
        
        model_acc = 0
        
        #usually you would not want to repeatedly train a model (i think) as it would take forever
        #however, we are always working with faily low sample sizes and small datasets, so i see the 
        #worth in trying to optimize the model
        print("Fitting model...")
        while fit_count <= 10:
                    
            model = adaboost.fit(train_Features, train_Labels)
        
            prediction = model.predict(test_Features)
            
            acc = metrics.accuracy_score(test_Labels, prediction)
            
            print("Pass %i Accuracy: %f" % (fit_count, acc))
            
            if acc > model_acc:
                                
                model_acc = acc
                
            fit_count += 1
                
        self.models["adaBoost"] = model
        
        print("Final Accuracy: %f" % acc)
        
        print("Predicting vegetation")
        
        shp = self.activeGeoArray.shape
        arr = self.activeGeoArray
        h = shp[0]
        l = shp[1]
        w = shp[2]
        
        veggiPy_Image = np.zeros((l, w))
        
        arr = np.reshape(arr, (l, w)).astype("uint8")
        
        #should rewrite this to actually use the tile class i wrote specifically for this purpose
        for y in range(0, l - self.patchSize, self.patchSize):
            
            for x in range(0, w - self.patchSize, self.patchSize):
                
                
                activeTile = arr[int(y): int(y) + int(self.patchSize), int(x):int(x) + self.patchSize]
                
                if np.nanmax(activeTile) != 0:
                    
                    activeTileGlcm = greycomatrix(activeTile, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
                    correlation = greycoprops(activeTileGlcm, 'correlation')[0,0]
                    dissimilarity = greycoprops(activeTileGlcm, 'dissimilarity')[0,0]
                    homogeneity = greycoprops(activeTileGlcm, 'homogeneity')[0,0]
                    energy = greycoprops(activeTileGlcm, 'energy')[0,0]
                    contrast = greycoprops(activeTileGlcm, 'contrast')[0,0]
                    asm = greycoprops(activeTileGlcm, 'ASM')[0,0]
                    average = np.average(activeTile)
                    multi = 0
                
                    if np.nanmax(self.multiDetect[int(y): int(y) + int(self.patchSize), int(x):int(x) + self.patchSize]) > 0:
                    
                        percent_zeros = self.multiDetect[int(y): int(y) + int(self.patchSize), int(x):int(x) + self.patchSize]
                    
                        multi = (np.count_nonzero(percent_zeros) / (percent_zeros.shape[1] * percent_zeros.shape[1])) * 255
                    
                    prediction = model.predict([[contrast, dissimilarity, homogeneity, asm, energy, correlation, average, multi]])
                    
                    if int(prediction[0]) == self.vegetationLabelIndex:
                    
                        veg_Prob = model.predict_proba([[contrast, dissimilarity, homogeneity, asm, energy, correlation, average, multi]])
                    
                        veggiPy_Image[int(y): int(y) + int(self.patchSize), int(x):int(x) + self.patchSize] = veg_Prob[0][self.vegetationLabelIndex] * 100
                    
        self.classified_image = veggiPy_Image
        
        self.classifiedToTif()
            
    def classify_all(self, model):
        
        shp = self.activeGeoArray.shape
        arr = self.activeGeoArray
        
        h = shp[0]
        l = shp[1]
        w = shp[2]

        self.classified_image = np.zeros((l,w))

        if self.isSpectrum == True:
            
            arr = np.reshape(arr, (h, l, w)).astype("uint8")
            
            for y in range(l):
            
                for x in range(w):
                
                    self.classified_image[y, x] = self.models[model].predict([arr[:,y,x]])
        
            self.classified_image = classified_image.reshape(l, w)
            
            img = Image.fromarray((self.classified_image / np.nanmax(self.classified_image)) * 255)
            
            img2 = img.convert("L")
            
            img2.save("testBW.png")
        
        else:
                    
            arr = np.reshape(arr, (l, w)).astype("uint8")
            
            for y in range(0, l - self.patchSize, self.patchSize):
                
                for x in range(0, w - self.patchSize, self.patchSize):
                    
                    activeTile = arr[int(y): int(y) + int(self.patchSize), int(x):int(x) + self.patchSize]

                    activeTileGlcm = greycomatrix(activeTile, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
                    
                    correlation = greycoprops(activeTileGlcm, 'correlation')[0,0]
                    dissimilarity = greycoprops(activeTileGlcm, 'dissimilarity')[0,0]
                    homogeneity = greycoprops(activeTileGlcm, 'homogeneity')[0,0]
                    energy = greycoprops(activeTileGlcm, 'energy')[0,0]
                    contrast = greycoprops(activeTileGlcm, 'contrast')[0,0]
                    asm = greycoprops(activeTileGlcm, 'ASM')[0,0]
                    average = np.average(activeTile)
                    
                    if correlation == 1.0:
                        
                        self.classified_image[int(y): int(y) + int(self.patchSize), int(x):int(x) + self.patchSize] = np.nan
                        
                    else:
                                       #broken while multi exists                                         
                        prediction = self.models[model].predict([[contrast, dissimilarity, homogeneity, asm, energy, correlation, average]])

                        self.classified_image[int(y): int(y) + int(self.patchSize), int(x):int(x) + self.patchSize] = prediction[0]
            
            imLabels = [x / len(self.orderedTileClasses) * 255 for x in range(len(self.orderedTileClasses))]
            
            nanval = np.nanmax(self.classified_image) + 1
            
            self.classified_image = np.nan_to_num(x=self.classified_image, nan=nanval)
            
            self.classifiedToTif()
            
            im = Image.fromarray((self.classified_image / np.nanmax(self.classified_image)) * 255)
            
            im = im.convert("L")
            
            im.save("testBW_Tiles.png")
            
            im.show()

#Extracts useful information from geo imagery, stores it, and closes the geoimage file
#This is used to be able to save and access the relevant metadata, without clogging
#up memory. This allows us to load only the files we want into memory when we need them
#by using the filename
class geoImageReference(object):

    def __init__(self, geoImagePath):
        
        gimage = gdal.Open(geoImagePath)
        
        prj = gimage.GetProjection()
        
        gt = gimage.GetGeoTransform()
        
        srs = osr.SpatialReference(wkt=prj)
        
        self.filePath = geoImagePath
        
        self.projectedCoordinateSystem = srs.GetAttrValue('projcs')
        
        self.geoCoordinateSystem = srs.GetAttrValue('geogcs')
        
        self.datum = srs.GetAttrValue('datum')
        
        self.spherioid = srs.GetAttrValue('spheroid')
        
        self.projection = srs.GetAttrValue('projection')
        
        self.x = gt[0]
        
        self.y = gt[3]
        
        self.resolution = gt[1]
        
        self.spatialReferenceSystem = srs
        
        self.geoTransform = gt
        
        self.bands = gimage.RasterCount
        
        gimage = None

#2D array extracted from an image
class tile(object):
    
    def __init__(self, data, x, y, classification, dataSize):
        
        self.data = data.reshape(dataSize, dataSize)
        self.data = self.data.astype(int)
        
        self.x = int(x)
        
        self.y = int(y)
        
        self.shape = data.shape
        
        self.classification = classification
        
        self.cleared = False
        
        self.avg = np.average(self.data)
        
        self.glcm = greycomatrix(self.data.astype("uint8"), distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        
        self.multi = None
        
        self.isMulti = 0
        
        self.multiPercent = 0
        
        contrast = greycoprops(self.glcm, "contrast")[0][0]
        correlation = greycoprops(self.glcm, "correlation")[0][0]
        dissimilarity = greycoprops(self.glcm, "dissimilarity")[0][0]
        energy = greycoprops(self.glcm, "energy")[0][0]
        homogeneity = greycoprops(self.glcm, "homogeneity")[0][0]
        asm = greycoprops(self.glcm, "ASM")[0][0]
        """
        print("\n---------------------------------------\n")
        print("Contrast: %f" % contrast)
        print("Correlation: %f" % correlation)
        print("Dissimilarity: %f" % dissimilarity)
        print("Energy: %f" % energy)
        print("Homogeneity: %f" % homogeneity)
        print("ASM: %f" % asm)
        print("Average: %f" % self.avg)
        print("Min: %f" % np.min(self.data))
        """
        self.glcm_props = {"contrast":contrast, "correlation":correlation, 
                           "dissimilarity":dissimilarity, "energy":energy, 
                           "homogeneity":homogeneity, "asm":asm, "average":self.avg,
                           "multi":self.multiPercent}
        
    def get_multi_percent(self):
            
        num_zeros = np.count_nonzero(self.multi == 0)
            
        num_total = self.shape[1] * self.shape[2]
            
        return((num_zeros / num_total) * 255)
        
    def __str__(self):
        
        return(self.data)
        
class spectrum(object):
    
    def __init__(self, data, x, y, classification, dataSize):
        
        self.data = data
        
        self.x = int(x)
        
        self.y = int(y)
        
        self.classification = classification
        
        self.dataSize = dataSize
        
        self.cleared = False
    
    def __str__(self):
        
        return(self.data)
        
if __name__ == "__main__":
    
    app = QtWidgets.QApplication(argv)
    form = tilePicker_Form()
    form.show()
    app.exec_()
