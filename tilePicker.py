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
        defaultClasses = ["Tree", "Snow", "Water", "Asphalt", "Soil", "Rock"]
        
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
            
            #Remove this when testing is done
            self.spectrumSelectRadio.setCheckable(True)
            
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
        self.activeClass = "Tree"
        
        #Machine Learning stuff
        self.labels = None
        self.testFeatures = None
        self.models = {}

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
 
                self.tiles[self.tile_index] = newTile
                
                self.tileAdded.emit()
                
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

        mouse_coords = self.mapToScene(event.x(), event.y())

        self.mouse_coordinates = (mouse_coords.x(), mouse_coords.y())

        self.displayed_coordinates.setPlainText("X: %i, Y: %i" % (mouse_coords.x(), mouse_coords.y()))

        self.displayed_coordinates.setPos(mouse_coords.x(), mouse_coords.y())

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
        print(self.activeGeoArray.shape)

        self.displayGeoImage()
    
    #This is a bit redundant but more intuitive
    def changeGeoImage(self, geoImagePath):
        
        self.activeGeoImagePath = geoImagePath
        
        self.activeGeoArray = self.readAs3DArray(geoImagePath)
        
        self.displayGeoImage(band=self.activeBand)
    
    def displayGeoImage(self, band=0):
        
        try:
            
            self.displayArray(self.activeGeoArray[int(band)])
        
        except IndexError as error:
            
            self.displayArray(self.activeGeoArray[0])
            
    def getGeodetics(self, inputGeoImage):
        
        geoTransform = inputGeoImage.GetGeoTransform()
        
        projection = inputGeoImage.GetProjection()
        
        return(geoTransform, projection)
    
    #Takes in a GDAL object and returns a 3D Array of that object
    def readAs3DArray(self, inRasterPath):
        
        gimage = gdal.Open(inRasterPath)
        
        bands = gimage.RasterCount
        print(bands)
        
        x = gimage.RasterXSize
        y = gimage.RasterYSize
        
        arr = np.zeros((bands, y, x))

        for band in range(bands):
            
            arr[band] = gimage.GetRasterBand(band + 1).ReadAsArray(0,0,x,y)
            
            arr[band] = ((arr[band] - arr[band].min()) / (arr[band].max() - arr[band].min()) * 256).astype("uint8")
        
        return(arr)
    
    def exportTilesToFile(self):
        
        fname = QtWidgets.QFileDialog.getSaveFileName(self, "Save Tiles","",".til(*.til)")
        
        pickle.dump(self.tiles, open(fname[1], 'wb'))
        
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
    
    def tileToDataFrame(self):
        
        data = self.tiles
        
        glcm_prop_headers = ["contrast","dissimilarity","homogeneity","asm","energy","correlation", "class"]
        
        tileDf = pd.DataFrame(columns=glcm_prop_headers)
        
        tileDf["id"] = [x for x in range(len(data))]
        
        b = 0
        
        for i in data:
            
            tileDf["class"][b] = data[i].classification
            tileDf["contrast"][b] = data[i].glcm_props["contrast"]
            tileDf["dissimilarity"][b] = data[i].glcm_props["dissimilarity"]
            tileDf["homogeneity"][b] = data[i].glcm_props["homogeneity"]
            tileDf["asm"][b] = data[i].glcm_props["asm"]
            tileDf["energy"][b] = data[i].glcm_props["energy"]
            tileDf["correlation"][b] = data[i].glcm_props["correlation"]
            
            b += 1
        

        return(tileDf)
            
    #Encode the labels from ascii text into integer values for ML functions
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
        
        print("--------Test Features-------")
        print(test_Features)
        print("=========Test Labels=========")
        print(test_Labels)
        
        adaboost = AdaBoostClassifier(n_estimators = 50, learning_rate=1)
        
        model = adaboost.fit(train_Features, train_Labels)

        self.models["adaBoost"] = model
        
        prediction = model.predict(test_Features)

        print("Accuracy", metrics.accuracy_score(test_Labels, prediction))
        print(self.activeGeoArray)
        
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

    def classify_all(self, model):
        
        shp = self.activeGeoArray.shape
        arr = self.activeGeoArray
        
        h = shp[0]
        l = shp[1]
        w = shp[2]

        classified_image = np.zeros((l,w))

        if self.isSpectrum == True:
            
            arr = np.reshape(arr, (h, l, w)).astype("uint8")
            for y in range(l):
            
                for x in range(w):
                
                    classified_image[y, x] = self.models[model].predict([arr[:,y,x]])
        
            classified_image = classified_image.reshape(l, w)
            
            img = Image.fromarray((classified_image / np.nanmax(classified_image)) * 255)
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
                                                                            
                    prediction = self.models[model].predict([[contrast, dissimilarity, homogeneity, asm, energy, correlation]])

                    classified_image[int(y): int(y) + int(self.patchSize), int(x):int(x) + self.patchSize] = prediction[0]
                    
                    if x%100 == 0:
                        
                        print(prediction)
                        
            im = Image.fromarray((classified_image / np.nanmax(classified_image)) * 255)
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
        
        #self.norm_vals = ((self.data - self.data.min()) / (self.data.max() - self.data.min()) * 256).astype("uint8")
        
        self.glcm = greycomatrix(self.data.astype("uint8"), distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        
        contrast = greycoprops(self.glcm, "contrast")[0][0]
        correlation = greycoprops(self.glcm, "correlation")[0][0]
        dissimilarity = greycoprops(self.glcm, "dissimilarity")[0][0]
        energy = greycoprops(self.glcm, "energy")[0][0]
        homogeneity = greycoprops(self.glcm, "homogeneity")[0][0]
        asm = greycoprops(self.glcm, "ASM")[0][0]
        
        self.glcm_props = {"contrast":contrast, "correlation":correlation, 
                           "dissimilarity":dissimilarity, "energy":energy, 
                           "homogeneity":homogeneity, "asm":asm}
        
        """print("-------------- Tile Greyscale Values ---------------")
        print(self.norm_vals)
        print("--------------- Greycoprops Vales ---------------")
        print(greycoprops(self.glcm, "contrast")[0][0])
        print(greycoprops(self.glcm, "correlation")[0][0])
        print(greycoprops(self.glcm, "dissimilarity")[0][0])
        print(greycoprops(self.glcm, "energy")[0][0])
        print(greycoprops(self.glcm, "homogeneity")[0][0])
        print(greycoprops(self.glcm, "ASM")[0][0])"""
        
        #Testing to ensure the data being used for the tiles is as expected
        #print(data.reshape(10,10))
        #test = Image.fromarray((data.reshape(10,10) / np.nanmax(data)) * 255)
        #test = test.convert("L")
        #test.save("test.png")

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
