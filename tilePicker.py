# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:14:27 2020

@author: nick.viner
"""
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageQt
from sys import argv
import tilePicker_ui
import gdal
import numpy as np
import osr
from os.path import basename

class tilePicker_Form(QtWidgets.QMainWindow, tilePicker_ui.Ui_imageTilePicker):
    
    tileSizeSignal = QtCore.pyqtSignal(int)
    
    def __init__(self, parent = None):
        
        super().__init__()
        
        self.setupUi(self)
        
        #Set up the layout for interacting with and viewing geoImages
        self.canvas = geoCanvas()
        self.canvasView.addWidget(self.canvas)
        
        #Set up the layout for interacting with loaded image files
        self.loadedImagesTable = tableWidget()
        self.loadedImagesTableLayout.addWidget(self.loadedImagesTable)
        self.loadedImagesTable.setColumnCount(1)
        self.loadedImagesTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        #Set up the layout for interacting with selected tiles
        self.selectedTilesTable = tableWidget()
        self.selectedTilesTableLayout.addWidget(self.selectedTilesTable)
        self.selectedTilesTable.setColumnCount(1)
        self.selectedTilesTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        
        # Containers for GUI variables
        self.tileSize = self.tileSizeEdit.text()
        
        #Signals
        self.tileSizeEdit.textChanged.connect(self.tileSizeChanged)
        self.actionLoad_Images.triggered.connect(self.loadImage)
        self.loadedImagesTable.itemDoubleClicked.connect(self.changeActiveImage)
        self.tileSelectRadio.toggled.connect(self.tileSelectRadioUpdate)
        self.spectrumSelectRadio.toggled.connect(self.spectrumSelectRadioUpdate)
        self.bandSelectCombo.currentTextChanged.connect(self.updateActiveBand)
        self.classNameCombo.currentTextChanged.connect(self.updateActiveClass)
        self.canvas.tileAdded.connect(self.updateSelectedTiles)
        
        #Add default object classes
        defaultClasses = ["Tree", "Snow", "Water", "Asphalt", "Soil", "Rock"]
        
        for i in defaultClasses:
            
            self.classNameCombo.addItem(i)
    
    def updateSelectedTiles(self):
        
        tileInfo = "%i, %i, %s, %s" % (self.canvas.tiles[self.canvas.tile_index].x, 
                                   self.canvas.tiles[self.canvas.tile_index].y,
                                   self.canvas.tiles[self.canvas.tile_index].classification,
                                   basename(self.canvas.activeGeoImagePath))
        
        self.selectedTilesTable.addToNext(tileInfo)
        
    def tileSelectRadioUpdate(self):
        
        #Prevent user from using multiple selection types when choosing tiles
        if len(self.canvas.tiles) > 0:
            
            pass
        
        else:
        
            self.tileSizeEdit.setReadOnly(False)
            
            self.spectrumSelectRadio.setChecked(False)
            
            self.canvas.isSpectrum = False
        
    def spectrumSelectRadioUpdate(self):
        
        #Prevent user from using multiple selection types when choosing tiles
        if len(self.canvas.tiles) > 0:
            
            pass
        
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
        
        for f in range(1, bands + 1):
            
            self.bandSelectCombo.addItem(str(f))

class tableWidget(QtWidgets.QTableWidget):
    
    def __init__(self):
        
        super(tableWidget, self).__init__()
        
        self.totalRows = 0
        
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
        self.itemDoubleClicked.connect(self.deleteCell)
        
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        
        
        
    def deleteCell(self):
        
        cleared = QtWidgets.QTableWidgetItem("Cleared")
        print(self.currentItem().text())
        self.setItem(self.currentItem().row(), self.currentItem().column(), cleared)
        
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
    
    patchSizeChanged = QtCore.pyqtSignal(int)
    
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
        self.patchSize = 10
        self.isSpectrum = False
        self.activeTileShape = QtWidgets.QGraphicsRectItem(0,0, self.patchSize, self.patchSize)
        self.tiles = {}
        self.tile_index = 0

        # Coordinate values for the mouse cursor
        self.mouse_coordinates = None
        self.selected_coordinates = None

        # Graphical coordinate indicators
        self.displayed_coordinates = QtWidgets.QGraphicsTextItem()
        self.displayed_coordinates.setTransformOriginPoint(self.displayed_coordinates.boundingRect().topLeft())
        self.displayed_coordinates_font = self.displayed_coordinates.font()
        self.displayed_coordinates_font.setPointSize(20)
        self.displayed_coordinates.setFont(self.displayed_coordinates_font)
        self.displayed_coordinates_scale = 1

        # Add the initialized image and text to the graphics view
        self.scene.addItem(self._QtImage)
        self.scene.addItem(self.displayed_coordinates)
        
        #Active geo image information
        self.activeGeoImagePath = None
        self.activeGeoArray = None
        self.activeBand = None
        
        self.activeClass = "Tree"

        
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
            
            if self.isSpectrum == False:
                
                tileCorner = (int(selected_coordinates.x() - (self.patchSize / 2)), int(selected_coordinates.y() - (self.patchSize / 2)))
            
                rect = QtWidgets.QGraphicsRectItem(0,0, self.patchSize, self.patchSize)
            
                rect.setPos(tileCorner[0], tileCorner[1])
                
                self.scene.addItem(rect)
                
               
                if self.isSpectrum == False:
                    
                    dataType = "tile"
                    
                else:
                    
                    dataType = "spectrum"
                    
                newTile = tile(self.activeGeoArray[tileCorner[1]:tileCorner[1] + self.patchSize, tileCorner[0]:tileCorner[0] + self.patchSize],
                               tileCorner[0], tileCorner[1], self.activeClass, dataType)
 
                
                self.tiles[self.tile_index] = newTile
                
                self.tileAdded.emit()
                
            else:
                
                xy = (selected_coordinates.x(), selected_coordinates.y())
                
                lineH = QtWidgets.QGraphicsLineItem(xy[0] - 10, xy[1], xy[0] + 10, xy[1])
                lineV = QtWidgets.QGraphicsLineItem(xy[0], xy[1] + 10, xy[0], xy[1] - 10)
                
                self.scene.addItem(lineH)
                self.scene.addItem(lineV)
                
                arr = activeGeoArray[:,xy[1], xy[0]]
            
            self.tile_index += 1
            
        super(geoCanvas, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):

        super(geoCanvas, self).mouseReleaseEvent(event)

        self.viewport().setCursor(QtCore.Qt.CrossCursor)

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
            
            pass
        
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
    
    #This is a bit redundant but more intuitive
    def changeGeoImage(self, geoImagePath):
        
        self.activeGeoImagePath = geoImagePath
        
        self.activeGeoArray = self.readAs3DArray(geoImagePath)
        
        self.displayGeoImage()
    
    def displayGeoImage(self, band=0):

        self.displayArray(self.activeGeoArray[band])
        
    def getGeodetics(self, inputGeoImage):
        
        geoTransform = inputGeoImage.GetGeoTransform()
        
        projection = inputGeoImage.GetProjection()
        
        return(geoTransform, projection)
    
    #Takes in a GDAL object and returns a 3D Array of that object
    #This should only be used if the image in question has more than one band as it can potentially use a lot of memory
    def readAs3DArray(self, inRasterPath):
        
        gimage = gdal.Open(inRasterPath)
        
        bands = gimage.RasterCount
        
        x = gimage.RasterXSize
        y = gimage.RasterYSize
        
        arr = np.zeros((bands, y, x))

        for band in range(bands):
            
            arr[band] = gimage.GetRasterBand(band + 1).ReadAsArray(0,0,x,y)
            
        return(arr)

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
        
        gImage = None
        
class tile(object):
    
    def __init__(self, data, x, y, classification, dataType):
        
        self.data = data
        
        self.x = x
        
        self.y = y
        
        self.shape = data.shape
        
        self.classification = classification
        
        self.dataType = dataType
        
    def __str__(self):
        
        return(self.data)
        
if __name__ == "__main__":
    
    app = QtWidgets.QApplication(argv)
    form = tilePicker_Form()
    form.show()
    app.exec_()
