# Implements the Array ADT using array capabilities of the ctypes module.
import ctypes


class Array:
    # Creates an array with size elements.
    def __init__(self, size):
        assert size > 0, "Array size must be > 0"
        self._size = size
        # Create the array structure using the ctypes module.
        PyArrayType = ctypes.py_object * size
        self._elements = PyArrayType()
        # Initialize each element.
        self.clear(None)

    def __len__(self):
        # Returns the size of the array.
        return self._size

    def __getitem__(self, index):
        # Gets the contents of the index element.
        #assert index >= 0 and index < len(self), "Array subscript out of range"
        return self._elements[index]

    def __setitem__(self, index, value):
        # Puts the value in the array element at index position.
        assert index >= 0 and index < len(self), "Array subscript out of range"
        self._elements[index] = value

    def __add__(self, rhsArray):
        assert self._size == rhsArray._size, "Array can't be added"
        newArray = Array(self._size)
        for i in range(self._size):
            newArray[i] = self[i] + rhsArray[i]
        return newArray

    def clear(self, value):
        # Clears the array by setting each element to the given value.
        for i in range(len(self)):
            self._elements[i] = value

    def __iter__(self):
        # Returns the array's iterator for traversing the elements.
        return _ArrayIterator(self._elements)

        
        

    def __repr__(self):
        # Returns the string reputation of an object
        s = '[ '
        for x in self._elements:
            s = s + str(x) + ', '
        s = s[:-2] + ' ]'
        return s


# An iterator for the Array ADT.
class _ArrayIterator:

    def __init__(self, theArray):
        self._arrayRef = theArray
        self._curNdx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curNdx < len(self._arrayRef):
            entry = self._arrayRef[self._curNdx]
            self._curNdx += 1
            return entry
        else:
            raise StopIteration

# Implementation of the Array2D ADT using an array of arrays.


class Array2D:
    # Creates a 2-D array of size numRows x numCols.
    def __init__(self, numRows, numCols):
        # Create a 1-D array to store an array reference for each row.
        self._theRows = Array(numRows)
        # Create the 1-D arrays for each row of the 2-D array.
        for i in range(numRows):
            self._theRows[i] = Array(numCols)
    # Returns the number of rows in the 2-D array.

    def numRows(self):
        return len(self._theRows)
    # Returns the number of columns in the 2-D array.

    def numCols(self):
        return len(self._theRows[0])
    # Clears the array by setting every element to the given value.

    def clear(self, value):
        for row in range(self.numRows()):
            self._theRows[row].clear(value)
    # Gets the contents of the element at position [i, j]

    def __getitem__(self, ndxTuple):
        assert len(ndxTuple) == 2, "Invalid number of array subscripts."
        row = ndxTuple[0]
        col = ndxTuple[1]
        assert row >= 0 and row < self.numRows() \
            and col >= 0 and col < self.numCols(), \
            "Array subscript out of range."
        the1dArray = self._theRows[row]
        return the1dArray[col]
    # Sets the contents of the element at position [i,j] to value.

    def __setitem__(self, ndxTuple, value):
        assert len(ndxTuple) == 2, "Invalid number of array subscripts."
        row = ndxTuple[0]
        col = ndxTuple[1]
        assert row >= 0 and row < self.numRows() \
            and col >= 0 and col < self.numCols(), \
            "Array subscript out of range."
        the1dArray = self._theRows[row]
        the1dArray[col] = value
    # Returns the string reputation of an object

    def __repr__(self):
        s = '['
        for r in range(self.numRows()):
            for c in self._theRows[r]:
                s = s + str(c) + ', '
            s = s[:-2] + ' \n '
        s = s[:-3] + ' ]'
        return s


class Matrix(Array2D):
    # Creates a matrix of size numRows x numCols initialized to 0.
    def __init__(self, numRows, numCols):
        super().__init__(numRows, numCols)

    # Scales the matrix by the given scalar.
    def scaleBy(self, scalar):
        for r in range(self.numRows()):
            for c in range(self.numCols()):
                self[r, c] *= scalar
        return self

    # Creates and returns a new matrix that is the transpose of this matrix.
    def transpose(self):
        newMatrix = Matrix(self.numCols(), self.numRows())
        for r in range(self.numRows()):
            for c in range(self.numCols()):
                newMatrix[c, r] = self[r, c]
        return newMatrix

    # Creates and returns a new matrix that results from matrix addition.
    def __add__(self, rhsMatrix):
        assert rhsMatrix.numRows() == self.numRows() and \
            rhsMatrix.numCols() == self.numCols(), \
            "Matrix sizes not compatible for the add operation."
        # Create the new matrix.
        newMatrix = Matrix(self.numRows(), self.numCols())
        # Add the corresponding elements in the two matrices.
        for r in range(self.numRows()):
            for c in range(self.numCols()):
                newMatrix[r, c] = self[r, c] + rhsMatrix[r, c]
        return newMatrix

    # Creates and returns a new matrix that results from matrix subtraction.
    def __sub__(self, rhsMatrix):
        """ assert rhsMatrix.numRows() == self.numRows() and \
            rhsMatrix.numCols() == self.numCols(), \
            "Matrix sizes not compatible for the substract operation." """
        # Create the new matrix.
        newMatrix = Matrix(self.numRows(), self.numCols())
        # Add the corresponding elements in the two matrices.
        for r in range(self.numRows()):
            for c in range(self.numCols()):
                newMatrix[r, c] = self[r, c] - rhsMatrix[r, c]
        return newMatrix
    # Creates and returns a new matrix resulting from matrix multiplication.

    def __mul__(self, rhsMatrix):
        """ assert self.numCols() == rhsMatrix.numRows(), \
            "Matrix size not compatible for the multiple operation" """
        newMatrix = Matrix(self.numRows(), rhsMatrix.numCols())
        newMatrix.clear(0)
        for r in range(newMatrix.numRows()):
            for c in range(newMatrix.numCols()):
                for i in range(self.numCols()):
                    newMatrix[r, c] += (self[r, i] * rhsMatrix[i, c])
        return newMatrix

    # Practice Examination Solution
    def det(self):
        # for Matrix 2x2 or 3x3 only
        assert self.numCols() == self.numRows() == 2 or self.numCols() == self.numRows() == 3, \
            "Matrix 2x2 or 3x3 only for the matrix determination."
        if self.numCols() == 2:
            deter = self[0, 0]*self[1, 1] - self[0, 1]*self[1, 0]
        else:
            deter = self[0, 0]*self[1, 1]*self[2, 2] + self[0, 1]*self[1, 2]*self[2, 0] + \
                self[0, 2]*self[1, 0]*self[2, 1] - self[0, 2]*self[1, 1]*self[2, 0] - \
                self[0, 1]*self[1, 0]*self[2, 2] - self[0, 0]*self[1, 2]*self[2, 1]
        return deter

    def inverse(self):
        # for Matrix 2x2 only
        assert self.numCols() == self.numRows() == 2 and self.det() != 0, \
            "Matrix 2x2 and det is not zero can use this inverse matrix operation."
        newMatrix = Matrix(self.numRows(), self.numCols())
        newMatrix[0, 0] = self[1, 1]
        newMatrix[0, 1] = self[0, 1] * -1
        newMatrix[1, 0] = self[1, 0] * -1
        newMatrix[1, 1] = self[0, 0]
        # scale = 1/self.det()
        # newMatrix.scaleBy(scale)
        newMatrix.scaleBy(1/self.det())
        return newMatrix

    def submatrix(self, row, col):
        newMatrix = Matrix(self.numRows() - 1, self.numCols() - 1)
        subrow, subcol = 0, 0
        for xrow in [r for r in range(self.numRows()) if r is not row]:
            for xcol in [c for c in range(self.numCols()) if c is not col]:
                newMatrix[subrow, subcol] = self[xrow, xcol]
                subcol += 1
            subcol = 0
            subrow += 1
        return newMatrix

    def slice(self,r_start,r_stop,c_start,c_stop):
        row = r_stop+1-r_start
        col = c_stop+1-c_start
        s_row, s_col = 0, 0
        newMatrix = Matrix(row,col)
        for xrow in range(r_start , r_stop+1):
            for xcol in range(c_start , c_stop+1):
                newMatrix[s_row, s_col] = self[xrow, xcol]
                s_col += 1
            s_col = 0
            s_row += 1
        return newMatrix

    def dot_product(matrix, other):
        dp = 0
        for i in range(len(matrix)):
            dp += (matrix[i]*other[i])
        return dp


    def ref(self,row):
        s=""
        for c in self._theRows[row]:
            s = s + str(c) + ', '
        s = s[:-2]
        return(s)

    def zeros(row=1,col=1):
        NewMatrix = Matrix(row,col)
        NewMatrix.clear(0)
        return NewMatrix

    def shape(self):
        return self.numRows() ,self.numCols() 

    def argmax(self, axis=None):
        if axis == 0:
            argmax_array = Matrix(1,self.numCols())
            for i in range(self.numCols()):
                if self[0,i] >= self[1,i]:
                    argmax_array[0,i] = 0
                else:
                    argmax_array[0,i] = 1
        else:
            print("Wait for me")
        return argmax_array