cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport ceil
import gudhi as gd
import numpy as np
cimport numpy as cnp
cnp.import_array()


dtype_float = np.float32


cdef class EccBackbone:
    cdef Py_ssize_t w, grid_h, grid_w, num_cells, steps
    cdef float t_min, t_max, resolution, lower_bound, impulse
    cdef const float[:] dimension

    def __init__(self, t_const=True, size=[28, 28], interval=[0.02, 0.28], steps=32, impulse=5):
        """
        Args:
            t_const (bool, optional): Use T-construction. If False, V-construction will be used. Defaults to True.
            size (list, optional): [Height, Width] of image. Defaults to [28, 28].
            interval (list, optional): Minimum and maximum value of interval to consider. Defaults to [0.02, 0.28].
            steps (int, optional):  Number of discretized points. Defaults to 32.
            impulse (float, optional): Value used as approximation of dirac delta. Defaults to 5.
        """
        assert size[0] > 1 and size[1] > 1, "Size of image should be larger than [1, 1]"
        assert interval[1] > interval[0], "End point of interval must be larger than starting point of interval"
        assert steps > 1, "Number of steps should be larger than 1"
        self.w = size[1]
        self.grid_h, self.grid_w = [2*i + 1 if t_const else 2*i - 1 for i in size]    # size of the cubical complex
        self.num_cells = self.grid_h * self.grid_w
        self.dimension = self._set_dimension()
        self.t_min, self.t_max = interval
        self.steps = steps
        self.resolution = (self.t_max - self.t_min) / (self.steps - 1)
        self.lower_bound = self.t_min - self.resolution    # lower bound for skipping gradient calculation in backpropagation step
        self.impulse = impulse

    # V-construction
    @cython.boundscheck(False)      # turn off bounds-checking for entire function
    @cython.wraparound(False)       # turn off negative index wrapping for entire function
    @cython.cdivision(True)         # turn off checking for division by zero
    @cython.initializedcheck(False) # turn off initialization check
    def cal_ecc_vtx(self, object x, bint backprop):
        """_summary_

        Args:
            x (anything convertible to a numpy.ndarray): Array of shape [B, C, H, W].
            backprop (bint): Whether or not the input requires gradient calculation.

        Returns:
            _type_: _description_
        """
        cdef Py_ssize_t batch_size, num_channels, b, c, cell, t, pix, i, num_max
        cdef float filt, dim
        cdef Py_ssize_t *neighbor_vtx
        cdef Py_ssize_t *vtx
        cdef float[4] vtx_filt

        batch_size, num_channels, h, w = x.shape
        
        ecc = np.zeros((batch_size, num_channels, self.steps), dtype=dtype_float)
        cdef float[:, :, :] ecc_view = ecc

        grad_local = np.zeros((batch_size, num_channels, h*w, self.steps), dtype=dtype_float) if backprop else None
        cdef float[:, :, :, :] grad_local_view = grad_local

        cdef float[:] filtration

        for b in range(batch_size):         # iterate over batch
            for c in range(num_channels):   # iterate over channel
                cub_cpx = gd.CubicalComplex(vertices=x[b, c])   # V-contruction
                filtration = cub_cpx.all_cells().astype(dtype_float).flatten()
                for cell in range(self.num_cells):  # iterate over all cells in cubical complex
                    filt = filtration[cell]
                    if filt > self.t_max:
                        continue 
                    dim = self.dimension[cell]
                    t = max(<Py_ssize_t>ceil((filt - self.t_min) / self.resolution), 0)
                    ecc_view[b, c, t] += (-1.)**dim

                    # calculation of gradient only for inputs that require gradient
                    if backprop:
                        if filt < self.lower_bound:                             # skip bc. gradient is 0 for simplices with filtration value under lower bound
                            continue
                        # vertex
                        if dim == 0:
                            pix = self._vtx2pix(cell)                           # index of the corresponding pixel in flattened original image
                            grad_local_view[b, c, pix, t] -= self.impulse
                        # edge
                        elif dim == 1:
                            neighbor_vtx = self._find_neighbor_vtx(cell, dim)   # neighbor_vtx points at a C array containing index of 2 neighbor vertices
                            for i in range(2):
                                vtx_filt[i] = filtration[neighbor_vtx[i]]       # filtration value of neighbor vertices
                            
                            if vtx_filt[0] > vtx_filt[1]:
                                pix = self._vtx2pix(neighbor_vtx[0])
                                grad_local_view[b, c, pix, t] += self.impulse
                            elif vtx_filt[0] < vtx_filt[1]:
                                pix = self._vtx2pix(neighbor_vtx[1])
                                grad_local_view[b, c, pix, t] += self.impulse
                            else:                                               # split gradient when the neighboring vertices have the same filtration value
                                for i in range(2):
                                    pix = self._vtx2pix(neighbor_vtx[i])
                                    grad_local_view[b, c, pix, t] += (self.impulse / 2.)
                            free(neighbor_vtx)
                        # square
                        else:
                            neighbor_vtx = self._find_neighbor_vtx(cell, dim)   # neighbor_vtx points at a C array containing index of 4 neighbor vertices
                            for i in range(4):
                                vtx_filt[i] = filtration[neighbor_vtx[i]]       # filtration value of neighbor vertices
                            
                            vtx = self._find_max_vtx(vtx_filt, neighbor_vtx, 4, &num_max)   # vtx points at a C array containing index of vertices that contribute to constructing the cell
                            for i in range(num_max):
                                pix = self._vtx2pix(vtx[i])
                                grad_local_view[b, c, pix, t] -= (self.impulse / num_max)
                            free(vtx)
                            free(neighbor_vtx)
                # cumsum
                for i in range(self.steps - 1):
                    ecc_view[b, c, i+1] += ecc_view[b, c, i]
        return ecc, grad_local

    @cython.cdivision(True)     # turn off checking for division by zero
    cdef inline Py_ssize_t _vtx2pix(self, Py_ssize_t vtx):
        """Given the index of a vertex, this function returns the index of the corresponding pixel.
        Used for V-constructed cubical complexes.

        Args:
            vtx (Py_ssize_t): Index of vertex.

        Returns:
            Py_ssize_t: Index of corresponding pixel.
        """
        return (vtx // (2*self.grid_w))*self.w + (vtx % self.grid_w)/2

    @cython.cdivision(True)     # turn off checking for division by zero
    cdef inline Py_ssize_t* _find_neighbor_vtx(self, Py_ssize_t cell, float dim):
        """Returns the indices of a cell's neighboring vertices.
        Used for V-constructed cubical complexes.
        Do not use for cells that are already vertices.

        Args:
            cell (Py_ssize_t): Index of cell.
            dim (float): Dimension of cell.

        Returns:
            Py_ssize_t pointer: C array containing index of neighboring squares.
        """
        cdef Py_ssize_t row_num
        cdef Py_ssize_t *neighbor_vtx = <Py_ssize_t *> malloc(<Py_ssize_t>dim * 2 * sizeof(Py_ssize_t)) # assign size 2 array for edges and size 4 array for squares
        # edge
        if dim == 1:
            row_num = cell // self.grid_w
            if row_num % 2 == 0:    # even row
                neighbor_vtx[:] = [cell-1, cell+1]
            else:                   # odd row
                neighbor_vtx[:] = [cell-self.grid_w, cell+self.grid_w]
        # square
        else:
            neighbor_vtx[:] = [cell-self.grid_w-1, cell-self.grid_w+1, cell+self.grid_w-1, cell+self.grid_w+1]
        return neighbor_vtx

    cdef inline Py_ssize_t* _find_max_vtx(self, float *vtx_filt, Py_ssize_t *neighbor_vtx, Py_ssize_t arr_size, Py_ssize_t *num_max):
        """

        """
        cdef float max_val = vtx_filt[0]
        cdef Py_ssize_t j = 0, count = 0

        # find maximum filtration value
        for i in range(1, arr_size):
            if vtx_filt[i] > max_val:
                max_val = vtx_filt[i]
        
        # count how many times max_val occurs
        for i in range(arr_size):
            if vtx_filt[i] == max_val:
                count += 1

        cdef Py_ssize_t *vtx = <Py_ssize_t *> malloc(count * sizeof(Py_ssize_t))
        
        # store the index of vertices that have maximum filtration value
        for i in range(arr_size):
            if vtx_filt[i] == max_val:
                vtx[j] = neighbor_vtx[i]
                j += 1

        # number of max_val occurences
        num_max[0] = count
        return vtx

    # T-construction
    @cython.boundscheck(False)      # turn off bounds-checking for entire function
    @cython.wraparound(False)       # turn off negative index wrapping for entire function
    @cython.cdivision(True)         # turn off checking for division by zero
    @cython.initializedcheck(False) # turn off initialization check
    def cal_ecc_topdim(self, object x, bint backprop):
        """_summary_

        Args:
            x (anything convertible to a numpy.ndarray): Array of shape [B, C, H, W].
            backprop (bint): Whether or not the input requires gradient calculation

        Returns:
            _type_: _description_
        """
        cdef Py_ssize_t batch_size, num_channels, b, c, cell, t, pix, i, num_neighbors, num_min
        cdef float filt, dim
        cdef Py_ssize_t *neighbor_sq
        cdef Py_ssize_t *sq
        cdef float[4] sq_filt

        batch_size, num_channels, h, w = x.shape

        ecc = np.zeros((batch_size, num_channels, self.steps), dtype=dtype_float)
        cdef float[:, :, :] ecc_view = ecc

        grad_local = np.zeros((batch_size, num_channels, h*w, self.steps), dtype=dtype_float) if backprop else None
        cdef float[:, :, :, :] grad_local_view = grad_local

        cdef float[:] filtration

        for b in range(batch_size):         # iterate over batch
            for c in range(num_channels):   # iterate over channel
                cub_cpx = gd.CubicalComplex(top_dimensional_cells=x[b, c])  # T-construction
                filtration = cub_cpx.all_cells().astype(dtype_float).flatten()
                for cell in range(self.num_cells): # iterate over all cells in cubical complex
                    filt = filtration[cell]
                    if filt > self.t_max:
                        continue
                    dim = self.dimension[cell]
                    t = max(<Py_ssize_t>ceil((filt - self.t_min) / self.resolution), 0)
                    ecc_view[b, c, t] += (-1.)**dim

                    # calculation of gradient only for inputs that require gradient
                    if backprop:
                        if filt < self.lower_bound:                         # skip bc. gradient is 0 for simplices with filtration value under lower bound
                            continue
                        # square
                        if dim == 2:
                            pix = self._sq2pix(cell)                        # index of the corresponding pixel in flattened original image
                            grad_local_view[b, c, pix, t] -= self.impulse
                        # edge
                        elif dim == 1:
                            neighbor_sq = self._find_neighbor_sq(cell, dim, &num_neighbors) # neighbor_sq points at a C array containing index of neighbor squares
                            if num_neighbors == 1:                          # there is 1 neighbor square
                                pix = self._sq2pix(neighbor_sq[0])
                                grad_local_view[b, c, pix, t] += self.impulse
                            else:                                           # there are 2 neighbor squares
                                for i in range(2):
                                    sq_filt[i] = filtration[neighbor_sq[i]] # filtration value of neighbor squares

                                if sq_filt[0] > sq_filt[1]:    
                                    pix = self._sq2pix(neighbor_sq[1])
                                    grad_local_view[b, c, pix, t] += self.impulse
                                elif sq_filt[0] < sq_filt[1]:
                                    pix = self._sq2pix(neighbor_sq[0])
                                    grad_local_view[b, c, pix, t] += self.impulse
                                else:                                       # split gradient when the neighboring squares have the same filtration value
                                    for i in range(2):
                                        pix = self._sq2pix(neighbor_sq[i])
                                        grad_local_view[b, c, pix, t] += (self.impulse / 2.)
                            free(neighbor_sq)
                        # vertex
                        else:
                            neighbor_sq = self._find_neighbor_sq(cell, dim, &num_neighbors) # neighbor_sq points at a C array containing index of neighbor squares
                            if num_neighbors == 1:                          # there is 1 neighbor square
                                pix = self._sq2pix(neighbor_sq[0])
                                grad_local_view[b, c, pix, t] -= self.impulse
                            elif num_neighbors == 2:                        # there are 2 neighbor squares
                                for i in range(2):
                                    sq_filt[i] = filtration[neighbor_sq[i]] # filtration value of neighbor squares

                                if sq_filt[0] > sq_filt[1]:    
                                    pix = self._sq2pix(neighbor_sq[1])
                                    grad_local_view[b, c, pix, t] -= self.impulse
                                elif sq_filt[0] < sq_filt[1]:
                                    pix = self._sq2pix(neighbor_sq[0])
                                    grad_local_view[b, c, pix, t] -= self.impulse
                                else:                                       # split gradient when the neighboring squares have the same filtration value
                                    for i in range(2):
                                        pix = self._sq2pix(neighbor_sq[i])
                                        grad_local_view[b, c, pix, t] -= (self.impulse / 2.)
                            else:                                           # there are 4 neighbor squares
                                for i in range(4):
                                    sq_filt[i] = filtration[neighbor_sq[i]] # filtration value of neighbor squares
                                
                                sq = self._find_min_sq(sq_filt, neighbor_sq, 4, &num_min)
                                for i in range(num_min):
                                    pix = self._sq2pix(sq[i])
                                    grad_local_view[b, c, pix, t] -= (self.impulse / num_min)
                                free(sq)
                            free(neighbor_sq)
                # cumsum
                for i in range(self.steps - 1):
                    ecc_view[b, c, i+1] += ecc_view[b, c, i]
        return ecc, grad_local

    @cython.cdivision(True)     # turn off checking for division by zero
    cdef inline Py_ssize_t _sq2pix(self, Py_ssize_t sq):
        """Given the index of a square, this function returns the index of the corresponding pixel.
        Used for T-constructed cubical complexes.

        Args:
            vtx (Py_ssize_t): Index of square.

        Returns:
            Py_ssize_t: Index of corresponding pixel.
        """
        return (sq // (2*self.grid_w))*self.w + (sq % self.grid_w)/2

    @cython.cdivision(True)     # turn off checking for division by zero
    cdef inline Py_ssize_t* _find_neighbor_sq(self, Py_ssize_t cell, float dim, Py_ssize_t* num_neighbors):
        """Returns the indices of a cell's neighboring squares.
        Used for T-constructed cubical complexes.
        Do not use for cells that are already squares.

        Args:
            cell (Py_ssize_t): Index of cell.
            dim (float): Dimension of cell.
            num_neighors(Py_ssize_t pointer): Number of neighboring squares of cell.

        Returns:
            Py_ssize_t pointer: C array containing index of neighboring squares.
        """
        cdef Py_ssize_t row_num
        cdef Py_ssize_t *neighbor_sq = <Py_ssize_t *> malloc(<Py_ssize_t>(dim -2) * -2 * sizeof(Py_ssize_t))    # assign size 2 array for edges and size 4 array for vertices
        # edge
        if dim == 1:
            row_num = cell // self.grid_w
            # even row
            if row_num % 2 == 0:
                if row_num == 0:                        # top row
                    neighbor_sq[0] = cell + self.grid_w
                    num_neighbors[0] = 1
                elif row_num == self.grid_h - 1:        # bottom row
                    neighbor_sq[0] = cell - self.grid_w
                    num_neighbors[0] = 1
                else:
                    neighbor_sq[:] = [cell-self.grid_w, cell+self.grid_w]
                    num_neighbors[0] = 2
            # odd row
            else:
                col_num = cell % self.grid_w
                if col_num == 0:                        # left-most column
                    neighbor_sq[0] = cell + 1
                    num_neighbors[0] = 1
                elif col_num == self.grid_w - 1:        # right-most column
                    neighbor_sq[0] = cell - 1
                    num_neighbors[0] = 1
                else:
                    neighbor_sq[:] = [cell-1, cell+1]
                    num_neighbors[0] = 2
        # vertex
        else:
            row_num = cell // self.grid_w
            col_num = cell % self.grid_w
            # top row
            if row_num == 0:
                if cell == 0:                              # top left corner
                    neighbor_sq[0] = cell + self.grid_w + 1
                    num_neighbors[0] = 1
                elif cell == self.grid_w - 1:              # top right corner
                    neighbor_sq[0] = cell + self.grid_w - 1
                    num_neighbors[0] = 1
                else:
                    neighbor_sq[:2] = [cell+self.grid_w-1, cell+self.grid_w+1]
                    num_neighbors[0] = 2
            # bottom row
            elif row_num == self.grid_h - 1:
                if cell == self.grid_w * (self.grid_h-1):  # bottom left corner
                    neighbor_sq[0] = cell - self.grid_w + 1
                    num_neighbors[0] = 1
                elif cell == self.grid_w*self.grid_h - 1:  # bottom right corner
                    neighbor_sq[0] = cell - self.grid_w - 1
                    num_neighbors[0] = 1
                else:
                    neighbor_sq[:2] = [cell-self.grid_w-1, cell-self.grid_w+1]
                    num_neighbors[0] = 2
            # left-most column
            elif col_num == 0:                      
                neighbor_sq[:2] = [cell+1-self.grid_w, cell+1+self.grid_w]
                num_neighbors[0] = 2
            # right-most column
            elif col_num == self.grid_w - 1:
                neighbor_sq[:2] = [cell-1-self.grid_w, cell-1+self.grid_w]
                num_neighbors[0] = 2
            else:
                neighbor_sq[:] = [cell-self.grid_w-1, cell-self.grid_w+1, cell+self.grid_w-1, cell+self.grid_w+1]
                num_neighbors[0] = 4
        return neighbor_sq

    cdef inline Py_ssize_t* _find_min_sq(self, float *sq_filt, Py_ssize_t *neighbor_sq, Py_ssize_t arr_size, Py_ssize_t *num_min):
        """

        """
        cdef float min_val = sq_filt[0]
        cdef Py_ssize_t j = 0, count = 0

        # find minimum filtration value
        for i in range(1, arr_size):
            if sq_filt[i] < min_val:
                min_val = sq_filt[i]
        
        # count how many times min_val occurs
        for i in range(arr_size):
            if sq_filt[i] == min_val:
                count += 1

        cdef Py_ssize_t *sq = <Py_ssize_t *> malloc(count * sizeof(Py_ssize_t))
        
        # store the index of squares that have minimum filtration value
        for i in range(arr_size):
            if sq_filt[i] == min_val:
                sq[j] = neighbor_sq[i]
                j += 1

        # number of min_val occurences
        num_min[0] = count
        return sq

    cdef _set_dimension(self):
        """
        Sets dimension for all cubes in the cubical complex. Dimensions of vertice, edge, square are 0, 1, 2 respectively. Even rows consist of (vertice, edge, vertice, edge, ..., vertice, edge, vertice) and odd rows consist of (edge, square, edge, square, ..., edge, square, edge).

        Returns:
            _type_: _description_
        """
        dimension = np.zeros([self.grid_h, self.grid_w], dtype=dtype_float)
        dimension[[i for i in range(self.grid_h) if i % 2 == 1], :] += 1
        dimension[:, [i for i in range(self.grid_w) if i % 2 == 1]] += 1
        dimension.setflags(write=False)
        return dimension.flatten()