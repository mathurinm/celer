cdef double fmax(double, double) nogil
cdef double fmin(double, double) nogil
cdef double ST(double, double) nogil
cdef double primal_value(double, int, double *, int, double *) nogil
cdef double dual_value(int, double, double, double *, double *) nogil
cdef int GEOM_GROWTH = 0
cdef int LIN_GROWTH = 1
