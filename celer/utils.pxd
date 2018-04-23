# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

cdef double fmax(double, double) nogil
cdef double fmin(double, double) nogil
cdef double ST(double, double) nogil
cdef double primal_value(double, int, double *, int, double *) nogil
cdef double dual_value(int, double, double, double *, double *) nogil
