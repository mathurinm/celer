# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause
from cython cimport floating

cdef double fmax(double, double) nogil
cdef double fmin(double, double) nogil
cdef double ST(double, double) nogil
cdef double primal_value(double, int, double *, int, double *) nogil
cdef double dual_value(int, double, double, double *, double *) nogil

cdef floating fused_dot(int *, floating *, int *, floating *, int *) nogil
cdef floating fused_asum(int *, floating *, int *) nogil
cdef void fused_axpy(int *, floating *, floating *, int *, floating *, int *) nogil
cdef floating fused_nrm2(int * , floating *, int *) nogil
cdef void fused_copy(int *, floating *, int *, floating *, int *) nogil
cdef void fused_scal(int *, floating *, floating *, int *) nogil

# cdef double fused_posv
