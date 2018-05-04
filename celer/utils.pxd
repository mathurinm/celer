# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause
from cython cimport floating

# cdef floating fmax(floating, floating) nogil
# cdef floating fmin(floating, floating) nogil
cdef floating ST(floating, floating) nogil
cdef floating primal_value(floating, int, floating *, int, floating *) nogil
cdef floating dual_value(int, floating, floating, floating *, floating *) nogil

cdef floating fused_dot(int *, floating *, int *, floating *, int *) nogil
cdef floating fused_asum(int *, floating *, int *) nogil
cdef void fused_axpy(int *, floating *, floating *, int *, floating *, int *) nogil
cdef floating fused_nrm2(int * , floating *, int *) nogil
cdef void fused_copy(int *, floating *, int *, floating *, int *) nogil
cdef void fused_scal(int *, floating *, floating *, int *) nogil

cdef void fused_posv(char *, int *, int *, floating *,
                     int *, floating *, int *, int *) nogil
