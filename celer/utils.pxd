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

cdef floating fdot(int *, floating *, int *, floating *, int *) nogil
cdef floating fasum(int *, floating *, int *) nogil
cdef void faxpy(int *, floating *, floating *, int *, floating *, int *) nogil
cdef floating fnrm2(int * , floating *, int *) nogil
cdef void fcopy(int *, floating *, int *, floating *, int *) nogil
cdef void fscal(int *, floating *, floating *, int *) nogil

cdef void fposv(char *, int *, int *, floating *,
                     int *, floating *, int *, int *) nogil
