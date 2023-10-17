Install Amber22 by INTEL
-----------------------

Typical error:

.. code:: bash

   src/mpi4py.MPI.c:198:12: fatal error: longintrepr.h: No such file or directory
   198 | #include "longintrepr.h"

How to solve it?


.. code:: bash

   Adding 'set(MINICONDA_VERSION py310_23.5.2-0)' line 91
   i.e. forcing the use of python3.10 instead of python3.11, before:
   set(MINICONDA_INSTALLER_FILENAME "Miniconda${PYTHON_MAJOR_RELEASE}-...
   in amber22_src/cmake/UseMiniconda.cmake seems to solve the problem.

Refer: http://archive.ambermd.org/202307/0083.html
