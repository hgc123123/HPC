Install Amber18 by GNU
-----------------------


.. code:: bash

   module load cmake/3.26.3-gcc-8.5.0 openmpi/main-gcc-8.5.0
   cmake $AMBER_PREFIX/amber18 -DCMAKE_INSTALL_PREFIX=$AMBER_PREFIX/Amber18 -DCOMPILER=GNU -DMPI=TRUE -DCUDA=FALSE -DINSTALL_TESTS=TRUE -DDOWNLOAD_MINICONDA=TRUE -DBUILD_PYTHON=FALSE  -DBUILD_PERL=FALSE
   make install
