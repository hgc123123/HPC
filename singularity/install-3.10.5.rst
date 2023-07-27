
.. code:: bash

   sudo yum install -y \
       libseccomp-devel \
       glib2-devel \
       squashfs-tools \
       cryptsetup \
       runc
       
    
   export VERSION=1.19.5 OS=linux ARCH=amd64 
   wget -O /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz   https://dl.google.com/go/go${VERSION}.${OS}-${ARCH}.tar.gz
   tar -C /usr/local -xzf /tmp/go${VERSION}.${OS}-${ARCH}.tar.gz
   echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
   source ~/.bashrc

   curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin

   echo 'export PATH=$PATH:$(go env GOPATH)/bin' >> ~/.bashrc
   source ~/.bashrc

   git clone --recurse-submodules https://github.com/sylabs/singularity.git
   cd singularity

   git checkout --recurse-submodules v3.10.5

   ./mconfig
   make -C builddir
   sudo make -C builddir install
