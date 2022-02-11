#!/bin/sh

prefixdir=`realpath ..`

aclocal
automake -a -c --foreign
autoconf
./configure --prefix=$prefixdir
make clean
make
make install
