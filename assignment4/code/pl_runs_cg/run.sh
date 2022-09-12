#!/bin/bash -x
mpisubmit.pl -p $1 -w 00:15 --stdout mc_master_slave.$(jobid).$1.$2.out mc_master_slave $2
