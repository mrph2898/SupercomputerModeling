#!/bin/bash -x
mpisubmit.bg -n $1 -w 00:15:00 --stdout mc_master_slave.$(jobid).$1.$2.out mc_master_slave $2
