#! /bin/bash
# export NNODES=32
# export HOST_FILE="/share/work/sh/work/70B/InternEvo/scripts/hostfile8.txt"
export NNODES=64
export HOST_FILE="/share/work/sh/work/soft/test/hostfiledo64.txt"

# /opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} /bin/bash /share/work/sh/work/soft/test/sw.sh
cd /share/work/huangting/InternEvo
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} ps -ef |grep torch |grep root
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} ps -ef |grep python |grep root

# /opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
#    docker load -i /share/work/huangting/mxc500-torch2.1-py310-mc2.24.0.5-ubuntu22.04-amd64.container

# /opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
#    docker stop ht_test_0927

# /opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
#    docker run -itd \
#                 -u root  \
#                 --device=/dev/dri/card6 \
#                 --device=/dev/dri/renderD134 \
#                 --device=/dev/dri/card7 \
#                 --device=/dev/dri/renderD135 \
#                 --device=/dev/mxcd  \
#                 --device=/dev/infiniband \
#                 --privileged=true \
#                 --group-add=video \
#                 --name ht_test_0927 \
#                 --security-opt seccomp=unconfined \
#                 --security-opt apparmor=unconfined \
#                 --shm-size 160gb \
#                 --ulimit memlock=-1 \
#                 -v /share:/share  \
#                 -v /data:/data \
#                 --network host \
#                 mxc500-torch2.1-py310:mc2.24.0.5-ubuntu22.04-amd64 /bin/bash

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker exec ht_test_0927 /bin/bash -c "/share/work/sh/work/soft/test/kill.sh"

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} -output-filename logs/llama2_7B \
   docker exec ht_test_0927 /bin/bash -c "bash /share/work/huangting/InternEvo/train_70B.sh ${NNODES} ${HOST_FILE}" #2>&1|tee ht.log