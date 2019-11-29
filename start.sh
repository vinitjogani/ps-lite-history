export DMLC_NUM_SERVER=1
export DMLC_NUM_WORKER=1
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8002
export DMLC_ROLE=$1

if [ "$DMLC_ROLE" = 'worker' ]; then
    export HEAPPROFILE=./W0
fi

if [ "$DMLC_ROLE" = 'server' ]; then
    export HEAPPROFILE=./S0
fi

./main.out 