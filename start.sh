bin=$1
shift

export DMLC_NUM_SERVER=2
export DMLC_NUM_WORKER=3
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8002
export DMLC_ROLE=$1

if [ "$DMLC_ROLE" = 'worker' ]; then
    export HEAPPROFILE=./W$2
fi

if [ "$DMLC_ROLE" = 'server' ]; then
    export HEAPPROFILE=./S$2
fi

./build/${bin} 