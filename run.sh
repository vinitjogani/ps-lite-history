export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib

killall main.out
./start.sh scheduler &
./start.sh server 0 &
./start.sh server 1 &
./start.sh server 2 &
./start.sh worker 0 &
./start.sh worker 1 &
./start.sh worker 2