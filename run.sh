export LD_LIBRARY_PATH=lib
killall main.out
./start.sh scheduler &
./start.sh server &
./start.sh worker