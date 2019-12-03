export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib

killall $1.out
./start.sh $1.out scheduler &
./start.sh $1.out server 0 &
./start.sh $1.out server 1 &
# ./start.sh $1.out server 2 &
./start.sh $1.out worker 1 &
./start.sh $1.out worker 2 &
./start.sh $1.out worker 0 