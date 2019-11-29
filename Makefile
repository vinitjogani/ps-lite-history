setup:
	cp -r ../ps-lite/deps/lib lib 
	cp -r ../ps-lite/include include  
	cp ../ps-lite/build/libps.a .

main.o: main.cpp 
	g++ -Iinclude -c main.cpp -std=c++11

main.out: main.o 
	g++ main.o libps.a -lprotobuf-lite -lpthread -lzmq -Llib -o main.out
