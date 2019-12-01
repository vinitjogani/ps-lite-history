main.out: main.o dataset.o reader.o include/ps-history/dataset.h include/ps-history/reader.h 
	g++ main.o dataset.o reader.o libps.a -lprotobuf-lite -lpthread -lzmq -Llib -o main.out -std=c++11 -g

dataset.o: dataset.cpp include/ps-history/dataset.h 
	g++ -Iinclude -c $< -std=c++11 -g

reader.o: reader.cpp include/ps-history/reader.h 
	g++ -Iinclude -c $< -std=c++11 -g

main.o: main.cpp 
	g++ -Iinclude -c main.cpp -std=c++11 -g

clean:
	rm *.o *.out

setup:
	cp -r ../ps-lite/deps/lib lib 
	cp -r ../ps-lite/include include  
	cp ../ps-lite/build/libps.a .