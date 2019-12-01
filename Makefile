INCLUDE  = include/ps-history
SRC = src
BUILD = build

$(BUILD)/main.out: $(BUILD)/main.o $(BUILD)/dataset.o $(BUILD)/reader.o $(INCLUDE)/dataset.h $(INCLUDE)/reader.h 
	g++ $*.o lib/libps.a -lprotobuf-lite -lpthread -lzmq -Llib -o $@ -std=c++11 -g

$(BUILD)/dataset.o: $(SRC)/dataset.cpp $(INCLUDE)/dataset.h 
	g++ -Iinclude -c $< -std=c++11 -o $@

$(BUILD)/reader.o: $(SRC)/reader.cpp $(INCLUDE)/reader.h 
	g++ -Iinclude -c $< -std=c++11 -o $@

$(BUILD)/main.o: $(SRC)/main.cpp 
	g++ -Iinclude -c $< -std=c++11 -o $@

clean:
	rm $(BUILD)/*.o $(BUILD)/*.out

setup:
	cp -r ../ps-lite/deps/lib lib 
	cp -r ../ps-lite/include include  
	cp ../ps-lite/build/libps.a lib