INCLUDE  = include/ps-history
SRC = src
BUILD = build

HEADERS = $(INCLUDE)/dataset.h $(INCLUDE)/worker.h $(INCLUDE)/reader.h 
OBJECTS = $(BUILD)/main.o $(BUILD)/reader.o 

$(BUILD)/asgd.out: $(BUILD)/asgd.o $(OBJECTS) $(HEADERS)
	g++ $(BUILD)/asgd.o $(OBJECTS) lib/libps.a -lprotobuf-lite -lpthread -lzmq -Llib -o $@ -std=c++11 -g

$(BUILD)/reader.o: $(SRC)/reader.cpp $(INCLUDE)/reader.h 
	g++ -Iinclude -c $< -std=c++11 -o $@

$(BUILD)/asgd.o: $(SRC)/asgd.cpp 
	g++ -Iinclude -c $< -std=c++11 -o $@ 
	
$(BUILD)/main.o: $(SRC)/main.cpp 
	g++ -Iinclude -c $< -std=c++11 -o $@ 

load:
	module load boost

clean:
	rm $(BUILD)/*.o $(BUILD)/*.out

setup:
	cp -r ../ps-lite/deps/lib lib 
	cp -r ../ps-lite/include include  
	cp ../ps-lite/build/libps.a lib