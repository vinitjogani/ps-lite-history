INCLUDE  = include/ps-history
SRC = src
BUILD = build

HEADERS = $(INCLUDE)/dataset.h $(INCLUDE)/worker.h $(INCLUDE)/reader.h 
OBJECTS = $(BUILD)/main.o $(BUILD)/reader.o 

all: $(BUILD)/asgd.out $(BUILD)/asaga.out

$(BUILD)/asgd.out: $(BUILD)/asgd.o $(OBJECTS) $(HEADERS)
	g++ $(BUILD)/asgd.o $(OBJECTS) lib/libps.a -lprotobuf-lite -lpthread -lzmq -Llib -o $@ -std=c++11 -g

$(BUILD)/asaga.out: $(BUILD)/asaga.o $(OBJECTS) $(HEADERS)
	g++ $(BUILD)/asaga.o $(OBJECTS) lib/libps.a -lprotobuf-lite -lpthread -lzmq -Llib -o $@ -std=c++11 -g

$(BUILD)/reader.o: $(SRC)/reader.cpp $(INCLUDE)/reader.h 
	g++ -Iinclude -c $< -std=c++11 -o $@

$(BUILD)/asgd.o: $(SRC)/asgd.cpp 
	g++ -Iinclude -c $< -std=c++11 -o $@ 

$(BUILD)/asaga.o: $(SRC)/asaga.cpp 
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