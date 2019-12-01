#pragma once

#include <iostream>
#include <stdio.h>

template <class T>
class Dataset {

private:

    int num_records, num_features;

public:
    
    T *features;
    uint8_t *labels;

    Dataset(int num_records, int num_features) {
        this->num_records = num_records;
        this->num_features = num_features;
        this->features = (T*)malloc(sizeof(T)*num_records*num_features);
        this->labels = (uint8_t*)malloc(sizeof(uint8_t)*num_records);
    }

    T *getFeatures(int i);
    uint8_t getLabel(int i);

};
