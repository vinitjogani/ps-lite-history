
#pragma once 
#include <iostream>
#include <stdio.h>

template <class T>
class Dataset {
private:

    int num_records;
    int num_features;

public:
    T *features;
    uint8_t *labels;

    Dataset(int num_records, int num_features);

    T *getFeatures(int i);
    uint8_t getLabel(int i);
    
    void setNumRecords(int num_records);
    int getNumRecords();

};


template <class T>
Dataset<T>::Dataset(int num_records, int num_features) {
    this->num_records = num_records;
    this->num_features = num_features;
    this->features = (T*)malloc(sizeof(T)*num_records*num_features);
    this->labels = (uint8_t*)malloc(sizeof(uint8_t)*num_records);
}

template <class T>
T *Dataset<T>::getFeatures(int i) {
    return this->features + (i * this.num_features);
}

template <class T>
uint8_t Dataset<T>::getLabel(int i) {
    return this->labels[i];
}

template <class T>
void Dataset<T>::setNumRecords(int num_records) {
    if (num_records < this->num_records) 
        this->num_records = num_records;
}

template <class T>
int Dataset<T>::getNumRecords() {
    return this->num_records;
}