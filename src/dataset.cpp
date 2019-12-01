#include <iostream>
#include "ps-history/dataset.h"

template <class T>
T *Dataset<T>::getFeatures(int i) {
    return this->features + (i * this.num_features);
}

template <class T>
uint8_t Dataset<T>::getLabel(int i) {
    return this->labels[i];
}