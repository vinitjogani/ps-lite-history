#ifndef _READER_H
#define _READER_H

#include <iostream>
#include "dataset.h"

Dataset<uint8_t> *read_mnist(int skip, int take);

#endif