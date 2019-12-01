#include <iostream>
#include <stdio.h>

#include "ps-history/reader.h"
#include "ps-history/dataset.h"

const char* MNIST_IMAGES = "train-images.idx3-ubyte";
const char* MNIST_LABELS = "train-labels.idx1-ubyte";
const int IMAGE_HEADER_SIZE = 16;
const int LABEL_HEADER_SIZE = 8;
const int NUM_FEATURES = 784;

Dataset<uint8_t> *read_mnist(int skip, int take) {
    auto *db = new Dataset<uint8_t>(take, NUM_FEATURES);

    FILE* images_fp = fopen(MNIST_IMAGES, "r");
    if (images_fp == NULL) {
        fprintf(stderr, "Could not find images file.\n");
        exit(1);
    }
    fseek(images_fp, IMAGE_HEADER_SIZE + (NUM_FEATURES * skip), SEEK_SET);
    fread(db->features, sizeof(uint8_t), NUM_FEATURES*take, images_fp);
    fclose(images_fp);

    FILE* labels_fp = fopen(MNIST_LABELS, "r");
    if (labels_fp == NULL) {
        fprintf(stderr, "Could not find images file.\n");
        exit(1);
    }
    fseek(labels_fp, LABEL_HEADER_SIZE + skip, SEEK_SET);
    int num_records = fread(db->labels, sizeof(uint8_t), take, labels_fp);
    fclose(labels_fp);
    db->setNumRecords(num_records);

    return db;
}