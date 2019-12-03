#include <iostream>
#include <stdio.h>

#include "ps-history/reader.h"
#include "ps-history/dataset.h"

const int MAX_LINE_LEN = 1024*10;
const char* MNIST_IMAGES = "mnist.scale";
const int IMAGE_HEADER_SIZE = 16;
const int LABEL_HEADER_SIZE = 8;
const int NUM_FEATURES = 784;

Dataset<uint8_t> *read_mnist(int skip, int take) {


void read_mnist(char* fname, int *rowidx, int *colidx, double *vals,
    double *y , int *nnz_local, int *m_local, int* nnzArray) {
    
    FILE *fp = fopen(MNIST_IMAGES, "r");
    char buffer[MAX_LINE_LEN];

    int i=0, cols=0, rows=0;
    int *labels = malloc(sizeof(int) * 60000);
    
    int *ptr, *ind;
    double *val;

    while (true)
    {
        if (fgets(buffer, MAX_LINE_LEN, fp) == NULL) break;
        char *str_label = strtok(buffer, " ");
        labels[i] = strtol(str_label, NULL, 10);
        
        char *value, *index;
        while(true) {
            index = strtok(NULL,":");
            value = strtok(NULL," ");
            if(value == NULL) break;
            
            colidx[count_colidx] = (int) strtol(index, NULL, 10);
            nnzArray[cols-1]++;

            val =  strtod(value, &endptr);
            vals[count_colidx]=val;
            count_colidx++;
            i++;
        }
        count_rowidx++;
        rowidx[count_rowidx]=i+1;
        //printf("%d\n",count_rowidx );

        y[count_rowidx-1]=x;
    }
    
    *nnz_local = count_colidx;
    *m_local = count_rowidx;
    fclose(images_fp);
}

