
#pragma once 

#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

const int MAX_LINE_LEN = 1024*10;

class SparseDataset {
private:

    std::vector<int> _rows;
    std::vector<int> _cols;
    std::vector<double> _vals;
    std::vector<int> labels;

public:

    int num_records;
    static SparseDataset *from(const std::string fname, int skip, int take);

    double *vals();
    int *rows();
    int *cols();
    int label(int o);
};

int SparseDataset::label(int i) {
    return labels[i];
}

double *SparseDataset::vals() {
    return _vals.data();
}

int *SparseDataset::rows() {
    return _rows.data();
}

int *SparseDataset::cols() {
    return _cols.data();
}

SparseDataset *SparseDataset::from(const std::string fname, int skip, int take) {

    SparseDataset *db = new SparseDataset();
    
    FILE *fp = fopen(fname.c_str(), "r");
    char buffer[MAX_LINE_LEN];

    while (take > 0)
    {
        if (fgets(buffer, MAX_LINE_LEN, fp) == NULL) break;
        if (skip > 0) {
            skip --;
            continue;
        }

        char *str_label = strtok(buffer, " ");
        db->labels.push_back(strtol(str_label, NULL, 10));
        db->_rows.push_back(db->_cols.size());
        
        char *value, *index;
        while(true) {
            index = strtok(NULL,":");
            value = strtok(NULL," ");
            if(value == NULL) break;
            
            db->_cols.push_back(strtol(index, NULL, 10));
            db->_vals.push_back(strtod(value, NULL));
        }

        take--;
    }

    db->num_records = db->_rows.size();
    db->_rows.push_back(db->_cols.size());

    return db;
}

