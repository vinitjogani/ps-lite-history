#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "ps/ps.h"
#include "ps-history/reader.h"
#include "ps-history/dataset.h"
#include "ps-history/worker.h"

using namespace ps;
namespace np = boost::numeric::ublas;

#define MnistDataset Dataset<uint8_t>
#define StdMatrix np::matrix<double, np::row_major, std::vector<double> > 

const int NUM_RECORDS = 60000;
const float MINIBATCH_FRAC = 0.1;
const int NUM_ITERS = 20;
const double LR = 0.001;
std::vector<Key> weight_keys(10*785);

int ComputeUpdate(StdMatrix &update, StdMatrix &W, MnistDataset *db, int r, double lr) {
    uint8_t *x_arr = db->getFeatures(r);
    int true_label = db->getLabel(r);

    np::vector<uint8_t> x(785);
    std::copy(&x_arr[0], &x_arr[784], x.data().begin());
    x(784) = 255;

    np::vector<double> pred = np::prod(W, x/255.0);
    double max_ = np::norm_inf(pred); // for stability
    for (int c = 0; c < 10; c++) pred(c) = std::exp(pred(c) - max_);
    pred /= np::sum(pred);
    
    int correct = (true_label == np::index_norm_inf(pred));
    pred(true_label) -= 1.0;
    update -= lr * np::outer_prod(pred, x);

    return correct;
}

void RunWorker(int rank, int num_workers) {
    for (int i = 0; i < weight_keys.size(); i++) weight_keys[i] = i; 
    
    KVWorker<double> kv(0, rank);
    int per_worker = (NUM_RECORDS * 1.0 / num_workers);
    MnistDataset *db = read_mnist(per_worker*rank, per_worker);
    
    int num_records = db->getNumRecords();
    int minibatch_size = num_records * MINIBATCH_FRAC;
    std::cout << num_records << " records loaded\n";

    StdMatrix W (10, 785);
    W *= 0;

    for (int t = 0; t < NUM_ITERS; t++) {
        kv.Pull(weight_keys, &W.data());

        StdMatrix update (10, 785);
        update *= 0;
        int correct = 0;

        for (int i = 0; i < minibatch_size; i++) {
            int r = (rand() * 1.0) / RAND_MAX * (num_records - 1);
            correct += ComputeUpdate(update, W, db, r, LR / minibatch_size);
        }

        if (rank != 0) {
            W += update;
            kv.Push(weight_keys, update.data());
        }
        else {
            double accuracy = correct * 1.0 / minibatch_size;
            std::cout << "Iter[" << t <<  "]. Validation accuracy: " << accuracy * 100 << std::endl;
        }
    }
}
