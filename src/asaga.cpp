#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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
#define StdKeys np::vector<Key, std::vector<Key> > 

const int NUM_RECORDS = 6000;
const float MINIBATCH_FRAC = 0.1;
const int NUM_ITERS = 20;
const double LR = 0.001;
StdKeys weight_keys(10*785);

int ComputeUpdate(StdMatrix &grad, StdMatrix &W, MnistDataset *db, int r, double lr) {
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
    grad = lr * np::outer_prod(pred, x);

    return correct;
}

void RunWorker(int rank, int num_workers) {
    for (int i = 0; i < weight_keys.size(); i++) weight_keys(i) = i; 
    np::scalar_vector<Key> offset (785*10,785*10);

    KVWorker<double> kv(0, rank);
    int per_worker = (NUM_RECORDS * 1.0 / num_workers);
    int skip = per_worker*rank;
    MnistDataset *db = read_mnist(skip, per_worker);
    
    int num_records = db->getNumRecords();
    int minibatch_size = num_records * MINIBATCH_FRAC;
    std::cout << num_records << " records loaded\n";

    StdMatrix W (10, 785);
    StdMatrix Sum (10, 785);
    StdKeys sum_keys = weight_keys + (offset * (NUM_RECORDS+1));

    int last = 0;
    for (int t = 0; t < NUM_ITERS; t++) {
        kv.Pull(weight_keys.data(), &W.data());
        kv.Pull(sum_keys.data(), &Sum.data());

        StdMatrix update (10, 785);
        update *= 0;
        int correct = 0;

        std::cout << rank << "Going in\n";
        if (rank == 0) sleep(1);

        for (int i = 0; i < minibatch_size; i++) {
            
            int r = (rand() * 1.0) / RAND_MAX * (num_records - 1);

            StdMatrix history (10, 785);
            StdKeys history_keys = weight_keys + (offset * (r+skip+1));
            if (rank != 0) {
                last = kv.Pull(history_keys.data(), &history.data());
            }

            StdMatrix grad;
            correct += ComputeUpdate(grad, W, db, r, LR / minibatch_size);
            
            if (rank != 0) {
                kv.Wait(last);            
                history = (grad - history);
                update -= history;
                kv.Push(history_keys.data(), history.data());
                kv.Push(sum_keys.data(), history.data());
            }
        }
        std::cout << rank << "Coming out\n";
        if (rank != 0) {
            W += (update - MINIBATCH_FRAC*Sum);
            kv.Wait(kv.Push(weight_keys.data(), update.data()));
        }
        else {
            double accuracy = correct * 1.0 / minibatch_size;
            std::cout << "Iter[" << t <<  "]. Validation accuracy: " << accuracy * 100 << std::endl;
        }
    }
    
    // kv.Wait(last);
}
