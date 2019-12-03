#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <unistd.h>

#include "ps/ps.h"
#include "ps-history/reader.h"
#include "ps-history/sparse_dataset.h"
#include "ps-history/worker.h"

using namespace ps;

const std::string DATA_PATH = "mnist.scale";
const int NUM_RECORDS = 10000;
const float MINIBATCH_FRAC = 0.3;
const int NUM_ITERS = 50;
const double LR = 0.01;
const int NUM_FEATURES = 784;
const int GRAD_DIM = NUM_FEATURES+1;
const int BOUND = 2;

double ComputeUpdate(double *update, double *W, SparseDataset *db, 
    std::vector<int> &minibatch, double lr, std::vector<Key> &sum_keys, 
    int history_offset, KVWorker<double> &kv, int push) {

    int m = minibatch.size(), k = NUM_FEATURES;
    int ptrB[m], ptrE[m];
    for (int i = 0; i < m; i++) {
        ptrB[i] = db->rows()[minibatch[i]];
        ptrE[i] = db->rows()[minibatch[i]+1];
    }

    double alpha = 1, beta = 0;
    char trans = 'N';
    char descrA[6] = {'G', '\0', '\0', 'C', '\0', '\0'};
    double pred[m];
    memset(pred, 0, sizeof(double)*m);
    
    mkl_dcsrmv (&trans, &m, &k, &alpha, descrA, db->vals() + ptrB[0], 
        db->cols() + ptrB[0], ptrB, ptrE, W, &beta, pred);

    int r, nz, idx;
    double error = 0;
    for (int i = 0; i < m; i++) {
        r = minibatch[i];
        nz = db->rows()[r+1] - db->rows()[r];
        idx = db->rows()[r];

        std::vector<Key> history_keys(GRAD_DIM);
        int ho = history_offset + r*GRAD_DIM;
        std::iota(std::begin(history_keys), std::end(history_keys), ho);
        std::vector<double> history;
        kv.Wait(kv.Pull(history_keys, &history));

        pred[i] += W[NUM_FEATURES] - db->label(r);
        error += (pred[i] * pred[i]);
        if (!push) continue;

        cblas_daxpyi(nz, -lr*pred[i], db->vals() + idx, db->cols() + idx, history.data());
        history[NUM_FEATURES] -= lr*pred[i];
        // history = lr(old - new)

        vdAdd(GRAD_DIM, history.data(), update, update);
        // update += -lr(new - old) == lr(old - new)

        cblas_dscal(GRAD_DIM, -1, history.data(), 1);
        // history = lr(new - old)
        kv.Wait(kv.Push(history_keys, history));
        kv.Wait(kv.Push(sum_keys, history));
    }

    return error / m;
}

void RunWorker(int rank, int num_workers) {
    std::vector<Key> weight_keys(GRAD_DIM);
    std::iota(std::begin(weight_keys), std::end(weight_keys), 0);

    std::vector<Key> my_rank = {(Key)(GRAD_DIM + rank)};
    std::vector<Key> rank_keys(num_workers);
    std::iota(std::begin(rank_keys), std::end(rank_keys), GRAD_DIM);
    
    KVWorker<double> kv(0, rank);
    int per_worker = (NUM_RECORDS * 1.0 / num_workers);
    int skip = per_worker*rank;
    SparseDataset *db = SparseDataset::from(DATA_PATH, skip, per_worker);
    
    int sum_offset = GRAD_DIM + num_workers;
    std::vector<Key> sum_keys(GRAD_DIM);
    std::iota(std::begin(sum_keys), std::end(sum_keys), sum_offset);

    int history_offset = sum_offset + (skip+1)*GRAD_DIM;
    int num_records = db->num_records, minibatch_size = num_records * MINIBATCH_FRAC;
    double error = 0, lr = LR / minibatch_size;
    std::cout << num_records << " records loaded\n";

    std::vector<double> W (GRAD_DIM);
    std::vector<double> update (GRAD_DIM);
    std::vector<double> progress(num_workers);

    for (int t = 0; t < NUM_ITERS; t++) {
        memset(update.data(), 0, sizeof(double) * (GRAD_DIM));
        kv.Pull(weight_keys, &W);

        std::vector<double> avg_vector;
        kv.Wait(kv.Pull(sum_keys, &avg_vector));
        cblas_dscal(GRAD_DIM, -LR*MINIBATCH_FRAC, avg_vector.data(), 1);

        std::vector<int> minibatch;
        for (int i = 0; i < minibatch_size; i++)
            minibatch.push_back((rand() * 1.0) / RAND_MAX * (num_records - 1));
        
        error = ComputeUpdate(update.data(), W.data(), db, 
            minibatch, lr, sum_keys, 
            history_offset, kv, rank != 0);
        vdAdd(GRAD_DIM, update.data(), avg_vector.data(), update.data());

        if (rank != 0) kv.Wait(kv.Push(weight_keys, update)); 
        else std::cout << "Iter[" << t <<  "] MSE: " << error << '\n';

        while (true) {
            progress.clear();
            kv.Wait(kv.Pull(rank_keys, &progress));
            if (*std::min_element(progress.begin(), progress.end()) < t - BOUND) sleep(1);
            else break;
        }
        std::vector<double> my_progress = {1.0};
        kv.Wait(kv.Push(my_rank, my_progress));
    }
}
