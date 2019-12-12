#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <unistd.h>

#include "ps/ps.h"
#include "ps-history/reader.h"
#include "ps-history/sparse_dataset.h"
#include "ps-history/timer.h"
#include "ps-history/worker.h"
#include "ps-history/async_controller.h"

using namespace ps;


const bool VALIDATE = true;
const std::string DATA_PATH = "mnist.scale";
const int NUM_RECORDS = 60000;
const float MINIBATCH_FRAC = 0.1;
const int NUM_ITERS = 50;
const double LR = 0.001;
const int NUM_FEATURES = 784;

double ComputeUpdate(double *update, double *W, SparseDataset *db, 
    std::vector<int> &minibatch, double lr) {

    int m = minibatch.size(), k = NUM_FEATURES;
    int ptrB[m], ptrE[m];
    for (int i = 0; i < minibatch.size(); i++) {
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

        pred[i] += W[NUM_FEATURES] - db->label(r);
        error += (pred[i] * pred[i]);

        cblas_daxpyi(nz, -lr*pred[i], db->vals() + idx, db->cols() + idx, update);
        update[NUM_FEATURES] -= lr*pred[i];
    }        

    return error / m;
}


void RunWorker(int rank, int num_workers) {

    std::vector<Key> weight_keys;
    for (int i = 0; i < NUM_FEATURES+1; i++) weight_keys.push_back(i); 
    
    KVWorker<double> kv(0, rank);
    AsyncController control(rank, num_workers, weight_keys.size(), kv);

    int per_worker = (NUM_RECORDS * 1.0 / num_workers);
    SparseDataset *db = SparseDataset::from(DATA_PATH, per_worker*rank, per_worker);
    
    int num_records = db->num_records, minibatch_size = num_records * MINIBATCH_FRAC;
    std::cout << num_records << " records loaded\n";

    std::vector<double> W (NUM_FEATURES + 1);
    std::vector<double> update (NUM_FEATURES + 1);

    Timer W_timer(TimerType::COMM);
    kv.Wait(kv.Pull(weight_keys, &W));
    W_timer.Stop();

    int last = 0;
    for (int t = 0; t < NUM_ITERS; t++) {
        memset(update.data(), 0, sizeof(double) * (NUM_FEATURES+1));

        Timer comp_timer(TimerType::COMP);
        std::vector<int> minibatch;
        for (int i = 0; i < minibatch_size; i++)
            minibatch.push_back((rand() * 1.0) / RAND_MAX * (num_records - 1));
        double error = ComputeUpdate(update.data(), W.data(), db, minibatch, LR / minibatch.size());
        comp_timer.Stop();

        Timer update_timer(TimerType::COMM_ASYNC);
        last = kv.Push(weight_keys, update, {}, 0, [&]() { update_timer.Stop(); }); 
        if (rank == 0 && VALIDATE) std::cout << "Iter[" << t <<  "] MSE: " << error << '\n';

        Timer wait_timer(TimerType::WAITING);
        double slept = control.CompleteIteration() / 1e+6;
        wait_timer.Sleep(slept);
        wait_timer.Stop();
        
        W_timer.Start();
        kv.Wait(kv.Pull(weight_keys, &W));
        W_timer.Stop();
    }

    kv.Wait(last);
    std::cout << "Worker " << rank << " summary:\n";
    Timer::PrintSummary();
}
