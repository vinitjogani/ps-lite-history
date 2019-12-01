#include <cmath>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "ps/ps.h"

#include "ps-history/reader.h"
#include "ps-history/dataset.h"

using namespace ps;

#define MnistDataset Dataset<uint8_t>
#define Weights std::vector<uint8_t>

const int NUM_RECORDS = 100;

void StartServer() {
    if (!IsServer()) return;

    auto server = new KVServer<float>(0);
    server->set_request_handle(KVServerDefaultHandle<float>());
    RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
    if (!IsWorker()) return;
    int rank = MyRank();
    int num_workers = NumWorkers();

    int per_worker = NUM_RECORDS / num_workers;
    MnistDataset *db = read_mnist(per_worker * rank, per_worker);
    std::cout << db->getNumRecords() << " records loaded\n";
}

int main(int argc, char *argv[]) {
  // start system
  Start(0);
  // setup server nodes
  StartServer();
  // run worker nodes
  RunWorker();
  // stop system
  Finalize(0, true);
  return 0;
}
