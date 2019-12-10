

#include "ps/ps.h"

#include "ps-history/worker.h"

using namespace ps;

void StartServer() {
    auto server = new KVServer<double>(0);
    server->set_request_handle(KVServerDefaultHandle<double>());
    RegisterExitCallback([server](){ delete server; });
}

int main(int argc, char *argv[]) {    
    // start system
    Start(0);
    // setup server nodes
    if (IsServer()) StartServer();
    // run worker nodes
    if (IsWorker()) RunWorker(MyRank(), NumWorkers());
    // stop system
    Finalize(0, true);
    return 0;
}
