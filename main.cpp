#include <cmath>
#include <iostream>
#include "ps/ps.h"

using namespace ps;

void StartServer() {
  if (!IsServer()) {
    return;
  }
  auto server = new KVServer<float>(0);
  server->set_request_handle(KVServerDefaultHandle<float>());
  RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!IsWorker()) return;

    std::vector<uint64_t> key = { 1, 3, 5 };
	std::vector<float> val = { 1, 1, 1 };
	std::vector<float> recv_val;
	ps::KVWorker<float> w(0,0);
	w.Wait(w.Push(key, val));
	w.Wait(w.Pull(key, &recv_val));

    for (auto i: recv_val) {
        std::cout << i << ' ';
    }
    std::cout << '\n';
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
