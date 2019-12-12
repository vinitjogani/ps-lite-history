#include <vector>
#include <unistd.h>

#include "ps/ps.h"

using namespace ps;

class AsyncController {

private:
    
    const double SLEEP_INTERVAL = 200;
    const int BOUND = 2;
    const float THROTTLE = 0.6;

    std::vector<Key> my_rank;
    std::vector<Key> rank_keys;

    std::vector<double> my_progress = {1.0};
    std::vector<double> progress;

    KVWorker<double> &kv;
    int rank, num_workers, iter = 0;

public:
    AsyncController(int rank, int num_workers, int num_weights, KVWorker<double> &kv) : 
        kv(kv), rank(rank), num_workers(num_workers) {
        my_rank.push_back(num_weights + rank);
        for (int i = 0; i < num_workers; i++) rank_keys.push_back(num_weights + i);
    }

    bool StalenessBounded(std::vector<double> &progress) {
        return *std::min_element(progress.begin(), progress.end()) >= (iter - BOUND);
    }

    bool ThrottledRelease(std::vector<double> &progress) {
        float num = 0;
        for (auto w : progress) {
            num += (w >= iter);
        }
        return (num / num_workers) >= THROTTLE;
    }

    double Barrier() {
        double slept = 0;
        while (true) {
            progress.clear();
            kv.Wait(kv.Pull(rank_keys, &progress));
            
            if (ThrottledRelease(progress) && StalenessBounded(progress)) {
                break;
            }
            else {
                usleep(SLEEP_INTERVAL);
                slept += SLEEP_INTERVAL;
            }
        }
        return slept;
    }

    double CompleteIteration() {
        kv.Wait(kv.Push(my_rank, my_progress));
        double slept = Barrier();
        iter ++;
        return slept;
    }

};