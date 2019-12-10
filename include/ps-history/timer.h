#pragma once

#include <time.h>
#include <iostream>

enum class TimerType { COMP, COMM, COMM_ASYNC, HISTORY };

class Timer {
private:
    static double computation;
    static double communication_sync;
    static double communication_async;

    clock_t start;
    TimerType type;

public: 

    Timer(TimerType type = TimerType::COMP, bool started = true) {
        if (started) Start();
        this->type = type;
    }

    void Start() {
        start = clock();
    }

    void Stop() {
        double elapsed = (clock() - start) / ((double)CLOCKS_PER_SEC);
        switch(type) {
            case TimerType::COMP:
                computation += elapsed;
                break;
            case TimerType::COMM:
                communication_sync += elapsed;
                break;
            case TimerType::COMM_ASYNC:
                communication_async += elapsed;
                break;
            case TimerType::HISTORY:
                communication_sync += elapsed;
                computation -= elapsed;
                break;
        }
    }

    static void PrintSummary() {
        std::cout << "-----------------------\n";
        std::cout << "Computation time:\t\t" << computation << std::endl;
        std::cout << "Communication time (sync):\t" << communication_sync << std::endl;
        std::cout << "Communication time (async):\t" << communication_async << std::endl;
        std::cout << "Total (COMP + COMM_SYNC):\t" << computation + communication_sync << std::endl;
        std::cout << std::endl;
    }

};

double Timer::computation = 0;
double Timer::communication_sync = 0;
double Timer::communication_async = 0;