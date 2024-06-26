/*
 *   progressbar.hpp
 * 
 *     Created on: Apr 13, 2023
 *         Author: Jeffery Wang
 * 
 */

#pragma once
#ifndef UTILS_PROGRESSBAR_HPP
#define UTILS_PROGRESSBAR_HPP

#include <chrono>
#include <iostream>
#include <boost/format.hpp>

namespace progresscpp {
    
    class ProgressBar {
        private:
            unsigned int ticks = 0;
            const unsigned int total_ticks;
            const unsigned int bar_width;
            const char complete_char = '=';
            const char incomplete_char = ' ';
            const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

        public:
            ProgressBar(unsigned int total, unsigned int width, char complete, char incomplete) :
                    total_ticks{total}, bar_width{width}, complete_char{complete}, incomplete_char{incomplete} {}

            ProgressBar(unsigned int total, unsigned int width) : total_ticks{total}, bar_width{width} {}

            unsigned int operator++() { return ++ticks; }

            void display() const {
                float progress = (float) ticks / total_ticks;
                int pos = (int) (bar_width * progress);

                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

                std::cout << "[";

                for (int i = 0; i < (int)bar_width; ++i) {
                    if (i < pos) std::cout << complete_char;
                    else if (i == pos) std::cout << ">";
                    else std::cout << incomplete_char;
                }
                
                std::cout << boost::format("] %d%% %.3fs\r") % int(progress*100.0) % (float(time_elapsed)/1000.0);
                std::cout.flush();
            }

            void done() const {
                display();
                std::cout << std::endl;
            }
    };

} // namespace progresscpp

#endif // UTILS_PROGRESSBAR_HPP