/**
 * Solution from 
 * https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
 * C++ progress bar
*/
#pragma once
#include <iostream>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 100

void printProgress(int spp_now, int spp_full) {
    float percentage = float(spp_now + 1) / float(spp_full); 
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% (%4d / %4d) [%.*s%*s]", val, spp_now + 1, spp_full, lpad, PBSTR, rpad, "");
    fflush(stdout);
}
