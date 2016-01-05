#ifndef QUASIONED_H
#define QUASIONED_H

#include<vector>

double quasiOneD(std::vector <double> x, 
                 std::vector <double> dx, 
                 std::vector <double> S,
                 std::vector <double> designVar,
                 std::vector <double> &W);

double TotalPressureLoss(std::vector <double> W);

void ioTargetPressure(int io, std::vector <double> &p);

#endif
