#include "ceres/jet.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <sophus/so2.hpp>

using namespace ceres;

typedef Jet<double, 2> J;
// Convenient shorthand for making a jet.
J MakeJet(double a, double v0, double v1) {
  J z;
  z.a = a;
  z.v[0] = v0;
  z.v[1] = v1;
  return z;
}

void PrintJet(const J& x) {
  std::cout << x.a << " " << x.v[0] << " " << x.v[1] << "\n";
}

int main() {
  const J x = MakeJet(2.3, -2.7, 1e-3);
  const J y = MakeJet(1.7, 0.5, 1e+2);
  const J z = MakeJet(5.3, -4.7, 1e-3);
  const J w = MakeJet(9.7, 1.5, 10.1);

  Sophus::SO2f R(0.01);
  Eigen::Matrix<J, 2, 1> v;

  v << x, y;

  // Check that M * v == M * v.cast<J>().
  const Eigen::Matrix<J, 2, 1> r1 = R.cast<double>() * v;
  const Eigen::Matrix<J, 2, 1> r2 = R.cast<J>() * v;

  PrintJet(r1(0));
  PrintJet(r2(0));
  PrintJet(r1(1));
  PrintJet(r2(1));
}