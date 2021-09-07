#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  Eigen::LDLT<Eigen::MatrixXd> ldlt;
  Eigen::Matrix3d m = Eigen::Matrix3d::Identity();

  ldlt.compute(m);

  Eigen::Matrix3d u = ldlt.matrixU();
  std::cout << u;
}
