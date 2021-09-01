#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::Matrix3d x = Eigen::Matrix3d::Random();
  Eigen::Matrix3d A = x.transpose() * x;

  Eigen::Matrix3d A_inv = A.inverse();

  std::cout << "A_inv:\n" << A_inv << "\n";

  Eigen::LDLT<Eigen::Matrix3d> solver;
  solver.compute(A);
  Eigen::Matrix3d A_inv2 = solver.solve(Eigen::Matrix3d::Identity());
  std::cout << "A_inv2:\n" << A_inv2 << "\n";
}
