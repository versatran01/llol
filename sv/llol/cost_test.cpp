#include "sv/llol/cost.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv {
namespace {

struct Cost {
  using Scalar = double;
  enum { NUM_PARAMETERS = 3, NUM_RESIDUALS = 3 };
  static constexpr int kNumParams = NUM_PARAMETERS;
  int NumResiduals() const { return NUM_RESIDUALS; }

  bool operator()(const double* _x, double* _r, double* _J) const {
    Eigen::Map<const Eigen::Vector3d> x(_x);

    const auto eR = Sophus::SO3d::exp(x);
    Eigen::Map<Eigen::Vector3d> r(_r);

    r = a - eR * (R * b);

    if (_J) {
      Eigen::Map<Eigen::Matrix3d> J(_J);
      J = Hat3(R * b);
    }

    return true;
  }

  template <typename T>
  bool operator()(const T* _x, T* _r) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> x(_x);

    const auto eR = Sophus::SO3<T>::exp(x);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> r(_r);

    r = a - eR * (R * b);
    return true;
  }

  Eigen::Vector3d a;
  Eigen::Vector3d b;
  Sophus::SO3d R;
};

Cost MakeCost() {
  Cost c;
  c.b = Eigen::Vector3d::Random();
  c.R.setQuaternion(Eigen::Quaterniond::UnitRandom());
  const Eigen::Vector3d e = Eigen::Vector3d::Random() * 0.001;
  c.a = Sophus::SO3d::exp(e) * c.R * c.b;
  return c;
}

// TEST(CostTest, TestJacobian) {
//  Cost c = MakeCost();

//  Eigen::Vector3d x0;
//  x0.setZero();
//  Eigen::Vector3d r0;
//  Eigen::Matrix3d J0;
//  c(x0.data(), r0.data(), J0.data());

//  Eigen::Vector3d x1;
//  x1.setZero();
//  Eigen::Vector3d r1;
//  Eigen::Matrix3d J1;
//  AdCost<Cost> adc(c);
//  adc(x1.data(), r1.data(), J1.data());

//  EXPECT_TRUE(x0.isApprox(x1));
//  EXPECT_TRUE(J0.isApprox(J1));
//}

void BM_CostManual(benchmark::State& state) {
  Cost c = MakeCost();

  Eigen::Vector3d x0;
  x0.setZero();
  Eigen::Vector3d r0;
  Eigen::Matrix3d J0;

  for (auto _ : state) {
    c(x0.data(), r0.data(), J0.data());
    benchmark::DoNotOptimize(r0);
  }
}
BENCHMARK(BM_CostManual);

// void BM_CostAutodiff(benchmark::State& state) {
//  Cost c = MakeCost();
//  AdCost<Cost> adc(c);

//  Eigen::Vector3d x0;
//  x0.setZero();
//  Eigen::Vector3d r0;
//  Eigen::Matrix3d J0;

//  for (auto _ : state) {
//    adc(x0.data(), r0.data(), J0.data());
//    benchmark::DoNotOptimize(r0);
//  }
//}
// BENCHMARK(BM_CostAutodiff);

}  // namespace
}  // namespace sv
