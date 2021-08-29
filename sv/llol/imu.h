#pragma once

#include <boost/circular_buffer.hpp>
#include <sophus/se3.hpp>

namespace sv {

static const Eigen::Vector3d kZero3d = Eigen::Vector3d::Zero();
static const Eigen::Matrix3d kIdent3 = Eigen::Matrix3d::Identity();

struct NavState {
  double time{};
  Sophus::SO3d rot{};
  Eigen::Vector3d pos{kZero3d};
  Eigen::Vector3d vel{kZero3d};
};

struct ImuBias {
  Eigen::Vector3d acc{kZero3d};
  Eigen::Vector3d gyr{kZero3d};
};

/// @brief Time-stamped Imu data
struct ImuData {
  double time{};
  Eigen::Vector3d acc{kZero3d};
  Eigen::Vector3d gyr{kZero3d};

  void Debias(const ImuBias& bias) {
    acc -= bias.acc;
    gyr -= bias.gyr;
  }

  ImuData DeBiased(const ImuBias& bias) const {
    ImuData out = *this;
    out.Debias(bias);
    return out;
  }
};

/// Integrate rotation one step, assumes de-biased gyro data
Sophus::SO3d IntegrateRot(const Sophus::SO3d& R0,
                          const Eigen::Vector3d& omg,
                          double dt);

/// Integrate nav state one step, assume de-biased imu data
/// @param s0 is current state, imu is imu data, g_w is gravity in world frame
NavState IntegrateEuler(const NavState& s0,
                        const ImuData& imu,
                        const Eigen::Vector3d& g_w,
                        double dt);

NavState IntegrateMidpoint(const NavState& s0,
                           const ImuData& imu0,
                           const ImuData& imu1,
                           const Eigen::Vector3d& g_w);

using ImuBuffer = boost::circular_buffer<ImuData>;

/// @brief Discrete time IMU noise
struct ImuNoise {
  using Vector12d = Eigen::Matrix<double, 12, 1>;
  enum Index { NA = 0, NW = 3, BA = 6, BW = 9 };

  ImuNoise() = default;
  ImuNoise(double dt,
           double acc_noise,
           double gyr_noise,
           double acc_bias_noise,
           double gyr_bias_noise);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const ImuNoise& rhs) {
    return os << rhs.Repr();
  }

  Vector12d sigma2{Vector12d::Zero()};  // discrete time noise covar
};

/// @brief Get the index of the imu right after time t
int FindNextImu(const ImuBuffer& buf, double t);

/// @brief Accumulates imu data and integrate
/// @todo for now only integrate gyro for rotation
struct ImuTrajectory {
  ImuTrajectory() = default;
  explicit ImuTrajectory(int size) { states.resize(size); }

  ImuBuffer buf{16};
  ImuBias bias;
  ImuNoise noise;

  Eigen::Vector3d gravity;       // gravity vector in pano frame
  Sophus::SE3d T_init_pano{};    // tf from pano to init frame
  Sophus::SE3d T_imu_lidar{};    // extrinsics lidar to imu
  std::vector<NavState> states;  // imu state wrt current pano

  int size() const { return states.size(); }
  auto& StateAt(int i) { return states.at(i); }
  const auto& StateAt(int i) const { return states.at(i); }

  void InitGravity(double gravity_norm);
  void InitExtrinsic(const Sophus::SE3d& T_i_l);

  /// @brief Add imu data into buffer
  void Add(const ImuData& imu) { buf.push_back(imu); }

  /// @brief Given the first pose in poses, predict using imu
  /// @return Number of imus used
  int Predict(double t0, double dt);
};

/// @brief Imu preintegration
struct ImuPreintegration {
  static constexpr int kDim = 15;
  using Matrix15d = Eigen::Matrix<double, kDim, kDim>;
  enum Index { ALPHA = 0, BETA = 3, THETA = 6, BA = 9, BW = 12 };

  void Reset();
  void Integrate(double dt, const ImuData& imu, const ImuNoise& noise);
  void Integrate(double dt,
                 const ImuData& imu0,
                 const ImuData& imu1,
                 const ImuNoise& noise);
  void SqrtInfo();

  /// Data
  int n{0};  // number of imus used
  Eigen::Vector3d alpha{kZero3d};
  Eigen::Vector3d beta{kZero3d};
  Sophus::SO3d gamma{};
  Matrix15d F{Matrix15d::Identity()};  // State transition matrix
  Matrix15d P{Matrix15d::Zero()};      // Covariance matrix
  Matrix15d U{Matrix15d::Zero()};      // Square root information matrix
};

}  // namespace sv
