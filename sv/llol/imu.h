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

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const NavState& rhs) {
    return os << rhs.Repr();
  }
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

/// Integrate nav state one step, assume de-biased imu data
/// @param s0 is current state, imu is imu data, g is gravity in fixed frame
void IntegrateEuler(const NavState& s0,
                    const ImuData& imu,
                    const Eigen::Vector3d& g,
                    double dt,
                    NavState& s1);

NavState IntegrateMidpoint(const NavState& s0,
                           const ImuData& imu0,
                           const ImuData& imu1,
                           const Eigen::Vector3d& g_w);

using ImuBuffer = boost::circular_buffer<ImuData>;

/// @brief Discrete time IMU noise
struct ImuNoise {
  static constexpr int kDim = 12;
  using Vector12d = Eigen::Matrix<double, kDim, 1>;
  enum Index { kNa = 0, kNw = 3, kBa = 6, kBw = 9 };

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
  Sophus::SE3d T_odom_pano{};    // tf from pano to odom frame
  Sophus::SE3d T_imu_lidar{};    // extrinsics lidar to imu
  std::vector<NavState> states;  // imu state wrt current pano

  int size() const { return states.size(); }
  NavState& StateAt(int i) { return states.at(i); }
  const NavState& StateAt(int i) const { return states.at(i); }
  const ImuData& ImuAt(int i) const { return buf.at(i); }

  /// @return the acc vector used to initialize gravity
  void InitExtrinsic(const Sophus::SE3d& T_i_l, double gravity_norm);

  /// @brief Add imu data into buffer
  void Add(const ImuData& imu) { buf.push_back(imu); }

  /// @brief Given the first pose in poses, predict using imu
  /// @return Number of imus used
  /// @todo Need to handle partial sweep
  int Predict(double t0, double dt, int n);
};

/// @brief Imu preintegration
struct ImuPreintegration {
  static constexpr int kDim = 15;
  using Matrix15d = Eigen::Matrix<double, kDim, kDim>;
  enum Index { kAlpha = 0, kBeta = 3, kTheta = 6, kBa = 9, kBw = 12 };

  /// @brief Compute measurement for imu trajectory
  int Compute(const ImuTrajectory& traj);
  void Reset();

  void Integrate(double dt, const ImuData& imu, const ImuNoise& noise);

  /// Data
  int n{0};           // number of times integrated
  double duration{};  // duration of integration
  Eigen::Vector3d alpha{kZero3d};
  Eigen::Vector3d beta{kZero3d};
  Sophus::SO3d gamma{};
  Matrix15d F{Matrix15d::Identity()};  // State transition matrix discrete time
  Matrix15d P{Matrix15d::Zero()};      // Covariance matrix
  Matrix15d U{Matrix15d::Zero()};      // Square root information matrix
};

}  // namespace sv
