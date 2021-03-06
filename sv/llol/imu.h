#pragma once

#include <boost/circular_buffer.hpp>
#include <sophus/se3.hpp>

namespace sv {

static const Eigen::Vector3d kVecZero3d = Eigen::Vector3d::Zero();
static const Eigen::Matrix3d kMatZero3d = Eigen::Matrix3d::Zero();
static const Eigen::Matrix3d kMatEye3d = Eigen::Matrix3d::Identity();

struct NavState {
  double time{};
  Sophus::SO3d rot{};
  Eigen::Vector3d pos{kVecZero3d};
  Eigen::Vector3d vel{kVecZero3d};

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const NavState& rhs) {
    return os << rhs.Repr();
  }
};

struct ImuBias {
  ImuBias() = default;
  ImuBias(double acc_bias_std, double gyr_bias_std);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const ImuBias& rhs) {
    return os << rhs.Repr();
  }

  void UpdateAcc(const Eigen::Vector3d& z, const Eigen::Vector3d& R);
  void UpdateGyr(const Eigen::Vector3d& z, const Eigen::Vector3d& R);

  Eigen::Vector3d acc{kVecZero3d};
  Eigen::Vector3d gyr{kVecZero3d};
  Eigen::Vector3d acc_var{kVecZero3d};
  Eigen::Vector3d gyr_var{kVecZero3d};
};

/// @brief Time-stamped Imu data
struct ImuData {
  double time{};
  Eigen::Vector3d acc{kVecZero3d};
  Eigen::Vector3d gyr{kVecZero3d};

  void Debias(const ImuBias& bias);
  ImuData DeBiased(const ImuBias& bias) const;
};

using ImuBuffer = boost::circular_buffer<ImuData>;

Sophus::SO3d IntegrateRot(const Sophus::SO3d& rot,
                          double time,
                          const ImuData& imu0,
                          const ImuData& imu1,
                          double dt);

/// Integrate nav state one step, assume de-biased imu data
/// @param s0 is current state, imu is imu data, g is gravity in fixed frame
void IntegrateEuler(const NavState& s0,
                    const ImuData& imu,
                    const Eigen::Vector3d& g,
                    double dt,
                    NavState& s1);

/// Integrate nav state for dt, assume de-biased imu data
/// @param s0 is current state, imu is imu data, g is gravity in fixed frame
void IntegrateState(const NavState& s0,
                    const ImuData& imu0,
                    const ImuData& imu1,
                    const Eigen::Vector3d& g,
                    double dt,
                    NavState& s1);

/// @brief Discrete time IMU noise
struct ImuNoise {
  static constexpr int kDim = 12;
  using Vec3CMap = Eigen::Map<const Eigen::Vector3d>;
  using Vector12d = Eigen::Matrix<double, kDim, 1>;
  enum Index { kNa = 0, kNw = 3, kNba = 6, kNbw = 9 };

  ImuNoise() = default;
  ImuNoise(double rate,
           double acc_noise,
           double gyr_noise,
           double acc_bias_noise,
           double gyr_bias_noise);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const ImuNoise& rhs) {
    return os << rhs.Repr();
  }

  /// d means discrete-time
  auto nad() const { return Vec3CMap{sigma2.data() + Index::kNa}; }
  auto nwd() const { return Vec3CMap{sigma2.data() + Index::kNw}; }
  auto nbad() const { return Vec3CMap{sigma2.data() + Index::kNba}; }
  auto nbwd() const { return Vec3CMap{sigma2.data() + Index::kNbw}; }

  Vector12d sigma2{Vector12d::Zero()};  // discrete time noise covar
};

/// @brief Get the index of the imu right after time t
int GetImuIndexAfterTime(const ImuBuffer& buf, double t);

struct ImuQueue {
  ImuQueue() = default;
  explicit ImuQueue(int buffer_size) : buf(buffer_size) {}

  ImuBias bias;
  ImuNoise noise;
  ImuBuffer buf{20};

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const ImuQueue& rhs) {
    return os << rhs.Repr();
  }

  int size() const { return buf.size(); }
  int full() const { return buf.full(); }
  bool empty() const { return buf.empty(); }
  int capacity() const { return buf.capacity(); }

  /// @brief Add imu data into buffer
  void Add(const ImuData& imu);

  /// @brief At
  const ImuData& RawAt(int i) const { return buf.at(i); }
  ImuData DebiasedAt(int i) const { return buf.at(i).DeBiased(bias); }

  /// @brief Get index into buffer with time rgith next to t
  /// @return -1 if not found
  int IndexAfter(double t) const;

  /// @brief Compute mean imu data
  ImuData CalcMean(int last_n = 0) const;
};

/// @brief Imu preintegration
struct ImuPreintegration {
  static constexpr int kDim = 15;
  using Matrix15d = Eigen::Matrix<double, kDim, kDim>;
  enum Index { kAlpha = 0, kBeta = 3, kTheta = 6, kBa = 9, kBw = 12 };

  /// @brief Compute measurement for imu trajectory
  int Compute(const ImuQueue& imuq, double t0, double t1);
  void Reset();
  void Integrate(double dt, const ImuData& imu, const ImuNoise& noise);
  bool Ok() const noexcept { return n > 0; }

  /// Data
  int n{0};           // number of times integrated
  double duration{};  // duration of integration

  Eigen::Vector3d alpha{kVecZero3d};
  Eigen::Vector3d beta{kVecZero3d};
  Sophus::SO3d gamma{};

  Matrix15d F{Matrix15d::Identity()};  // State transition matrix discrete time
  Matrix15d P{Matrix15d::Zero()};      // Covariance matrix
  Matrix15d U{Matrix15d::Zero()};      // Square root information matrix
};

}  // namespace sv
