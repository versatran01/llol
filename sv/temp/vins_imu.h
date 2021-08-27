#pragma once
#include <ceres/ceres.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>
#include <cstring>

namespace sv {

using namespace Eigen;

enum StateOrder { O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12 };

enum NoiseOrder { O_AN = 0, O_GN = 3, O_AW = 6, O_GW = 9 };

class Utility {
 public:
  template <typename Derived>
  static Quaternion<typename Derived::Scalar> deltaQ(
      const MatrixBase<Derived>& theta) {
    typedef typename Derived::Scalar Scalar_t;

    Quaternion<Scalar_t> dq;
    Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
  }

  template <typename Derived>
  static Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(
      const MatrixBase<Derived>& q) {
    Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1), q(2),
        typename Derived::Scalar(0), -q(0), -q(1), q(0),
        typename Derived::Scalar(0);
    return ans;
  }

  template <typename Derived>
  static Quaternion<typename Derived::Scalar> positify(
      const QuaternionBase<Derived>& q) {
    return q;
  }

  template <typename Derived>
  static Matrix<typename Derived::Scalar, 4, 4> Qleft(
      const QuaternionBase<Derived>& q) {
    Quaternion<typename Derived::Scalar> qq = positify(q);
    Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(
        1, 0) = qq.vec(),
           ans.template block<3, 3>(1, 1) =
               qq.w() * Matrix<typename Derived::Scalar, 3, 3>::Identity() +
               skewSymmetric(qq.vec());
    return ans;
  }

  template <typename Derived>
  static Matrix<typename Derived::Scalar, 4, 4> Qright(
      const QuaternionBase<Derived>& p) {
    Quaternion<typename Derived::Scalar> pp = positify(p);
    Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(
        1, 0) = pp.vec(),
           ans.template block<3, 3>(1, 1) =
               pp.w() * Matrix<typename Derived::Scalar, 3, 3>::Identity() -
               skewSymmetric(pp.vec());
    return ans;
  }

  static Vector3d R2ypr(const Matrix3d& R) {
    Vector3d n = R.col(0);
    Vector3d o = R.col(1);
    Vector3d a = R.col(2);

    Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r =
        atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
  }

  template <typename Derived>
  static Matrix<typename Derived::Scalar, 3, 3> ypr2R(
      const MatrixBase<Derived>& ypr) {
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0) / 180.0 * M_PI;
    Scalar_t p = ypr(1) / 180.0 * M_PI;
    Scalar_t r = ypr(2) / 180.0 * M_PI;

    Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;

    Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p);

    Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r);

    return Rz * Ry * Rx;
  }

  static Matrix3d g2R(const Vector3d& g);

  template <size_t N>
  struct uint_ {};

  template <size_t N, typename Lambda, typename IterT>
  void unroller(const Lambda& f, const IterT& iter, uint_<N>) {
    unroller(f, iter, uint_<N - 1>());
    f(iter + N);
  }

  template <typename Lambda, typename IterT>
  void unroller(const Lambda& f, const IterT& iter, uint_<0>) {
    f(iter);
  }

  template <typename T>
  static T normalizeAngle(const T& angle_degrees) {
    T two_pi(2.0 * 180);
    if (angle_degrees > 0)
      return angle_degrees -
             two_pi * std::floor((angle_degrees + T(180)) / two_pi);
    else
      return angle_degrees +
             two_pi * std::floor((-angle_degrees + T(180)) / two_pi);
  }
};

static constexpr double ACC_N = 1e-3;
static constexpr double GYR_N = 1e-4;
static constexpr double ACC_W = 1e-4;
static constexpr double GYR_W = 1e-5;

class IntegrationBase {
 public:
  IntegrationBase() = delete;
  IntegrationBase(const Vector3d& _acc_0,
                  const Vector3d& _gyr_0,
                  const Vector3d& _linearized_ba,
                  const Vector3d& _linearized_bg)
      : acc_0{_acc_0},
        gyr_0{_gyr_0},
        linearized_acc{_acc_0},
        linearized_gyr{_gyr_0},
        linearized_ba{_linearized_ba},
        linearized_bg{_linearized_bg},
        jacobian{Matrix<double, 15, 15>::Identity()},
        covariance{Matrix<double, 15, 15>::Zero()},
        sum_dt{0.0},
        delta_p{Vector3d::Zero()},
        delta_q{Quaterniond::Identity()},
        delta_v{Vector3d::Zero()}

  {
    noise = Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Matrix3d::Identity();
  }

  void push_back(double dt, const Vector3d& acc, const Vector3d& gyr) {
    dt_buf.push_back(dt);
    acc_buf.push_back(acc);
    gyr_buf.push_back(gyr);
    propagate(dt, acc, gyr);
  }

  void repropagate(const Vector3d& _linearized_ba,
                   const Vector3d& _linearized_bg) {
    sum_dt = 0.0;
    acc_0 = linearized_acc;
    gyr_0 = linearized_gyr;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    linearized_ba = _linearized_ba;
    linearized_bg = _linearized_bg;
    jacobian.setIdentity();
    covariance.setZero();
    for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
      propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
  }

  void midPointIntegration(double _dt,
                           const Vector3d& _acc_0,
                           const Vector3d& _gyr_0,
                           const Vector3d& _acc_1,
                           const Vector3d& _gyr_1,
                           const Vector3d& delta_p,
                           const Quaterniond& delta_q,
                           const Vector3d& delta_v,
                           const Vector3d& linearized_ba,
                           const Vector3d& linearized_bg,
                           Vector3d& result_delta_p,
                           Quaterniond& result_delta_q,
                           Vector3d& result_delta_v,
                           Vector3d& result_linearized_ba,
                           Vector3d& result_linearized_bg,
                           bool update_jacobian) {
    // ROS_INFO("midpoint integration");
    Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
    Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    result_delta_q = delta_q * Quaterniond(1,
                                           un_gyr(0) * _dt / 2,
                                           un_gyr(1) * _dt / 2,
                                           un_gyr(2) * _dt / 2);
    Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
    result_delta_v = delta_v + un_acc * _dt;
    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    if (update_jacobian) {
      Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
      Vector3d a_0_x = _acc_0 - linearized_ba;
      Vector3d a_1_x = _acc_1 - linearized_ba;
      Matrix3d R_w_x, R_a_0_x, R_a_1_x;

      R_w_x << 0, -w_x(2), w_x(1), w_x(2), 0, -w_x(0), -w_x(1), w_x(0), 0;
      R_a_0_x << 0, -a_0_x(2), a_0_x(1), a_0_x(2), 0, -a_0_x(0), -a_0_x(1),
          a_0_x(0), 0;
      R_a_1_x << 0, -a_1_x(2), a_1_x(1), a_1_x(2), 0, -a_1_x(0), -a_1_x(1),
          a_1_x(0), 0;

      MatrixXd F = MatrixXd::Zero(15, 15);
      F.block<3, 3>(0, 0) = Matrix3d::Identity();
      F.block<3, 3>(0, 3) =
          -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
          -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x *
              (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
      F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(0, 9) =
          -0.25 *
          (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) *
          _dt * _dt;
      F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() *
                             R_a_1_x * _dt * _dt * -_dt;
      F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
      F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
      F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x *
                                (Matrix3d::Identity() - R_w_x * _dt) * _dt;
      F.block<3, 3>(6, 6) = Matrix3d::Identity();
      F.block<3, 3>(6, 9) =
          -0.5 *
          (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) *
          _dt;
      F.block<3, 3>(6, 12) =
          -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
      F.block<3, 3>(9, 9) = Matrix3d::Identity();
      F.block<3, 3>(12, 12) = Matrix3d::Identity();
      // cout<<"A"<<endl<<A<<endl;

      MatrixXd V = MatrixXd::Zero(15, 18);
      V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
      V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() *
                            R_a_1_x * _dt * _dt * 0.5 * _dt;
      V.block<3, 3>(0, 6) =
          0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
      V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
      V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 3) =
          0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
      V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
      V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
      V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
      V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

      // step_jacobian = F;
      // step_V = V;
      jacobian = F * jacobian;
      covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }
  }

  void propagate(double _dt, const Vector3d& _acc_1, const Vector3d& _gyr_1) {
    dt = _dt;
    acc_1 = _acc_1;
    gyr_1 = _gyr_1;
    Vector3d result_delta_p;
    Quaterniond result_delta_q;
    Vector3d result_delta_v;
    Vector3d result_linearized_ba;
    Vector3d result_linearized_bg;

    midPointIntegration(_dt,
                        acc_0,
                        gyr_0,
                        _acc_1,
                        _gyr_1,
                        delta_p,
                        delta_q,
                        delta_v,
                        linearized_ba,
                        linearized_bg,
                        result_delta_p,
                        result_delta_q,
                        result_delta_v,
                        result_linearized_ba,
                        result_linearized_bg,
                        1);

    delta_p = result_delta_p;
    delta_q = result_delta_q;
    delta_v = result_delta_v;
    linearized_ba = result_linearized_ba;
    linearized_bg = result_linearized_bg;
    delta_q.normalize();
    sum_dt += dt;
    acc_0 = acc_1;
    gyr_0 = gyr_1;
  }

  Matrix<double, 15, 1> evaluate(const Vector3d& Pi,
                                 const Quaterniond& Qi,
                                 const Vector3d& Vi,
                                 const Vector3d& Bai,
                                 const Vector3d& Bgi,
                                 const Vector3d& Pj,
                                 const Quaterniond& Qj,
                                 const Vector3d& Vj,
                                 const Vector3d& Baj,
                                 const Vector3d& Bgj) {
    Matrix<double, 15, 1> residuals;

    Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

    Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

    Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

    Vector3d dba = Bai - linearized_ba;
    Vector3d dbg = Bgi - linearized_bg;

    Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
    Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

    Vector3d G(0.0, 0.0, 9.81);
    residuals.block<3, 1>(O_P, 0) =
        Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) -
        corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) =
        2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    residuals.block<3, 1>(O_V, 0) =
        Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
    //    residuals.block<3, 1>(O_V, 0) =
    //        2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
    //    residuals.block<3, 1>(O_R, 0) =
    //    Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
  }

  double dt;
  Vector3d acc_0, gyr_0;
  Vector3d acc_1, gyr_1;

  const Vector3d linearized_acc, linearized_gyr;
  Vector3d linearized_ba, linearized_bg;

  Matrix<double, 15, 15> jacobian, covariance;
  Matrix<double, 15, 15> step_jacobian;
  Matrix<double, 15, 18> step_V;
  Matrix<double, 18, 18> noise;

  double sum_dt;
  Vector3d delta_p;
  Quaterniond delta_q;
  Vector3d delta_v;

  std::vector<double> dt_buf;
  std::vector<Vector3d> acc_buf;
  std::vector<Vector3d> gyr_buf;
};

class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
 public:
  IMUFactor() = delete;
  IMUFactor(IntegrationBase* _pre_integration)
      : pre_integration(_pre_integration) {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    //    Vector3d Pi(parameters[0][0], parameters[0][1],
    //    parameters[0][2]); Quaterniond Qi(parameters[0][6],
    //    parameters[0][3], parameters[0][4],
    //                          parameters[0][5]);
    Vector3d Pi(parameters[0][4], parameters[0][5], parameters[0][6]);
    Quaterniond Qi(
        parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);

    Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
    Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

    //    Vector3d Pj(parameters[2][0], parameters[2][1],
    //    parameters[2][2]); Quaterniond Qj(parameters[2][6],
    //    parameters[2][3], parameters[2][4],
    //                          parameters[2][5]);
    Vector3d Pj(parameters[2][4], parameters[2][5], parameters[2][6]);
    Quaterniond Qj(
        parameters[2][3], parameters[2][0], parameters[2][1], parameters[2][2]);

    Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
    Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

    Map<Matrix<double, 15, 1>> residual(residuals);
    residual =
        pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

    Matrix<double, 15, 15> sqrt_info =
        LLT<Matrix<double, 15, 15>>(pre_integration->covariance.inverse())
            .matrixL()
            .transpose();
    // sqrt_info.setIdentity();
    //    residual = sqrt_info * residual;

    if (jacobians) {
      double sum_dt = pre_integration->sum_dt;
      Matrix3d dp_dba =
          pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
      Matrix3d dp_dbg =
          pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

      Matrix3d dq_dbg =
          pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

      Matrix3d dv_dba =
          pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
      Matrix3d dv_dbg =
          pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

      if (pre_integration->jacobian.maxCoeff() > 1e8 ||
          pre_integration->jacobian.minCoeff() < -1e8) {
        //        ROS_WARN("numerical unstable in preintegration");
      }

      if (jacobians[0]) {
        Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_i(jacobians[0]);
        jacobian_pose_i.setZero();

        jacobian_pose_i.block<3, 3>(O_P, O_P) =
            -Qi.inverse().toRotationMatrix();
        Vector3d G(0.0, 0.0, 9.81);
        jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(
            Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

        Quaterniond corrected_delta_q =
            pre_integration->delta_q *
            Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
        jacobian_pose_i.block<3, 3>(O_R, O_R) =
            -(Utility::Qleft(Qj.inverse() * Qi) *
              Utility::Qright(corrected_delta_q))
                 .bottomRightCorner<3, 3>();

        jacobian_pose_i.block<3, 3>(O_V, O_R) =
            Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

        jacobian_pose_i = sqrt_info * jacobian_pose_i;

        if (jacobian_pose_i.maxCoeff() > 1e8 ||
            jacobian_pose_i.minCoeff() < -1e8) {
          //          ROS_WARN("numerical unstable in preintegration");
        }
      }
      if (jacobians[1]) {
        Map<Matrix<double, 15, 9, RowMajor>> jacobian_speedbias_i(jacobians[1]);
        jacobian_speedbias_i.setZero();
        jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) =
            -Qi.inverse().toRotationMatrix() * sum_dt;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
        jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

        jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) =
            -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q)
                 .bottomRightCorner<3, 3>() *
            dq_dbg;

        jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) =
            -Qi.inverse().toRotationMatrix();
        jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
        jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

        jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) =
            -Matrix3d::Identity();

        jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) =
            -Matrix3d::Identity();

        jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

        // ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
        // ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
      }
      if (jacobians[2]) {
        Map<Matrix<double, 15, 7, RowMajor>> jacobian_pose_j(jacobians[2]);
        jacobian_pose_j.setZero();

        jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

        Quaterniond corrected_delta_q =
            pre_integration->delta_q *
            Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
        jacobian_pose_j.block<3, 3>(O_R, O_R) =
            Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj)
                .bottomRightCorner<3, 3>();

        jacobian_pose_j = sqrt_info * jacobian_pose_j;

        // ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
        // ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
      }
      if (jacobians[3]) {
        Map<Matrix<double, 15, 9, RowMajor>> jacobian_speedbias_j(jacobians[3]);
        jacobian_speedbias_j.setZero();

        jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) =
            Qi.inverse().toRotationMatrix();

        jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) =
            Matrix3d::Identity();

        jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) =
            Matrix3d::Identity();

        jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
      }
    }

    return true;
  }

  IntegrationBase* pre_integration;
};

}  // namespace sv
