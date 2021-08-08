#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "sv/util/math.h"

// https://docs.ros.org/en/noetic/api/rviz/html/c++/covariance__visual_8cpp_source.html
using namespace visualization_msgs;
using namespace Eigen;

void MakeRightHanded(Matrix3d& eigvecs, Vector3d& eigvals) {
  // Note that sorting of eigenvalues may end up with left-hand coordinate
  // system. So here we correctly sort it so that it does end up being
  // right-handed and normalised.
  auto rhs = eigvecs.col(0).cross(eigvecs.col(1)).dot(eigvecs.col(2));
  if (rhs < 0) {
    eigvecs.col(0).swap(eigvecs.col(1));
    eigvals.row(0).swap(eigvals.row(1));
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "cov_viz");

  ros::NodeHandle pnh("~");
  ros::Publisher pub_point = pnh.advertise<Marker>("point", 1, true);
  ros::Publisher pub_covar = pnh.advertise<Marker>("covar", 1, true);

  Matrix3Xd X = Matrix3Xd::Random(3, 200) * 2.0;
  X.row(0).array() *= 2.0;
  X.row(1).array() /= 2.0;
  X.row(2).array() += 1.0;

  Matrix3d R = AngleAxisd(M_PI / 3, Vector3d::UnitZ()).toRotationMatrix();
  X = R * X;

  sv::MeanCovar3d mc;
  for (int i = 0; i < X.cols(); ++i) mc.Add(X.col(i));

  ROS_INFO_STREAM("mean:\n " << mc.mean.transpose());
  ROS_INFO_STREAM("covar:\n " << mc.covar());

  Marker m_points;
  m_points.header.frame_id = "os_lidar";
  m_points.header.stamp = ros::Time::now();
  m_points.ns = "point";
  m_points.type = visualization_msgs::Marker::SPHERE_LIST;
  m_points.action = visualization_msgs::Marker::ADD;
  m_points.color.a = 1.0;
  m_points.color.r = 1.0;
  m_points.scale.x = m_points.scale.y = m_points.scale.z = 0.2;
  m_points.pose.orientation.w = 1.0;

  for (int i = 0; i < X.cols(); ++i) {
    const Vector3d x = X.col(i);
    geometry_msgs::Point p;
    p.x = x.x();
    p.y = x.y();
    p.z = x.z();
    m_points.points.push_back(p);
  }
  SelfAdjointEigenSolver<Matrix3d> es;
  es.compute(mc.covar());
  Vector3d eigvals = es.eigenvalues();
  Matrix3d eigvecs = es.eigenvectors();
  MakeRightHanded(eigvecs, eigvals);
  Quaterniond q(eigvecs);
  Vector3d scales = eigvals.cwiseSqrt() * 2;

  Marker m_covar;
  m_covar.header.frame_id = "os_lidar";
  m_covar.header.stamp = ros::Time::now();
  m_covar.ns = "covar";
  m_covar.type = visualization_msgs::Marker::SPHERE;
  m_covar.action = visualization_msgs::Marker::ADD;
  m_covar.pose.position.x = mc.mean.x();
  m_covar.pose.position.y = mc.mean.y();
  m_covar.pose.position.z = mc.mean.z();
  m_covar.pose.orientation.w = q.w();
  m_covar.pose.orientation.x = q.x();
  m_covar.pose.orientation.y = q.y();
  m_covar.pose.orientation.z = q.z();
  m_covar.scale.x = scales.x();
  m_covar.scale.y = scales.y();
  m_covar.scale.z = scales.z();
  m_covar.color.a = 0.5;
  m_covar.color.g = 1.0;

  ros::Rate r(10);
  while (ros::ok()) {
    pub_point.publish(m_points);
    pub_covar.publish(m_covar);
    r.sleep();
  }
}
