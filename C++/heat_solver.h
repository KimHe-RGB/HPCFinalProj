//
// Created by Keigo Ando on 5/13/23.
//
#ifndef HEAT_SOLVER_H
#define HEAT_SOLVER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

const long double PI = 3.14159265358979323846264338327950288419716939937510;
Eigen::VectorXd u_sol_f(const Eigen::VectorXd &x, long double t);

long double lbry_f(long double t);

long double rbry_f(long double t);

Eigen::VectorXd diffusivity_f(const Eigen::VectorXd &x);

Eigen::VectorXd force_f(const Eigen::VectorXd &x, long double t);

Eigen::VectorXd init_cond_f(const Eigen::VectorXd &x);

std::pair<Eigen::MatrixXd, Eigen::VectorXd> Spatial_Operator_Heat1D(
    const std::vector<long double> &xspan,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df);

Eigen::VectorXd Heat1D_RK3_solver(
    const std::vector<long double> &xspan,
    const std::vector<long double> &tspan, const Eigen::VectorXd &init_con,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &, long double)>
        &F,
    const std::function<long double(long double)> &lbry,
    const std::function<long double(long double)> &rbry);

Eigen::VectorXd Heat1D_IE_solver(
    const std::vector<long double> &xspan,
    const std::vector<long double> &tspan, const Eigen::VectorXd &init_con,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &, long double)>
        &F,
    const std::function<long double(long double)> &lbry,
    const std::function<long double(long double)> &rbry);

Eigen::VectorXd Heat1D_CN_solver(
    const std::vector<long double> &xspan,
    const std::vector<long double> &tspan, const Eigen::VectorXd &init_con,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &, long double)>
        &F,
    const std::function<long double(long double)> &lbry,
    const std::function<long double(long double)> &rbry);

#endif  // HEAT_SOLVER_H
