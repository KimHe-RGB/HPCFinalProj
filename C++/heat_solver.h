//
// Created by Keigo Ando on 5/13/23.
//
#ifndef HEAT_SOLVER_H
#define HEAT_SOLVER_H

#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <functional>

const double PI = 3.14159265358979323846;

Eigen::VectorXd u_sol_f(const Eigen::VectorXd &x, double t);

double lbry_f(double t);

double rbry_f(double t);

Eigen::VectorXd diffusivity_f(const Eigen::VectorXd &x);

Eigen::VectorXd force_f(const Eigen::VectorXd &x, double t);

Eigen::VectorXd init_cond_f(const Eigen::VectorXd &x);

std::pair<Eigen::MatrixXd, Eigen::VectorXd> Spatial_Operator_Heat1D(
        const std::vector<double> &xspan,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df);

Eigen::VectorXd Heat1D_RK3_solver(
        const std::vector<double> &xspan,
        const std::vector<double> &tspan,
        const Eigen::VectorXd &init_con,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &, double)> &F,
        const std::function<double(double)> &lbry,
        const std::function<double(double)> &rbry);

Eigen::VectorXd Heat1D_IE_solver(
        const std::vector<double> &xspan,
        const std::vector<double> &tspan,
        const Eigen::VectorXd &init_con,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &, double)> &F,
        const std::function<double(double)> &lbry,
        const std::function<double(double)> &rbry);

Eigen::VectorXd Heat1D_CN_solver(
        const std::vector<double> &xspan,
        const std::vector<double> &tspan,
        const Eigen::VectorXd &init_con,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &, double)> &F,
        const std::function<double(double)> &lbry,
        const std::function<double(double)> &rbry);

#endif //HEAT_SOLVER_H
