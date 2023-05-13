#include "heat_solver.h"
#include "heat_solver.cpp"
#include <chrono>

int main() {
    double CFL = 1.0 / 6;
    std::vector<int> range = {-6, -5, -4, -3, -2, -1};
    Eigen::VectorXd hxs(range.size()), hts(range.size());
    for (int i = 0; i < range.size(); ++i) {
        hxs(i) = std::pow(2, range[i]);
        hts(i) = std::pow(hxs(i), 2) * CFL;
    }
    double x0 = 0, xfinal = 1;
    double t0 = 0, tfinal = 2;
    std::vector<double> err_1s_IE(range.size()), err_2s_IE(range.size()), err_infs_IE(range.size());
    std::vector<double> err_1s_CN(range.size()), err_2s_CN(range.size()), err_infs_CN(range.size());

    // Vectors to store execution times
    std::vector<double> exec_times_IE(range.size()), exec_times_CN(range.size());

    for (int i = 0; i < range.size(); ++i) {
        double hx = hxs(i), ht = hts(i);
        Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced((xfinal - x0 - hx) / hx + 1, x0 + hx / 2, xfinal - hx / 2);
        int Nxs = xs.size();

        Eigen::VectorXd init_con = init_cond_f(xs);
        std::vector<double> xspan = {x0, hx, xfinal};
        std::vector<double> tspan = {t0, ht, tfinal};

        Eigen::VectorXd real_u = u_sol_f(xs, tfinal);

        // Start timing for IE_solver
        auto start_IE = std::chrono::high_resolution_clock::now();

        Eigen::VectorXd u_IE = Heat1D_IE_solver(xspan, tspan, init_con, diffusivity_f, force_f, lbry_f, rbry_f);

        // Stop timing for IE_solver and calculate the duration
        auto stop_IE = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_IE = stop_IE - start_IE;
        exec_times_IE[i] = elapsed_IE.count();

        err_1s_IE[i] = (u_IE - real_u).lpNorm<1>();
        err_2s_IE[i] = (u_IE - real_u).norm();
        err_infs_IE[i] = (u_IE - real_u).lpNorm<Eigen::Infinity>();

        // Start timing for CN_solver
        auto start_CN = std::chrono::high_resolution_clock::now();

        Eigen::VectorXd u_CN = Heat1D_CN_solver(xspan, tspan, init_con, diffusivity_f, force_f, lbry_f, rbry_f);

        // Stop timing for CN_solver and calculate the duration
        auto stop_CN = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_CN = stop_CN - start_CN;
        exec_times_CN[i] = elapsed_CN.count();

        err_1s_CN[i] = (u_CN - real_u).lpNorm<1>();
        err_2s_CN[i] = (u_CN - real_u).norm();
        err_infs_CN[i] = (u_CN - real_u).lpNorm<Eigen::Infinity>();
    }

    // Print execution times and errors
    for (int i = 0; i < range.size(); ++i) {
        std::cout << "For hx = " << hxs(i) << ", the results are:\n";

        std::cout << "IE solver: " << exec_times_IE[i] << " seconds\n";
        std::cout << "IE L1 error: " << err_1s_IE[i] << "\n";
        std::cout << "IE L2 error: " << err_2s_IE[i] << "\n";
        std::cout << "IE Infinity norm error: " << err_infs_IE[i] << "\n";

        std::cout << "CN solver: " << exec_times_CN[i] << " seconds\n";
        std::cout << "CN L1 error: " << err_1s_CN[i] << "\n";
        std::cout << "CN L2 error: " << err_2s_CN[i] << "\n";
        std::cout << "CN Infinity norm error: " << err_infs_CN[i] << "\n";

        std::cout << std::endl;
    }
    return 0;
}
