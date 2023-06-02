// g++ -std=c++17 -fopenmp -O3 -I/usr/include/eigen3 parareal_OMP.cpp -o
// parareal

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iomanip>  // for std::setprecision

#include "heat_solver.cpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <hx> <ht>\n";
        return 1;
    }
    // Variable setting
    // double CFL = 1.0 / 6;
    // std::vector<int> range = {-7, -6, -5, -4, -3, -2, -1};
    std::vector<int> range = {1, 2, 4, 8, 16, 32, 64};
    /*
    Eigen::VectorXd hxs(range.size()), hts(range.size());
    for (int i = 0; i < range.size(); ++i) {
        hxs(i) = std::pow(2, range[i]);
        hts(i) = hxs(i);
    }
    */
    long double hx = std::pow(2, -1 * std::stold(argv[1]));
    long double ht = std::pow(2, -1 * std::stold(argv[2]));
    long double x0 = 0, xfinal = 1;
    long double t0 = 0, tfinal = std::pow(2, 7);

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            printf("Number of threads = %d\n", omp_get_max_threads());
        }
    }

    // int max_iterations = 8;  // Maximum number of parareal iterations
    // double tol = 1e-5;  // Convergence tolerance

    std::vector<long double> err_1s_PR(range.size()), err_2s_PR(range.size()),
        err_infs_PR(range.size());
    long double err_1s_IE, err_2s_IE, err_infs_IE;
    long double err_1s_CN, err_2s_CN, err_infs_CN;

    // Vectors to store execution times
    std::vector<double> exec_times_PR(range.size());
    double exec_times_IE, exec_times_CN;

    Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(
        (xfinal - x0 - hx) / hx + 1, x0 + hx / 2, xfinal - hx / 2);
    int Nxs = xs.size();

    // Make a vector to store the final solutions for each range iteration
    std::vector<Eigen::VectorXd> final_U_solutions(range.size(),
                                                   Eigen::VectorXd::Zero(Nxs));

    Eigen::VectorXd init_con = init_cond_f(xs);
    std::vector<long double> xspan = {x0, hx, xfinal};
    std::vector<long double> tspan = {t0, ht, tfinal};

    std::vector<Eigen::VectorXd> res(range.size() + 1,
                                     Eigen::VectorXd::Zero(Nxs));

    // real solution
    Eigen::VectorXd real_u = u_sol_f(xs, tfinal);

    for (int i = 0; i < range.size(); ++i) {
        // set variables
        // double hx = hxs(i), ht = hts(i);
        int num_iterations = 2;  // range[i];

        // Variables for parallelize
        int num_processors = range[i];  // 5;
        int num_subintervals =
            num_processors;  // Divide the time domain into subintervals

        // variables for parareal
        std::vector<Eigen::VectorXd> U_coarse(num_subintervals + 1,
                                              Eigen::VectorXd::Zero(Nxs));
        std::vector<Eigen::VectorXd> U_solution(num_subintervals + 1,
                                                Eigen::VectorXd::Zero(Nxs));
        std::vector<Eigen::VectorXd> U_solution_prev(
            num_subintervals + 1, Eigen::VectorXd::Zero(Nxs));
        std::vector<Eigen::VectorXd> U_fine(num_subintervals + 1,
                                            Eigen::VectorXd::Zero(Nxs));
        Eigen::VectorXd U_coarse_new;

        // Start timing for PR_solver
        auto start_PR = std::chrono::high_resolution_clock::now();

        U_coarse[0] = init_con;
        U_solution[0] = init_con;
        U_fine[0] = init_con;

        for (int i = 0; i < num_subintervals; i++) {
            long double t_s = (i + 1) * (tfinal - t0) / num_subintervals + t0;
            long double t_e = t_s + (tfinal - t0) / num_subintervals;
            long double t_i = (tfinal - t0) / num_subintervals / 5;
            std::vector<long double> tspan_mini = {t_s, ht, t_e};
            U_coarse[i + 1] =
                Heat1D_IE_solver(xspan, tspan_mini, U_coarse[i], diffusivity_f,
                                 force_f, lbry_f, rbry_f);
        }
        U_solution = U_coarse;

        int iter = 0;
        // double diff;
        // double max_diff;

        do {
            //    max_diff = 0;
// OpenMP parallel for loop
#pragma omp parallel for shared(U_fine, U_solution, xspan, tfinal, t0, \
                                    num_subintervals, ht) private(i)   \
    schedule(static)
            for (int i = 0; i < num_subintervals; i++) {
                long double t_s = i * (tfinal - t0) / num_subintervals + t0;
                long double t_e = t_s + (tfinal - t0) / num_subintervals;
                std::vector<long double> tspan_mini2 = {t_s, ht, t_e};
                U_fine[i + 1] =
                    Heat1D_CN_solver(xspan, tspan_mini2, U_solution[i],
                                     diffusivity_f, force_f, lbry_f, rbry_f);
            }

            for (int i = 0; i < num_subintervals; i++) {
                long double t_s = i * (tfinal - t0) / num_subintervals + t0;
                long double t_e = t_s + (tfinal - t0) / num_subintervals;
                std::vector<long double> tspan_mini2 = {t_s, ht * 5, t_e};
                U_solution_prev[i + 1] = U_solution[i + 1];
                U_coarse_new =
                    Heat1D_IE_solver(xspan, tspan_mini2, U_solution[i],
                                     diffusivity_f, force_f, lbry_f, rbry_f);
                U_solution[i + 1] =
                    U_fine[i + 1] + U_coarse_new - U_coarse[i + 1];
                U_coarse[i + 1] = U_coarse_new;
                // diff = (U_solution[i + 1] - U_solution_prev[i + 1]).norm();
                // max_diff = std::max(max_diff, diff);
            }
            /*
            if (iter % 2 == 0 || iter == 1) {
                std::cout << "Iteration " << iter << ": " << max_diff
                          << std::endl;
            }
            */
            iter++;
        } while (iter < num_iterations);

        final_U_solutions[i] = U_solution[num_subintervals];

        // Stop timing for PR_solver and calculate the duration
        auto stop_PR = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_PR = stop_PR - start_PR;
        exec_times_PR[i] = elapsed_PR.count();

        err_1s_PR[i] = (U_solution[num_subintervals] - real_u).lpNorm<1>();
        err_2s_PR[i] = (U_solution[num_subintervals] - real_u).norm();
        err_infs_PR[i] =
            (U_solution[num_subintervals] - real_u).lpNorm<Eigen::Infinity>();
    }

    // Start timing for IE_solver
    auto start_IE = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd u_IE = Heat1D_IE_solver(
        xspan, tspan, init_con, diffusivity_f, force_f, lbry_f, rbry_f);

    // Stop timing for IE_solver and calculate the duration
    auto stop_IE = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_IE = stop_IE - start_IE;
    exec_times_IE = elapsed_IE.count();

    err_1s_IE = (u_IE - real_u).lpNorm<1>();
    err_2s_IE = (u_IE - real_u).norm();
    err_infs_IE = (u_IE - real_u).lpNorm<Eigen::Infinity>();

    // Start timing for CN_solver
    auto start_CN = std::chrono::high_resolution_clock::now();

    Eigen::VectorXd u_CN = Heat1D_CN_solver(
        xspan, tspan, init_con, diffusivity_f, force_f, lbry_f, rbry_f);

    // Stop timing for CN_solver and calculate the duration
    auto stop_CN = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_CN = stop_CN - start_CN;
    exec_times_CN = elapsed_CN.count();

    err_1s_CN = (u_CN - real_u).lpNorm<1>();
    err_2s_CN = (u_CN - real_u).norm();
    err_infs_CN = (u_CN - real_u).lpNorm<Eigen::Infinity>();

    // Print execution times and errors

    for (int i = 0; i < range.size(); ++i) {
        std::cout << "Num core: : " << range[i] << " hx: " << hx
                  << " ht: " << ht << " T: " << tfinal << '\n';

        std::cout << "PR solver: " << exec_times_PR[i] << " seconds\n";
        // std::cout << "PR L1 error: " << err_1s_PR[i] << "\n";
        // std::cout << "PR L2 error: " << err_2s_PR[i] << "\n";
        std::cout << "PR Infinity norm error: " << err_infs_PR[i] << "\n";
        std::cout << std::endl;
    }
    std::cout << "IE solver: " << exec_times_IE << " seconds\n";
    // std::cout << "IE L1 error: " << err_1s_IE[i] << "\n";
    // std::cout << "IE L2 error: " << err_2s_IE[i] << "\n";
    std::cout << "IE Infinity norm error: " << err_infs_IE << "\n";

    std::cout << "CN solver: " << exec_times_CN << " seconds\n";
    // std::cout << "CN L1 error: " << err_1s_CN[i] << "\n";
    // std::cout << "CN L2 error: " << err_2s_CN[i] << "\n";
    std::cout << "CN Infinity norm error: " << err_infs_CN << "\n";

    std::ofstream file;
    file.open("time_OMP.csv");
    file << std::setprecision(15);

    file << "number of iterations,";
    for (int j = 0; j < range.size(); ++j) {
        file << range[j];
        if (j != range.size() - 1) {  // not the last element
            file << ",";
        }
    }
    file << "\n";

    file << "time " << ht << ",";
    for (int j = 0; j < range.size(); ++j) {
        file << exec_times_PR[j];
        if (j != range.size() - 1) {  // not the last element
            file << ",";
        }
    }
    file << "\n";

    file.close();

    return 0;
}
