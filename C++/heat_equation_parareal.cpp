#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>

using namespace std;
using namespace Eigen;

// Function to set up the initial condition
VectorXd initial_condition(int Nx, double dx) {
    VectorXd u0(Nx);
    for (int i = 0; i < Nx; i++) {
        double x = i * dx;
        u0(i) = sin(M_PI * x);
    }
    return u0;
}

VectorXd exact_solution(const VectorXd& x, double t, double alpha) {
    VectorXd u = (x*M_PI).array().sin();
    u *= exp(-alpha * M_PI * M_PI * t);
    return u;
}

// Coarse propagator (explicit Euler)
VectorXd explicit_euler(const VectorXd &u, double dt, double dx, double alpha) {
    int Nx = u.size();
    VectorXd u_new(Nx);
    u_new(0) = u(0); // Boundary condition
    u_new(Nx - 1) = u(Nx - 1); // Boundary condition
    for (int i = 1; i < Nx - 1; i++) {
        u_new(i) = u(i) + dt * alpha * (u(i + 1) - 2 * u(i) + u(i - 1)) / (dx * dx);
    }
    return u_new;
}

// Fine propagator1 (Runge Kutta 4)
VectorXd runge_kutta_4(const VectorXd &u, double dt, double dx, double alpha) {
    auto rhs = [&](const VectorXd &u) {
        VectorXd du(u.size());
        du(0) = 0; // Boundary condition
        du(u.size() - 1) = 0; // Boundary condition
        for (int i = 1; i < u.size() - 1; i++) {
            du(i) = alpha * (u(i + 1) - 2 * u(i) + u(i - 1)) / (dx * dx);
        }
        return du;
    };

    VectorXd k1 = rhs(u);
    VectorXd k2 = rhs(u + 0.5 * dt * k1);
    VectorXd k3 = rhs(u + 0.5 * dt * k2);
    VectorXd k4 = rhs(u + dt * k3);

    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}

// Fine propagator (Crank-Nicolson)
VectorXd crank_nicolson(const VectorXd &u, double dt, double dx, double alpha) {
    int Nx = u.size();
    VectorXd u_new(Nx);
    VectorXd rhs(Nx);
    MatrixXd A(Nx, Nx);
    A.setZero();

    // Set up the tridiagonal matrix for Crank-Nicolson
    double r = alpha * dt / (2 * dx * dx);
    for (int i = 1; i < Nx - 1; i++) {
        A(i, i - 1) = -r;
        A(i, i) = 1 + 2 * r;
        A(i, i + 1) = -r;
    }
    A(0, 0) = 1; // Boundary condition
    A(Nx - 1, Nx - 1) = 1; // Boundary condition

    // Set up the right-hand side for Crank-Nicolson
    rhs(0) = u(0); // Boundary condition
    rhs(Nx - 1) = u(Nx - 1); // Boundary condition
    for (int i = 1; i < Nx - 1; i++) {
        rhs(i) = r * u(i - 1) + (1 - 2 * r) * u(i) + r * u(i + 1);
    }

    // Solve the linear system
    u_new = A.partialPivLu().solve(rhs);
    return u_new;
}

VectorXd crank_nicolson_solution(const VectorXd &u0, double dt, double dx, double alpha, int num_time_steps) {
    VectorXd u = u0;
    for (int i = 0; i < num_time_steps; i++) {
        u = crank_nicolson(u, dt, dx, alpha);
    }
    return u;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <Nx> <Nt> <num_processors>" << endl;
        return 1;
    }

    int Nx = stoi(argv[1]); // Number of spatial points
    int Nt = stoi(argv[2]); // Number of time steps
    int num_processors = stoi(argv[3]); // Number of available processors
    double L = 1.0; // Length of the domain
    double T = 0.1; // Total time
    double alpha = 0.01; // Thermal diffusivity

    double dx = L / (Nx - 1);
    double dt = T / Nt;

    VectorXd u = initial_condition(Nx, dx);

    int num_subintervals = num_processors; // Divide the time domain into subintervals
    int num_iterations = 10000; // Maximum number of parareal iterations
    double tol = 1e-6; // Convergence tolerance

    auto parareal_start = std::chrono::high_resolution_clock::now();

    // Parareal algorithm
    vector<VectorXd> U_coarse(num_subintervals + 1, VectorXd::Zero(Nx));
    vector<VectorXd> U_fine(num_subintervals + 1, VectorXd::Zero(Nx));

    // Initialize the solution with the coarse propagator
    U_coarse[0] = u;
    for (int i = 0; i < num_subintervals; i++) {
        U_coarse[i + 1] = explicit_euler(U_coarse[i], dt, dx, alpha);
    }

    int iter = 0;
    double max_diff;
    do {
        max_diff = 0;
        U_fine[0] = U_coarse[0];

        // Solve each subinterval with the fine propagator in parallel
        // This code runs in serial. To run in parallel, you need to use
        // parallel programming techniques such as OpenMP or MPI.)
        for (int i = 0; i < num_subintervals; i++) {
            //the solver function of crank_nicolson
            U_fine[i + 1] = crank_nicolson(U_fine[i], dt, dx, alpha);
            //the solver function of runge_kutta_4
            //U_fine[i + 1] = runge_kutta_4(U_fine[i], dt, dx, alpha);
            double diff = (U_fine[i + 1] - U_coarse[i + 1]).norm();
            max_diff = max(max_diff, diff);

            // Update the initial condition for the next iteration
            U_coarse[i + 1] = U_fine[i + 1] + (U_coarse[i + 1] - U_coarse[i]);
        }
        iter++;
    } while (iter < num_iterations && max_diff > tol);

    VectorXd parareal_solution = U_fine[num_subintervals];

    auto parareal_end = chrono::high_resolution_clock::now();
    auto parareal_duration = chrono::duration_cast<chrono::milliseconds>(parareal_end - parareal_start).count();

    // Compute the exact solution
    VectorXd x = VectorXd::LinSpaced(Nx, 0, 1);
    VectorXd exact_sol = exact_solution(x, T, alpha);

    // Calculate the L2-norm of the difference between the numerical solutions and the exact solution
    double parareal_error = (parareal_solution - exact_sol).norm();

    
    // Output the errors
    cout << "Parareal error (L2-norm): " << parareal_error << endl;
    cout << "Parareal computational time (ms): " << parareal_duration << endl;

    // Calculate the Crane-Nicolson solution
    auto crank_nicolson_start = chrono::high_resolution_clock::now();
    VectorXd crank_nicolson_sol = crank_nicolson_solution(u, dt, dx, alpha, Nt);
    auto crank_nicolson_end = chrono::high_resolution_clock::now();
    auto crank_nicolson_duration = chrono::duration_cast<std::chrono::milliseconds>(crank_nicolson_end - crank_nicolson_start).count();

    //Calculate the L2-norm of the difference between the numerical solutions and the exact solution
    double crank_nicolson_error = (crank_nicolson_sol - exact_sol).norm();

    // Output the errors and computatinal time of crank-nicolson
    cout << "Crank-Nicolson error (L2-norm): " << crank_nicolson_error << endl;
    cout << "Crank-Nicolson computational time (ms): " << crank_nicolson_duration << endl;

    /*
    // Calculate 4th order Runge-Kutta solution
    auto runge_kutta_start = chrono::high_resolution_clock::now();
    VectorXd runge_kutta_sol = u;
    for (int i = 0; i < Nt; i++) {
        runge_kutta_sol = runge_kutta_4(runge_kutta_sol, dt, dx, alpha);
    }
    auto runge_kutta_end = chrono::high_resolution_clock::now();
    auto runge_kutta_duration = chrono::duration_cast<std::chrono::milliseconds>(runge_kutta_end - runge_kutta_start).count();

    // Calculate the L2-norm of the difference between the numerical solutions and the exact solution
    double runge_kutta_error = (runge_kutta_sol - exact_sol).norm();

    // Output the errors
    cout << "Runge-Kutta error (L2-norm): " << runge_kutta_error << endl;
    cout << "Runge-Kutta computational time (ms): " << runge_kutta_duration << endl;
    */

    // Output the solutions to a CSV file
    ofstream output_file("solutions.csv");
    output_file << "x,Exact,Parareal,Approximation" << std::endl;
    for (int i = 0; i < Nx; i++) {
        output_file << x(i) << "," << exact_sol(i) << "," << parareal_solution(i) << "," << crank_nicolson_sol(i) << endl;
    }
    output_file.close();

    return 0;
}
