#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

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

VectorXd exact_solution(const VectorXd &x, double t, double alpha) {
    VectorXd u = (x * M_PI).array().sin();
    u *= exp(-alpha * M_PI * M_PI * t);
    return u;
}

// Coarse propagator (explicit Euler)
VectorXd explicit_euler(const VectorXd &u, double dt, double dx, double alpha) {
    int Nx = u.size();
    VectorXd u_new(Nx);
    u_new(0) = u(0);            // Boundary condition
    u_new(Nx - 1) = u(Nx - 1);  // Boundary condition
    for (int i = 1; i < Nx - 1; i++) {
        u_new(i) =
            u(i) + dt * alpha * (u(i + 1) - 2 * u(i) + u(i - 1)) / (dx * dx);
    }
    return u_new;
}

vector<VectorXd> explicit_euler(const VectorXd &u, double dt, double dx,
                                double alpha, int num_time_steps) {
    vector<VectorXd> solutions(num_time_steps + 1);
    solutions[0] = u;
    for (int step = 0; step < num_time_steps; ++step) {
        solutions[step + 1] = explicit_euler(solutions[step], dt, dx, alpha);
    }
    return solutions;
}

// Fine propagator (Crank-Nicolson)
VectorXd crank_nicolson(const VectorXd &u, double dt, double dx, double alpha) {
    // ... (no changes here) ...
}

vector<VectorXd> crank_nicolson_solution(const VectorXd &u0, double dt,
                                         double dx, double alpha,
                                         int num_time_steps) {
    vector<VectorXd> solutions(num_time_steps + 1);
    solutions[0] = u0;
    for (int i = 0; i < num_time_steps; i++) {
        solutions[i + 1] = crank_nicolson(solutions[i], dt, dx, alpha);
    }
    return solutions;
}

int main() {
    // Set up the problem parameters
    int Nx = 101;
    double L = 1.0;
    double T = 0.1;
    double alpha = 0.01;
    int Nt = 100;

    double dx = L / (Nx - 1);
    double dt = T / Nt;

    // Initialize the solution with the coarse propagator
    U_coarse[0] = u;
    vector<VectorXd> explicit_euler_solutions =
        explicit_euler(u0, dt, dx, alpha, Nt);

    // Calculate and store the solutions of explicit euler for each time step
    VectorXd u0 = initial_condition(Nx, dx);
    vector<VectorXd> explicit_euler_solutions =
        explicit_euler(u0, dt, dx, alpha, Nt);

    // Crank-Nicolson solution
    vector<VectorXd> crank_nicolson_solutions =
        crank_nicolson_solution(u, dt, dx, alpha, Nt);
    VectorXd crank_nicolson_solution = crank_nicolson_solutions[Nt];

    // Compute the exact solution at time T
    VectorXd x = VectorXd::LinSpaced(Nx, 0, L);
    VectorXd exact_sol = exact_solution(x, T, alpha);

    // Calculate the L2-norm of the difference between the numerical solutions
    // and the exact solution
    VectorXd euler_solution = explicit_euler_solutions[Nt];
    double euler_error = (euler_solution - exact_sol).norm();

    // Output the errors
    cout << "Explicit Euler error (L2-norm): " << euler_error << endl;

    // Calculate the L2-norm of the difference between the Crank-Nicolson
    // solution and the exact solution at time T
    double crank_nicolson_error =
        (crank_nicolson_solution - exact_solution).norm();

    // Output the error
    cout << "Crank-Nicolson error (L2-norm): " << crank_nicolson_error << endl;

    return 0;
}
