#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

const double PI = 3.14159265358979323846;

Eigen::VectorXd u_sol_f(const Eigen::VectorXd &x, double t) {
    return (std::exp(-PI * PI * t / 4) * (PI * x / 2).array().cos()).matrix();
}

double lbry_f(double t) {
    return std::exp(-PI * PI * t / 4);
}

double rbry_f(double t) {
    return -PI / 2 * std::exp(-PI * PI * t / 4);
}

Eigen::VectorXd diffusivity_f(const Eigen::VectorXd &x) {
    return (2 + Eigen::cos(PI * x.array())).matrix();
}

Eigen::VectorXd force_f(const Eigen::VectorXd &x, double t) {
    return (PI * PI / 2 * std::exp(-PI * PI * t / 4) *
            (-Eigen::sin(PI * x.array()) * Eigen::sin(PI / 2 * x.array()) + Eigen::cos(PI / 2 * x.array()) / 2 +
             Eigen::cos(PI * x.array()) * Eigen::cos(PI / 2 * x.array()) / 2)).matrix();
}

Eigen::VectorXd init_cond_f(const Eigen::VectorXd &x) {
    return Eigen::cos(PI * x.array() / 2);
}

// Spatial_Operator_Heat1D function
std::pair<Eigen::MatrixXd, Eigen::VectorXd> Spatial_Operator_Heat1D(
        const std::vector<double> &xspan,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df) {
    double x0 = xspan.front();
    double hx = xspan[1] - xspan[0];
    double xfinal = xspan.back();

    int Nxs = static_cast<int>((xfinal - x0) / hx); // Calculate the number of elements in xs directly

    std::vector<double> xs(Nxs); // Initialize xs with the correct size
    for (int i = 0; i < Nxs; ++i) {
        xs[i] = x0 + hx / 2 + i * hx;
    }

    Eigen::VectorXd xr = Eigen::VectorXd::LinSpaced(xs.size() + 1, x0, xfinal);
    Eigen::VectorXd dxr = df(xr);

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(Nxs, Nxs);
    Eigen::VectorXd r = Eigen::VectorXd::Zero(Nxs);

    // Fill L with -dxr(i+1) - dxr(i) on the diagonal, dxr(i) on the upper subdiagonal, and dxr(i+1) on the lower subdiagonal
    for (int i = 0; i < Nxs; ++i) {
        if (i > 0) {
            L(i, i - 1) = dxr[i];
        }
        L(i, i) = (i < Nxs - 1 ? -dxr[i + 1] - dxr[i] : -dxr[i]);
        if (i < Nxs - 1) {
            L(i, i + 1) = dxr[i + 1];
        }
    }

    double L_corrector = -2 * dxr[0] / (hx * hx);
    L(0, 0) = -2 * dxr[0] - dxr[1];
    L(0, 1) = dxr[1];

    double R_corrector = -dxr[dxr.size() - 1] / hx;
    L(Nxs - 1, Nxs - 1) = -dxr[dxr.size() - 2];
    L(Nxs - 1, Nxs - 2) = dxr[dxr.size() - 2];

    L /= (hx * hx);

    r[0] = L_corrector;
    r[Nxs - 1] = R_corrector;

    return {L, r};
}

// Heat1D_RK3_solver function
Eigen::VectorXd Heat1D_RK3_solver(
        const std::vector<double> &xspan,
        const std::vector<double> &tspan,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &icf,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &, double)> &F,
        const std::function<double(double)> &lbry,
        const std::function<double(double)> &rbry) {

    double t0 = tspan.front();
    double ht = tspan[1] - tspan[0];
    double tfinal = tspan.back();
    double x0 = xspan.front();
    double hx = xspan[1] - xspan[0];
    double xfinal = xspan.back();

    int Nxs = static_cast<int>((xfinal - x0) / hx); // Calculate the number of elements in xs directly

    Eigen::VectorXd xs(Nxs); // Initialize xs with the correct size
    for (int i = 0; i < Nxs; ++i) {
        xs[i] = x0 + hx / 2 + i * hx;
    }

    //Eigen::VectorXd xr = Eigen::VectorXd::LinSpaced(xs.size(), xs.front(), xs.back());
    Eigen::VectorXd u = icf(xs);


    Eigen::MatrixXd L;
    Eigen::VectorXd r;
    std::tie(L, r) = Spatial_Operator_Heat1D(xspan, df);

    auto R = [lbry, rbry, r = r, N = xs.size()](double t) mutable -> Eigen::VectorXd {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(N);
        res[0] = lbry(t);
        res[N - 1] = rbry(t);
        return res.cwiseProduct(r);
    };

    auto RHS = [F, &xs, R](double t) mutable -> Eigen::VectorXd {
        return F(xs, t) - R(t);
    };

    Eigen::VectorXd hey = F(xs, 1.0) - R(1.0);

    auto Fn = [L, RHS](const Eigen::VectorXd &u, double t) mutable -> Eigen::MatrixXd {
        return L * u + RHS(t);
    };

    int i = 0;
    for (double t = t0; t < tfinal; t += ht) {
        if (t + ht > tfinal) {
            ht = tfinal - t;
        }
        Eigen::VectorXd k1 = Fn(u, t);
        Eigen::VectorXd k2 = Fn(u + ht * k1 / 2, t + ht / 2);
        Eigen::VectorXd k3 = Fn(u - ht * k1 + 2 * ht * k2, t + ht);
        u += ht * (k1 + 4 * k2 + k3) / 6;
    }
    return u;
}

Eigen::VectorXd Heat1D_IE_solver(
        const std::vector<double> &xspan,
        const std::vector<double> &tspan,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &icf,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &, double)> &F,
        const std::function<double(double)> &lbry,
        const std::function<double(double)> &rbry
) {

    double t0 = tspan.front();
    double ht = tspan[1] - tspan[0];
    double tfinal = tspan.back();
    double x0 = xspan.front();
    double hx = xspan[1] - xspan[0];
    double xfinal = xspan.back();

    int Nxs = static_cast<int>((xfinal - x0) / hx);

    Eigen::VectorXd xs(Nxs);
    for (int i = 0; i < Nxs; ++i) {
        xs[i] = x0 + hx / 2 + i * hx;
    }

    double t = t0;
    int i = 1;
    Eigen::VectorXd u = icf(xs);

    bool done = t >= tfinal;

    Eigen::MatrixXd L;
    Eigen::VectorXd r;
    std::tie(L, r) = Spatial_Operator_Heat1D(xspan, df);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(Nxs, Nxs);

    auto R = [lbry, rbry, r = r, N = xs.size()](double t) mutable -> Eigen::VectorXd {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(N);
        res[0] = lbry(t);
        res[N - 1] = rbry(t);
        return res.cwiseProduct(r);
    };

    auto RHS = [F, &xs, R](double t) mutable -> Eigen::VectorXd {
        return -F(xs, t) + R(t);
    };

    auto Fn = [I, L, RHS](const Eigen::VectorXd &u, double t, double ht) mutable -> Eigen::VectorXd {
        return (I - ht * L).colPivHouseholderQr().solve(u - ht * RHS(t));
    };

    while (!done) {
        ++i;
        if (t + ht >= tfinal) {
            ht = tfinal - t;
            done = true;
        }
        u = Fn(u, t, ht);
        t += ht;
    }
    return u;
}

Eigen::VectorXd Heat1D_CN_solver(
        const std::vector<double> &xspan,
        const std::vector<double> &tspan,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &icf,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
        const std::function<Eigen::VectorXd(const Eigen::VectorXd &, double)> &F,
        const std::function<double(double)> &lbry,
        const std::function<double(double)> &rbry
) {

    double t0 = tspan.front();
    double ht = tspan[1] - tspan[0];
    double tfinal = tspan.back();
    double x0 = xspan.front();
    double hx = xspan[1] - xspan[0];
    double xfinal = xspan.back();

    int Nxs = static_cast<int>((xfinal - x0) / hx);

    Eigen::VectorXd xs(Nxs);
    for (int i = 0; i < Nxs; ++i) {
        xs[i] = x0 + hx / 2 + i * hx;
    }

    double t = t0;
    int i = 1;
    Eigen::VectorXd u = icf(xs);

    bool done = t >= tfinal;

    Eigen::MatrixXd L;
    Eigen::VectorXd r;
    std::tie(L, r) = Spatial_Operator_Heat1D(xspan, df);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(Nxs, Nxs);

    auto R = [lbry, rbry, r = r, N = xs.size()](double t) mutable -> Eigen::VectorXd {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(N);
        res[0] = lbry(t);
        res[N - 1] = rbry(t);
        return res.cwiseProduct(r);
    };

    auto RHS = [F, &xs, R](double t) mutable -> Eigen::VectorXd {
        return -F(xs, t) + R(t);
    };

    auto Fn = [I, L, RHS](const Eigen::VectorXd &u, double t, double ht) mutable -> Eigen::VectorXd {
        return (I - ht * L / 2).colPivHouseholderQr().solve((I + ht * L / 2) * u - ht * RHS(t + ht / 2));
    };

    while (!done) {
        ++i;
        if (t + ht >= tfinal) {
            ht = tfinal - t;
            done = true;
        }
        u = Fn(u, t, ht);
        t += ht;
    }
    return u;
}

int main() {
    double CFL = 1.0 / 6;
    std::vector<int> range = {-5, -4, -3, -2, -1};//{-8, -7, -6, -5, -4, -3, -2, -1};
    Eigen::VectorXd hxs(range.size()), hts(range.size());
    for (int i = 0; i < range.size(); ++i) {
        hxs(i) = std::pow(2, range[i]);
        hts(i) = std::pow(hxs(i), 2) * CFL;
    }
    double x0 = 0, xfinal = 1;
    double t0 = 0, tfinal = 2;
    std::vector<double> err_1s(range.size()), err_2s(range.size()), err_infs(range.size());

    for (int i = 0; i < range.size(); ++i) {
        double hx = hxs(i), ht = hts(i);
        Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced((xfinal - x0 - hx) / hx + 1, x0 + hx / 2, xfinal - hx / 2);
        int Nxs = xs.size();

        std::vector<double> xspan = {x0, hx, xfinal};
        std::vector<double> tspan = {t0, ht, tfinal};

        Eigen::VectorXd u = Heat1D_IE_solver(xspan, tspan, init_cond_f, diffusivity_f, force_f, lbry_f, rbry_f);
        Eigen::VectorXd real_u = u_sol_f(xs, tfinal);
        int M = 1024;

        if (u.size() != real_u.size()) {
            std::cerr << "Error: For " << i << "th iteration, Dimensions of u and real_u do not match. u size: "
                      << u.size() << ", real_u size: " << real_u.size() << std::endl;
            return -1;  // Or handle the error in some other way
        }

        Eigen::VectorXd dif = u - real_u;

        int hey = 3;
        err_1s[i] = (u - real_u).lpNorm<1>();
        //Eigen::VectorXd u_int(M), real_u_int(M);
        // We need to implement an interpolation function here
        err_1s[i] = (u - real_u).lpNorm<1>();
        err_2s[i] = (u - real_u).norm();
        err_infs[i] = (u - real_u).lpNorm<Eigen::Infinity>();
    }

    // Print or use the error values
    for (int i = 0; i < range.size(); ++i) {
        std::cout << "For hx = " << hxs(i) << ", the errors are:\n";
        std::cout << "L1 error: " << err_1s[i] << "\n";
        std::cout << "L2 error: " << err_2s[i] << "\n";
        std::cout << "Infinity norm error: " << err_infs[i] << "\n";
        std::cout << std::endl;
    }
    return 0;
}
