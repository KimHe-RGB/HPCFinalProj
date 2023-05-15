//
// Created by Keigo Ando on 5/13/23.
//

#include "heat_solver.h"

Eigen::VectorXd u_sol_f(const Eigen::VectorXd &x, long double t) {
    return (std::exp(-PI * PI * t / 4) * Eigen::cos((PI * x / 2).array()))
        .matrix();
}

// long double lbry_f(long double t) { return std::exp(-PI * PI * t / 4); }
long double lbry_f(long double t) { return 0; }

long double rbry_f(long double t) {
    return -PI / 2 * std::exp(-PI * PI * t / 4);
}

Eigen::VectorXd diffusivity_f(const Eigen::VectorXd &x) {
    return (2 + Eigen::cos(PI * x.array())).matrix();
}

Eigen::VectorXd force_f(const Eigen::VectorXd &x, long double t) {
    return (PI * PI / 2 * std::exp(-PI * PI * t / 4) *
            (-Eigen::sin(PI * x.array()) * Eigen::sin(PI / 2 * x.array()) +
             Eigen::cos(PI / 2 * x.array()) / 2 +
             Eigen::cos(PI * x.array()) * Eigen::cos(PI / 2 * x.array()) / 2))
        .matrix();
}

Eigen::VectorXd init_cond_f(const Eigen::VectorXd &x) {
    return Eigen::cos(PI * x.array() / 2);
}

// Spatial_Operator_Heat1D function
std::pair<Eigen::MatrixXd, Eigen::VectorXd> Spatial_Operator_Heat1D(
    const std::vector<long double> &xspan,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df) {
    long double x0 = xspan.front();
    long double hx = xspan[1];
    long double xfinal = xspan.back();

    int Nxs = static_cast<int>(
        (xfinal - x0) / hx);  // Calculate the number of elements in xs directly

    std::vector<long double> xs(Nxs);  // Initialize xs with the correct size
    for (int i = 0; i < Nxs; ++i) {
        xs[i] = x0 + hx / 2 + i * hx;
    }

    Eigen::VectorXd xr = Eigen::VectorXd::LinSpaced(xs.size() + 1, x0, xfinal);
    Eigen::VectorXd dxr = df(xr);

    Eigen::VectorXd r = Eigen::VectorXd::Zero(Nxs);

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(Nxs, Nxs);
    // Fill L with -dxr(i+1) - dxr(i) on the diagonal, dxr(i) on the upper
    // subdiagonal, and dxr(i+1) on the lower subdiagonal
    for (int i = 0; i < Nxs; ++i) {
        if (i > 0) {
            L(i, i - 1) = dxr[i];
        }
        L(i, i) = (i < Nxs - 1 ? -dxr[i + 1] - dxr[i] : -dxr[i]);
        if (i < Nxs - 1) {
            L(i, i + 1) = dxr[i + 1];
        }
    }

    long double L_corrector = -dxr[0] / (hx);
    L(0, 0) = -dxr[1];
    L(0, 1) = dxr[1];

    long double R_corrector = -dxr[dxr.size() - 1] / hx;
    L(Nxs - 1, Nxs - 1) = -dxr[dxr.size() - 2];
    L(Nxs - 1, Nxs - 2) = dxr[dxr.size() - 2];

    L /= (hx * hx);

    r[0] = L_corrector;
    r[Nxs - 1] = R_corrector;

    return {L, r};
}

Eigen::VectorXd Heat1D_IE_solver(
    const std::vector<long double> &xspan,
    const std::vector<long double> &tspan, const Eigen::VectorXd &init_con,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &, long double)>
        &F,
    const std::function<long double(long double)> &lbry,
    const std::function<long double(long double)> &rbry) {
    long double t0 = tspan.front();
    long double ht = tspan[1];
    long double tfinal = tspan.back();
    long double x0 = xspan.front();
    long double hx = xspan[1];
    long double xfinal = xspan.back();

    int Nxs = static_cast<int>((xfinal - x0) / hx);

    Eigen::VectorXd xs(Nxs);
    for (int i = 0; i < Nxs; ++i) {
        xs[i] = x0 + hx / 2 + i * hx;
    }

    long double t = t0;
    int i = 1;
    Eigen::VectorXd u = init_con;

    bool done = t >= tfinal;

    Eigen::MatrixXd L;
    Eigen::VectorXd r;
    std::tie(L, r) = Spatial_Operator_Heat1D(xspan, df);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(Nxs, Nxs);

    auto R = [lbry, rbry, r = r,
              N = xs.size()](long double t) mutable -> Eigen::VectorXd {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(N);
        res[0] = lbry(t);
        res[N - 1] = rbry(t);
        return res.cwiseProduct(r);
    };

    auto RHS = [F, &xs, R](long double t) mutable -> Eigen::VectorXd {
        return -F(xs, t) + R(t);
    };

    auto qr = (I - ht * L).colPivHouseholderQr();
    auto Fn = [I, L, RHS, qr](const Eigen::VectorXd &u, long double t,
                              long double ht) mutable -> Eigen::VectorXd {
        return qr.solve(u - ht * RHS(t));
    };

    /*
    auto Fn = [I, L, RHS](const Eigen::VectorXd &u, double t,
                          double ht) mutable -> Eigen::VectorXd {
        // return Fn = @(u, t, ht)(I - ht * L) \ (u - ht * RHS(t));
        return (I - ht * L).lu().solve(u - ht * RHS(t));
    };
    */

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
    const std::vector<long double> &xspan,
    const std::vector<long double> &tspan, const Eigen::VectorXd &init_con,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &df,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &, long double)>
        &F,
    const std::function<long double(long double)> &lbry,
    const std::function<long double(long double)> &rbry) {
    long double t0 = tspan.front();
    long double ht = tspan[1];
    long double tfinal = tspan.back();
    long double x0 = xspan.front();
    long double hx = xspan[1];
    long double xfinal = xspan.back();

    int Nxs = static_cast<int>((xfinal - x0) / hx);

    Eigen::VectorXd xs(Nxs);
    for (int i = 0; i < Nxs; ++i) {
        xs[i] = x0 + hx / 2 + i * hx;
    }

    long double t = t0;
    int i = 1;
    Eigen::VectorXd u = init_con;

    bool done = t >= tfinal;

    Eigen::MatrixXd L;
    Eigen::VectorXd r;
    std::tie(L, r) = Spatial_Operator_Heat1D(xspan, df);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(Nxs, Nxs);

    auto R = [lbry, rbry, r = r,
              N = xs.size()](long double t) mutable -> Eigen::VectorXd {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(N);
        res[0] = lbry(t);
        res[N - 1] = rbry(t);
        return res.cwiseProduct(r);
    };

    auto RHS = [F, &xs, R](long double t) mutable -> Eigen::VectorXd {
        return -F(xs, t) + R(t);
    };
    // auto QR = (I - ht * L / 2).colPivHouseholderQr();
    auto Fn = [I, L, RHS](const Eigen::VectorXd &u, long double t,
                          long double ht) mutable -> Eigen::VectorXd {
        return (I - ht * L / 2)
            .colPivHouseholderQr()
            .solve((I + ht * L / 2) * u - ht * RHS(t + ht / 2));
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