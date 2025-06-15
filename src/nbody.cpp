#include "nbody.h"
#include <cmath>
#include <iostream>
#include <iomanip>

NBodySimulation::NBodySimulation(const std::vector<double> &masses_vec,
                                 const std::vector<std::vector<double>> &positions_vec,
                                 const std::vector<std::vector<double>> &velocities_vec,
                                 double G_val)
    : G(G_val), n_bodies(masses_vec.size())
{

    if (positions_vec.size() != n_bodies || velocities_vec.size() != n_bodies)
    {
        throw std::invalid_argument("Size of masses array is " + std::to_string(masses_vec.size()) +
                                    ", positions array size is " + std::to_string(positions_vec.size()) +
                                    " and velocities array size is " + std::to_string(velocities_vec.size()));
    }

    std::cout << VectorXd(n_bodies).size() << " bodies initialized with G = " << G << std::endl;

    masses = VectorXd(n_bodies);
    for (int i = 0; i < n_bodies; ++i)
    {
        masses(i) = masses_vec[i];
    }

    positions = MatrixXd(n_bodies, 3);
    for (int i = 0; i < n_bodies; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            positions(i, j) = positions_vec[i][j];
        }
    }

    velocities = MatrixXd(n_bodies, 3);
    for (int i = 0; i < n_bodies; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            velocities(i, j) = velocities_vec[i][j];
        }
    }

    history.push_back(positions);
}

MatrixXd NBodySimulation::compute_forces()
{
    MatrixXd forces = MatrixXd::Zero(n_bodies, 3);

    for (int i = 0; i < n_bodies; ++i)
    {
        for (int j = i + 1; j < n_bodies; ++j)
        {
            Vector3d r_vec = positions.row(j) - positions.row(i);
            double r_mag_sq = r_vec.squaredNorm();

            if (r_mag_sq < 1e-20)
                continue;

            double r_mag = std::sqrt(r_mag_sq);
            double force_mag = G * masses(i) * masses(j) / r_mag_sq;
            Vector3d force_vec = (force_mag / r_mag) * r_vec;

            forces.row(i) += force_vec.transpose();
            forces.row(j) -= force_vec.transpose();
        }
    }

    return forces;
}

void NBodySimulation::step_rk4(double dt)
{
    MatrixXd pos0 = positions;
    MatrixXd vel0 = velocities;

    MatrixXd forces1 = compute_forces();
    MatrixXd acc1(n_bodies, 3);
    for (int i = 0; i < n_bodies; ++i)
    {
        acc1.row(i) = forces1.row(i) / masses(i);
    }
    MatrixXd dpos1 = vel0;
    MatrixXd dvel1 = acc1;

    positions = pos0 + 0.5 * dt * dpos1;
    velocities = vel0 + 0.5 * dt * dvel1;
    MatrixXd forces2 = compute_forces();
    MatrixXd acc2(n_bodies, 3);
    for (int i = 0; i < n_bodies; ++i)
    {
        acc2.row(i) = forces2.row(i) / masses(i);
    }
    MatrixXd dpos2 = velocities;
    MatrixXd dvel2 = acc2;

    positions = pos0 + 0.5 * dt * dpos2;
    velocities = vel0 + 0.5 * dt * dvel2;
    MatrixXd forces3 = compute_forces();
    MatrixXd acc3(n_bodies, 3);
    for (int i = 0; i < n_bodies; ++i)
    {
        acc3.row(i) = forces3.row(i) / masses(i);
    }
    MatrixXd dpos3 = velocities;
    MatrixXd dvel3 = acc3;

    positions = pos0 + dt * dpos3;
    velocities = vel0 + dt * dvel3;
    MatrixXd forces4 = compute_forces();
    MatrixXd acc4(n_bodies, 3);
    for (int i = 0; i < n_bodies; ++i)
    {
        acc4.row(i) = forces4.row(i) / masses(i);
    }
    MatrixXd dpos4 = velocities;
    MatrixXd dvel4 = acc4;

    positions = pos0 + dt / 6.0 * (dpos1 + 2 * dpos2 + 2 * dpos3 + dpos4);
    velocities = vel0 + dt / 6.0 * (dvel1 + 2 * dvel2 + 2 * dvel3 + dvel4);

    history.push_back(positions);
}

double NBodySimulation::total_energy()
{
    double ke = 0.0;
    for (int i = 0; i < n_bodies; ++i)
    {
        double v_sq = velocities.row(i).squaredNorm();
        ke += 0.5 * masses(i) * v_sq;
    }

    double pe = 0.0;
    for (int i = 0; i < n_bodies; ++i)
    {
        for (int j = i + 1; j < n_bodies; ++j)
        {
            Vector3d r_vec = positions.row(j) - positions.row(i);
            double r_mag = r_vec.norm();
            if (r_mag > 1e-10)
            {
                pe -= G * masses(i) * masses(j) / r_mag;
            }
        }
    }

    return ke + pe;
}

void NBodySimulation::simulate(double t_end, double dt)
{
    double t = 0.0;
    double initial_energy = total_energy();
    time_steps.clear();
    time_steps.push_back(t);

    while (t + dt <= t_end)
    {
        step_rk4(dt);

        t += dt;
        time_steps.push_back(t);
    }
}

std::vector<std::vector<std::vector<double>>> NBodySimulation::get_history() const
{
    std::vector<std::vector<std::vector<double>>> result;
    for (const auto &frame : history)
    {
        std::vector<std::vector<double>> frame_data;
        for (int i = 0; i < frame.rows(); ++i)
        {
            std::vector<double> particle_pos;
            for (int j = 0; j < frame.cols(); ++j)
            {
                particle_pos.push_back(frame(i, j));
            }
            frame_data.push_back(particle_pos);
        }
        result.push_back(frame_data);
    }
    return result;
}

std::vector<std::vector<double>> NBodySimulation::get_positions() const
{
    std::vector<std::vector<double>> result;
    for (int i = 0; i < positions.rows(); ++i)
    {
        std::vector<double> pos;
        for (int j = 0; j < positions.cols(); ++j)
        {
            pos.push_back(positions(i, j));
        }
        result.push_back(pos);
    }
    return result;
}

std::vector<std::vector<double>> NBodySimulation::get_velocities() const
{
    std::vector<std::vector<double>> result;
    for (int i = 0; i < velocities.rows(); ++i)
    {
        std::vector<double> vel;
        for (int j = 0; j < velocities.cols(); ++j)
        {
            vel.push_back(velocities(i, j));
        }
        result.push_back(vel);
    }
    return result;
}

std::vector<double> NBodySimulation::get_masses() const
{
    std::vector<double> result;
    for (int i = 0; i < masses.size(); ++i)
    {
        result.push_back(masses(i));
    }
    return result;
}

std::vector<double> NBodySimulation::get_time_steps() const
{
    return time_steps;
}

int NBodySimulation::get_n_bodies() const { return n_bodies; }
double NBodySimulation::get_G() const { return G; }