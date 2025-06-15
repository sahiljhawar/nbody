#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class NBodySimulation
{
private:
    double G;
    VectorXd masses;
    MatrixXd positions;
    MatrixXd velocities;
    int n_bodies;
    std::vector<MatrixXd> history;
    std::vector<double> time_steps;

public:
    NBodySimulation(const std::vector<double> &masses_vec,
                    const std::vector<std::vector<double>> &positions_vec,
                    const std::vector<std::vector<double>> &velocities_vec,
                    double G_val = 1.0);

    MatrixXd compute_forces();
    void step_rk4(double dt);
    double total_energy();
    void simulate(double t_end, double dt);

    std::vector<std::vector<std::vector<double>>> get_history() const;
    std::vector<std::vector<double>> get_positions() const;
    std::vector<std::vector<double>> get_velocities() const;
    std::vector<double> get_time_steps() const;
    std::vector<double> get_masses() const;
    int get_n_bodies() const;
    double get_G() const;
};