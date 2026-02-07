// [[DEPRECATED]] Demo for genotype data generation and visualization

#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <matplot/matplot.h>


namespace plt = matplot;

// --- Helper Functions ---
inline std::string format_double(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::vector<std::vector<double>> eigen_to_vector(const Eigen::MatrixXd& mat) {
    std::vector<std::vector<double>> vec(mat.rows(), std::vector<double>(mat.cols()));
    for (std::size_t i = 0; i < mat.rows(); ++i) {
        for (std::size_t j = 0; j < mat.cols(); ++j) {
            vec[i][j] = mat(i, j);
        }
    }
    return vec;
}

class GenotypeGenerator {
public:
    virtual ~GenotypeGenerator() = default;
    virtual void generate(Eigen::MatrixXd& X, std::size_t n, std::size_t p, std::mt19937& gen) const = 0;
    virtual std::string name() const = 0;
};

class HWEGenotypeGenerator : public GenotypeGenerator {
private:
    double maf_;
public:
    explicit HWEGenotypeGenerator(double maf) : maf_(maf) {}
    void generate(Eigen::MatrixXd& X, std::size_t n, std::size_t p, std::mt19937& gen) const override {
        X.resize(n, p);
        double p0 = (1.0 - maf_) * (1.0 - maf_);
        double p1 = 2.0 * maf_ * (1.0 - maf_);
        std::uniform_real_distribution<double> unif_dist(0.0, 1.0);
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                double u = unif_dist(gen);
                if (u < p0) X(i, j) = 0.0;
                else if (u < p0 + p1) X(i, j) = 1.0;
                else X(i, j) = 2.0;
            }
        }
    }
    std::string name() const override { return "HWE (MAF=" + format_double(maf_) + ")"; }
};

class VariableMAFGenotypeGenerator : public GenotypeGenerator {
private:
    double maf_min_, maf_max_;
public:
    VariableMAFGenotypeGenerator(double min, double max) : maf_min_(min), maf_max_(max) {}
    void generate(Eigen::MatrixXd& X, std::size_t n, std::size_t p, std::mt19937& gen) const override {
        X.resize(n, p);
        std::uniform_real_distribution<double> maf_dist(maf_min_, maf_max_);
        std::uniform_real_distribution<double> unif_dist(0.0, 1.0);
        for (std::size_t j = 0; j < p; ++j) {
            double maf = maf_dist(gen);
            double p0 = (1.0 - maf) * (1.0 - maf);
            double p1 = 2.0 * maf * (1.0 - maf);
            for (std::size_t i = 0; i < n; ++i) {
                double u = unif_dist(gen);
                if (u < p0) X(i, j) = 0.0;
                else if (u < p0 + p1) X(i, j) = 1.0;
                else X(i, j) = 2.0;
            }
        }
    }
    std::string name() const override { return "Variable MAF"; }
};

class DesignMatrixFactory {
public:
    static std::unique_ptr<GenotypeGenerator> create_generator(const std::string& type) {
        if (type == "HWE_0.1") return std::make_unique<HWEGenotypeGenerator>(0.1);
        if (type == "HWE_0.3") return std::make_unique<HWEGenotypeGenerator>(0.3);
        return std::make_unique<VariableMAFGenotypeGenerator>(0.05, 0.45);
    }
};



// --- Main Demo ---
void demo_hwe_datagen() {
    const std::size_t n = 100;
    const std::size_t p = 150;

    Eigen::MatrixXd X;
    std::mt19937 gen(42);

    auto f = plt::figure();
    f->width(1200);
    f->height(400);

    std::vector<std::unique_ptr<GenotypeGenerator>> generators;
    generators.emplace_back(DesignMatrixFactory::create_generator("HWE_0.1"));
    generators.emplace_back(DesignMatrixFactory::create_generator("HWE_0.3"));
    generators.emplace_back(DesignMatrixFactory::create_generator("VariableMAF"));

    int plot_idx = 1;
    for (auto& generator : generators) {

        plt::subplot(1, 3, plot_idx++);

        generator->generate(X, n, p, gen);
        std::vector<std::vector<double>> X_mat = eigen_to_vector(X);

        plt::imagesc(X_mat);
        plt::colorbar();
        plt::title(generator->name());
        plt::xlabel("SNPs");

        if (plot_idx == 2) {
            plt::ylabel("Individuals");
        }
    }
    plt::show();
}

int main() {
    demo_hwe_datagen();
    return 0;
}
