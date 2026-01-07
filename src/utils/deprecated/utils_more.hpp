/**
 * @brief Generate dummy variables and append them to X_aug in-place
 *
 * @param X_aug Augmented data matrix [X | D] where D are the dummy variables
 * @param n Number of rows
 * @param p Number of original predictors (columns in X)
 * @param num_dummies Number of dummy variables to generate
 * @param seed Random seed for reproducibility
 */
inline void generate_dummies_inplace(
    Eigen::MatrixXd& X_aug,
    std::size_t n,
    std::size_t p,
    std::size_t num_dummies,
    unsigned int seed = 42)
{
    #pragma omp parallel for schedule(static) if(num_dummies >= 1000)
    for (std::size_t j = 0; j < num_dummies; ++j) {
        std::mt19937 rng(seed + 1000003UL + p + j);
        std::normal_distribution<double> normal(0.0, 1.0);

        for (std::size_t i = 0; i < n; ++i) {
            X_aug(i, p + j) = normal(rng);
        }
    }
}


/**
 * @brief Fill matrix with normal(mu = 0, sigma = 1) random values in column-major order
 *
 * @param X_data Pointer to column-major matrix data (Eigen::MatrixXd layout)
 * @param n Number of rows
 * @param p Number of columns
 * @param base_seed Base random seed for reproducibility
 *
 * @note Eigen matrices are column-major by default.
 */
inline void fill_matrix_column_major_normal(
    double* X_data,
    std::size_t n,
    std::size_t p,
    unsigned int base_seed = 42)
{
    std::atomic<std::size_t> completed_columns{0};
    std::size_t last_reported = 0;

    #pragma omp parallel
    {

        #pragma omp for schedule(static, 1000)
        for (std::size_t j = 0; j < p; ++j) {
            std::mt19937 gen(base_seed + 1000003UL + j);
            std::normal_distribution<double> dist(0.0, 1.0);

            // Fill column j (Eigen column-major: elements are contiguous)
            for (std::size_t i = 0; i < n; ++i) {
                X_data[i + j * n] = dist(gen);
            }

            // Progress reporting
            std::size_t current = ++completed_columns;
            if (current % 1000 == 0 || current == p) {
                #pragma omp critical(progress_reporting)
                {
                    if (current > last_reported) {
                        std::cout << "  Progress: " << current << "/" << p
                                  << " columns (" << std::fixed
                                  << std::setprecision(1)
                                  << (100.0 * current / p) << "%)\r"
                                  << std::flush;
                        last_reported = current;
                    }
                }
            }
        }
    }
    std::cout << "\n";
}



/**
 * @brief Create memory-mapped X matrix and y response vector
 *
 * Creates X ~ N(0,1) and y = X * beta_true + noise on disk as memory-mapped files.
 *
 * @param X_filename Path to X matrix file (will be created)
 * @param y_filename Path to y vector file (will be created)
 * @param n Number of observations (rows)
 * @param p Number of predictors (columns)
 * @param true_support Indices of true non-zero coefficients
 * @param true_support_coefs Values of true non-zero coefficients
 * @param snr Signal-to-noise ratio
 * @param seed Random seed for reproducibility
 */
inline void create_mmapped_X_and_y(
    const char* X_filename,
    const char* y_filename,
    std::size_t n,
    std::size_t p,
    const std::vector<std::size_t>& true_support,
    const std::vector<double>& true_support_coefs,
    double snr = 1.0,
    unsigned int seed = 42,
    bool reproducible_noise = true
    )
{
    std::cout << "=== Creating memory-mapped X and y ===\n";
    std::cout << "  Dimensions: n = " << n << ", p = " << p << "\n";
    std::cout << "  True support size: " << true_support.size() << "\n";
    std::cout << "  SNR: " << snr << "\n\n";

    // Validate input
    if (true_support.size() != true_support_coefs.size()) {
        throw std::invalid_argument(
            "true_support and true_support_coefs must have same size"
        );
    }
    for (std::size_t idx : true_support) {
        if (idx >= p) {
            throw std::invalid_argument(
                "Support index exceeds number of predictors"
            );
        }
    }

    // Step 1: Check disk space
    std::size_t X_size = n * p * sizeof(double);
    std::size_t y_size = n * sizeof(double);
    std::cout << "  X size: " << (X_size / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  y size: " << (y_size / 1024.0) << " KB\n";

    if (!utils_perf::check_disk_space(X_size + y_size)) {
        throw std::runtime_error("Insufficient disk space for memory-mapped files");
    }

    // Step 2: Create memory-mapped files
    // Phase 1: Create and fill X with N(0,1) random values
    std::cout << "  Creating files...\n";
    {
        auto X_map = TRex::Utils::memmap::create_empty_map<double>(X_filename, n * p);
        fill_matrix_column_major_normal(X_map.data, n, p, seed);
        TRex::Utils::memmap::flush_mapping(X_map);
    }

    // Phase 2: Reopen X and generate y
    std::cout << "  Reopening X and creating y...\n";
    auto X_map_readonly = TRex::Utils::memmap::map_file_rw<double>(X_filename, n * p);
    auto y_map = TRex::Utils::memmap::create_empty_map<double>(y_filename, n);

    // Create Eigen::Map views
    Eigen::Map<Eigen::MatrixXd> X(X_map_readonly.data, n, p);
    Eigen::Map<Eigen::VectorXd> y(y_map.data, n);

    // Step 3: Generate response: y = X * beta_true + noise
    std::cout << "  Generating response y = X*beta_true + noise...\n";
    y.setZero();

    // Add signal from true support
    for (std::size_t i = 0; i < true_support.size(); ++i) {
        std::size_t idx = true_support[i];
        double coef = true_support_coefs[i];
        y.noalias() += coef * X.col(idx);
        std::cout << "    beta[" << idx << "] = " << coef << "\n";
    }

    // Calculate noise level from SNR
    double y_mean = y.mean();
    double var_sum = (y.array() - y_mean).square().sum();
    double signal_power = (n <= 100) ? var_sum / (n - 1) : var_sum / n;

    // Determine Noise level
    double noise_std = 0.0;
    if (snr > 0 && signal_power > 0) {
        noise_std = std::sqrt(signal_power / snr);
    }

    std::cout << "  Signal mean: " << y_mean << "\n";
    std::cout << "  Signal variance: " << signal_power << "\n";
    std::cout << "  Noise std: " << noise_std << "\n";

    std::mt19937 gen;
    if (reproducible_noise) {
         // Large separation, reproducible, avoid spurious correlation
        gen.seed(seed + 100 * p);
        std::cout << "  Using reproducible noise with seed " << (seed + 100 * p) << "\n";
    } else {
        std::random_device rd;
        gen.seed(rd());
        std::cout << "  Using random noise (non-reproducible)\n";
    }

    // Add noise
    std::normal_distribution<double> noise_dist(0.0, noise_std);
    for (std::size_t i = 0; i < n; ++i) {
        y(i) += noise_dist(gen);
    }

    // Flush to disk
    std::cout << "  Flushing to disk...\n";
    TRex::Utils::memmap::flush_mapping(X_map_readonly);
    TRex::Utils::memmap::flush_mapping(y_map);

    std::cout << "  ✓ Files created: " << X_filename << ", " << y_filename << "\n\n";
}



/**
 * @brief Create X_aug = [X | D] from existing memory-mapped X.
 *
 * @param X_filename Path to existing X matrix file (n x p).
 * @param X_aug_filename Path to X_aug file to create (n x (p + num_dummies)).
 * @param n Number of observations.
 * @param p Number of original predictors.
 * @param num_dummies Number of dummy variables to append.
 * @param seed Random seed for dummy generation (reproducible).
 * @return MappedFile<double> handle to X_aug.
 */
inline TRex::Utils::memmap::MappedFile<double> augment_with_dummies(
    const char* X_filename,
    const char* X_aug_filename,
    std::size_t n,
    std::size_t p,
    std::size_t num_dummies,
    unsigned int seed = 42
) {
    std::cout << "=== Augmenting X with " << num_dummies << " dummies ===\n";

    // Step 1: Load X
    auto X_map = TRex::Utils::memmap::map_file_rw<double>(X_filename, n * p);
    const double* X_data = X_map.data;

    // Step 2: Create X_aug file
    auto X_aug_map = TRex::Utils::memmap::create_empty_map<double>(
        X_aug_filename, n * (p + num_dummies)
    );
    double* X_aug_data = X_aug_map.data;

    // Step 3: Copy X to first p columns (parallel)
    std::cout << "  Copying X..." << std::flush;
    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < p; ++j) {
        const double* src = X_data + j * n;
        double* dst = X_aug_data + j * n;
        std::memcpy(dst, src, n * sizeof(double));
    }
    std::cout << " done\n";

    // Step 4: Generate dummies (parallel, deterministic)
    std::cout << "  Generating " << num_dummies << " dummies..." << std::flush;
    #pragma omp parallel for schedule(static)
    for (std::size_t j = 0; j < num_dummies; ++j) {
        std::mt19937 gen(seed + 1000003UL + p + j);  // 1000003 is prime
        std::normal_distribution<double> dist(0.0, 1.0);

        double* col_start = X_aug_data + (p + j) * n;
        for (std::size_t i = 0; i < n; ++i) {
            col_start[i] = dist(gen);
        }
    }
    std::cout << " done\n";

    // Step 5: Flush
    TRex::Utils::memmap::flush_mapping(X_aug_map);
    std::cout << "  ✓ X_aug created (" << n << " x " << (p + num_dummies) << ")\n";

    return X_aug_map;
}
