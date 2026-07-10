// ===================================================================================
// trex_dummy_generator.hpp
// ===================================================================================
#ifndef TREX_SELECTOR_METHODS_UTILS_TREX_DUMMY_GENERATOR_HPP
#define TREX_SELECTOR_METHODS_UTILS_TREX_DUMMY_GENERATOR_HPP
// ===================================================================================
/**
 * @file trex_dummy_generator.hpp
 *
 * @brief Encapsulates all dummy matrix generation for the T-Rex Selector.
 *
 * @details
 *  Responsibilities:
 *    - Generate fresh dummy matrices D_k from configured distributions.
 *    - Generate permuted variants from a stored base dummy matrix.
 *    - Generate experiment-specific dummies from deterministic seeds (DIRECT).
 *    - Center + L2-normalize all generated dummies.
 *    - Store / manage the base dummy matrix for PERMUTATION strategies.
 *    - Store / manage K pre-generated dummy matrices for STANDARD/HCONCAT/SKIP.
 *
 *  Ownership model:
 *  TRexSelector owns a DummyGenerator member. Derived TRex classes that
 *  need different dummy generation can either:
 *    (a) override a virtual getDummyGenerator() accessor, or
 *    (b) call a custom DummyGenerator via the constructor.
 *
 *  Thread safety:
 *  generate() and generateForExperiment() are const and thread-safe
 *  (each call uses its own RNG state derived from the seed).
 *  Mutating methods (storeBaseDummies, storeGeneratedDummies, reset)
 *  are not thread-safe. They must be called outside parallel regions.
 */
// ===================================================================================

// std includes
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Eigen includes
#include <Eigen/Dense>

// Dummy generation utility
#include <utils/datageneration/utils_dummygen.hpp>

// Data normalizer (for center + L2 normalize)
#include <trex_selector_methods/trex_utils/trex_data_normalizer.hpp>

// ===================================================================================

// Embedded into namespace trex::trex_selector_methods::utils::dummy_generator
namespace trex::trex_selector_methods::utils::dummy_generator {

// Namespace alias
namespace dummygen = trex::utils::datageneration::dummygen;

// ===================================================================================

class DummyGenerator {
public:

    // ==========================================================================
    // Construction
    // ==========================================================================

    /**
     * @brief Construct a DummyGenerator.
     *
     * @param n             Number of observations (rows).
     * @param distribution  Distribution for dummy generation (default: Normal).
     * @param seed          Base random seed (< 0 for non-deterministic).
     * @param verbose       Enable warnings for near-zero norm columns.
     * @param scaling_mode  Column scaling convention applied to generated
     *                      dummies (default: L2). Must match the scaling used
     *                      for X so dummy and predictor columns share a scale.
     */
    explicit DummyGenerator(
        std::size_t n,
        dummygen::Distribution distribution = dummygen::Distribution::Normal(),
        int seed = -1,
        bool verbose = false,
        data_normalizer::ScalingMode scaling_mode = data_normalizer::ScalingMode::L2)
        : n_(n),
        distribution_(distribution),
        verbose_(verbose),
        scaling_mode_(scaling_mode),
        resolved_base_seed_((seed >= 0)
            ? static_cast<std::uint64_t>(seed)
            : ((static_cast<std::uint64_t>(std::random_device{}()) << 32)
               | static_cast<std::uint64_t>(std::random_device{}())))
    {}

    /** @brief Destructor of DummyGenerator */
    ~DummyGenerator() = default;

    /** @brief Deleted copy constructor */
    DummyGenerator(const DummyGenerator&) = delete;

    /** @brief Deleted copy assignment operator */
    DummyGenerator& operator=(const DummyGenerator&) = delete;

    /** @brief Defaulted move constructor */
    DummyGenerator(DummyGenerator&&) = default;

    /** @brief Defaulted move assignment operator */
    DummyGenerator& operator=(DummyGenerator&&) = default;


    // ==========================================================================
    // Core dummy generation — fresh dummies
    // ==========================================================================

    /**
     * @brief Generate a dummy matrix D of size n × num_dummies.
     *
     * @details Generates from the configured distribution, then centers and
     *          L2-normalizes each column.  Thread-safe (uses experiment_id
     *          to derive a unique seed — no shared mutable state).
     *
     * @param num_dummies    Number of dummy columns.
     * @param experiment_id  Experiment index (for seed derivation).
     *
     * @return Normalized dummy matrix (n × num_dummies).
     */
    Eigen::MatrixXd generate(std::size_t num_dummies,
                             std::size_t experiment_id) const
    {

        Eigen::MatrixXd D(n_, num_dummies);
        generateInto(D, experiment_id);
        return D;
    }


    /**
     * @brief Generate a dummy matrix with a seed derived via mix_seed.
     *
     * @details Used by the DIRECT strategy where each experiment k needs
     *          a deterministic seed that is uncorrelated with adjacent k's.
     *
     * @param num_dummies    Number of dummy columns.
     * @param experiment_id  Experiment index k.
     *
     * @return Normalized dummy matrix (n × num_dummies).
     */
    Eigen::MatrixXd generateDirect(std::size_t num_dummies,
                                    std::size_t experiment_id) const {

        Eigen::MatrixXd D(n_, num_dummies);
        generateDirectInto(D, experiment_id);
        return D;
    }

    // =========================================================================
    // In-place dummy generation (directly writes into target matrix)
    // =========================================================================

    /**
     * @brief Generate dummies directly into an existing matrix (zero-copy).
     *
     * @details Writes random dummies into `target`, then centers and
     *          L2-normalizes each column in-place.  The target can be
     *          an Eigen::Map over a memory-mapped region.
     *
     * @param target         Matrix to write into (n × num_dummies, pre-sized).
     * @param experiment_id  Experiment index (for seed derivation).
     */
    void generateInto(Eigen::Ref<Eigen::MatrixXd> target,
                      std::size_t experiment_id) const {

        const std::uint64_t base_seed = deriveBlockSeed64(experiment_id, 0);

        dummygen::generate_dummies(
            target, n_,
            static_cast<std::size_t>(target.cols()),
            base_seed, distribution_
        );

        utils::data_normalizer::centerAndL2NormalizeMatrix(
            target,
            std::numeric_limits<double>::epsilon(),
            verbose_,
            scaling_mode_
        );
    }


    /**
     * @brief Generate dummies into target with explicit L-loop tag.
     *
     * @details Used by memory-mapped Standard/SKIPL/HCONCAT strategies to
     *          mirror the seed derivation of generateAndStore(K, num_dummies,
     *          l_tag). With l_tag = 0 (HCONCAT convention) a full generation
     *          reproduces bit-exactly the matrix an incremental expandInto()
     *          build would produce — in-memory and memory-mapped paths agree.
     *
     * @param target       Matrix to write into (n × num_dummies, pre-sized).
     * @param experiment_k Experiment index k.
     * @param l_tag        L-loop tag (0 = prefix-stable; L_iter for STANDARD).
     */
    void generateInto(Eigen::Ref<Eigen::MatrixXd> target,
                      std::size_t experiment_k,
                      std::size_t l_tag) const {

        const std::uint64_t base_seed = deriveBlockSeed64(experiment_k, l_tag);

        dummygen::generate_dummies(
            target, n_,
            static_cast<std::size_t>(target.cols()),
            base_seed, distribution_
        );

        utils::data_normalizer::centerAndL2NormalizeMatrix(
            target,
            std::numeric_limits<double>::epsilon(),
            verbose_,
            scaling_mode_
        );
    }


    /**
     * @brief Expand a memory-mapped D region by generating new columns in-place.
     *
     * @details Used by the HCONCAT memory-mapped path.  The first
     *          `existing_cols` columns are already on disk from the prior
     *          L-loop iteration and are left untouched.  Only the rightmost
     *          `target.cols() - existing_cols` columns are generated — under
     *          the SAME per-experiment base (l_tag = 0) with the global column
     *          offset, so the grown design lives in one injective seed domain:
     *          duplicate columns across blocks are impossible by construction,
     *          and a later full generateInto(k, l_tag = 0) reproduces the
     *          identical matrix.
     *
     * @param target        Full D map (n × total_cols, pre-sized to new total).
     * @param existing_cols Number of columns already written from prior L.
     * @param experiment_k  Experiment index k.
     */
    void expandInto(Eigen::Ref<Eigen::MatrixXd> target,
                    std::size_t existing_cols,
                    std::size_t experiment_k) const {

        const std::size_t total_cols = static_cast<std::size_t>(target.cols());
        if (existing_cols >= total_cols) return;  // nothing to expand

        const std::size_t new_cols = total_cols - existing_cols;

        // Same base as the initial block; new columns are distinguished by
        // their global column offset (same derivation as expandStored)
        const std::uint64_t base_seed = deriveBlockSeed64(experiment_k, 0);

        // Generate only the new rightmost columns
        auto new_block = target.rightCols(static_cast<Eigen::Index>(new_cols));

        dummygen::generate_dummies(
            new_block, n_, new_cols, base_seed, distribution_,
            /*col_offset=*/existing_cols
        );

        utils::data_normalizer::centerAndL2NormalizeMatrix(
            new_block,
            std::numeric_limits<double>::epsilon(),
            verbose_,
            scaling_mode_
        );
    }


    /**
     * @brief Generate DIRECT-strategy dummies directly into an existing matrix.
     *
     * @details Seeds via `deriveBlockSeed64(experiment_id, 0)`, i.e. from the
     *          resolved base seed with the prefix-stable tag. This honours the
     *          user-supplied seed and is stable across repeated calls within
     *          one run, so every T-loop step re-derives the identical D_k even
     *          when `seed < 0` (previously each call drew fresh `random_device`
     *          entropy, silently changing the dummies mid-calibration).
     *
     * @param target         Matrix to write into (n × num_dummies, pre-sized).
     * @param experiment_id  Experiment index k.
     */
    void generateDirectInto(Eigen::Ref<Eigen::MatrixXd> target,
                            std::size_t experiment_id) const {

        const std::uint64_t seed_k = deriveBlockSeed64(experiment_id, 0);

        dummygen::generate_dummies(
            target, n_,
            static_cast<std::size_t>(target.cols()),
            seed_k, distribution_
        );

        utils::data_normalizer::centerAndL2NormalizeMatrix(
            target,
            std::numeric_limits<double>::epsilon(),
            verbose_,
            scaling_mode_
        );
    }


    /**
     * @brief Write a row-permutation of the base dummies into an existing matrix.
     *
     * @details For k == 0, copies the base dummies directly into target.
     *          For k > 0, applies the deterministic row permutation in-place.
     *          No temporary allocation.
     *
     * @param target       Matrix to write into (n × num_dummies, pre-sized).
     * @param k            Experiment index [0, K).
     * @param num_dummies  Expected number of dummy columns (validated).
     */
    void permuteInto(Eigen::Ref<Eigen::MatrixXd> target,
                     std::size_t k,
                     std::size_t num_dummies) const {

        if (!base_initialized_) {
            throw std::logic_error(
                "DummyGenerator::permuteInto: base dummies not initialized");
        }
        if (static_cast<std::size_t>(base_dummies_perm_.cols()) != num_dummies) {
            throw std::invalid_argument(
                "DummyGenerator::permuteInto: size mismatch, expected "
                + std::to_string(num_dummies) + " cols, got "
                + std::to_string(base_dummies_perm_.cols()));
        }

        if (k == 0) {
            // Direct copy of base
            target = base_dummies_perm_;
        } else {
            // Deterministic row permutation directly into target
            // (full 64-bit seed path; see dummygen::make_column_engine)
            std::mt19937 rng = dummygen::make_column_engine(base_seed_perm_, k);
            applyRowPermutationInto(base_dummies_perm_, target, rng);
        }
    }


    // ==========================================================================
    // Batch generation — K dummy matrices (STANDARD/HCONCAT/SKIP)
    // ==========================================================================

    /**
     * @brief Generate and store K dummy matrices.
     *
     * @details Fills stored_dummies_ with K freshly generated dummy matrices.
     *          Used by STANDARD/HCONCAT/SKIP strategies that need all K
     *          dummies available simultaneously (for parallel execution).
     *
     * @warning Destroys the previously stored matrices. Any retained
     *          warm-start solver views them, so invalidate the
     *          WarmStartManager BEFORE calling this.
     *
     * @param K             Number of experiments.
     * @param num_dummies   Number of dummy columns per matrix.
     * @param seed_factor   L-loop iteration factor for seed mixing (0 = none).
     */
    void generateAndStore(std::size_t K,
                          std::size_t num_dummies,
                          std::size_t l_tag = 0) {

        stored_dummies_.clear();
        stored_dummies_.reserve(K);

        for (std::size_t k = 0; k < K; ++k) {
            Eigen::MatrixXd D(n_, num_dummies);
            generateInto(D, k, l_tag);
            stored_dummies_.emplace_back(std::move(D));
        }
    }


    /**
     * @brief Expand each of the K stored dummy matrices by p_new columns.
     *
     * @details Used by the HCONCAT strategy.  Appends p_new freshly generated
     *          columns to the right of each existing D_k, preserving all
     *          previously generated columns exactly.  New columns are drawn
     *          under the SAME per-experiment base (l_tag = 0) with the global
     *          column offset, so the whole grown D_k lives in one injective
     *          seed domain: bit-identical columns across expansion blocks are
     *          impossible by construction (the former per-block 32-bit reseed
     *          could collide), and a full generateInto(k, l_tag = 0) of the
     *          final size reproduces the identical matrix (memory-mapped path
     *          equivalence).
     *
     *          After this call, stored_dummies_[k] has size n × (old_cols + p_new).
     *
     * @warning The conservativeResize REALLOCATES each stored matrix's heap
     *          buffer. Any retained warm-start solver holds a non-owning view
     *          into the old buffer, so the caller MUST invalidate the
     *          WarmStartManager BEFORE calling this (see
     *          TRexSelector::prepareDummiesForLStep, HCONCAT case).
     *
     * @param p_new  Number of new dummy columns to append per matrix.
     *
     * @throws std::runtime_error if stored_dummies_ is empty.
     */
    void expandStored(std::size_t p_new) {

        if (stored_dummies_.empty()) {
            throw std::runtime_error(
                "DummyGenerator::expandStored: no stored dummies to expand."
                " Call generateAndStore() first.");
        }

        const std::size_t K = stored_dummies_.size();

        for (std::size_t k = 0; k < K; ++k) {
            const std::size_t old_cols = stored_dummies_[k].cols();

            // Append: [D_k_old | D_k_new], then generate the new rightmost
            // columns in-place under the same base with global column offsets
            stored_dummies_[k].conservativeResize(Eigen::NoChange,
                static_cast<Eigen::Index>(old_cols + p_new));

            expandInto(stored_dummies_[k], old_cols, k);
        }
    }


    /**
     *  @brief Get stored dummy matrix for experiment k (const ref).
     *
     *  @param k Experiment index [0, K).
     *
     *  @return Const reference to the stored dummy matrix.
     *
     *  @throws std::out_of_range if k is out of bounds.
     */
    const Eigen::MatrixXd& getStored(std::size_t k) const {
        if (k >= stored_dummies_.size()) {
            throw std::out_of_range(
                "DummyGenerator::getStored: k=" + std::to_string(k)
                + " >= stored size " + std::to_string(stored_dummies_.size()));
        }
        return stored_dummies_[k];
    }


    /** @brief Number of currently stored dummy matrices. */
    std::size_t numStored() const noexcept { return stored_dummies_.size(); }


    /**
     *  @brief Whether K dummies are currently stored.
     *
     *  @param K Number of dummies to check.
     *
     *  @return True if K dummies are stored, false otherwise.
     */
    bool hasStored(std::size_t K) const noexcept {
        return stored_dummies_.size() == K;
    }


    // ==========================================================================
    // Permutation strategy — base dummy management
    // ==========================================================================

    /**
     * @brief Store a base dummy matrix for the PERMUTATION strategy.
     *
     * @param base_dummies  The base dummy matrix (n × num_dummies).
     * @param base_seed     Seed used to generate it (for reproducibility;
     *                      also the id under which expandBaseDummies() grows
     *                      the base prefix-stably).
     */
    void storeBaseDummies(Eigen::MatrixXd base_dummies,
                          std::uint64_t base_seed)
    {
        base_dummies_perm_ = std::move(base_dummies);
        base_seed_perm_ = base_seed;
        base_initialized_ = true;
    }


    /**
     * @brief Expand the stored PERMUTATION base dummies by p_new columns.
     *
     * @details Grows base_dummies_perm_ to [base_old | new_cols].  The new
     *          columns are generated under the SAME base id that produced the
     *          initial base (deriveBlockSeed64(base_seed_perm_, 0)) with the
     *          global column offset, so the grown base lives in one injective
     *          seed domain — duplicate columns between the original base and
     *          any expansion are impossible by construction (the former
     *          mix_seed(base, L) per-expansion reseed could collide), and the
     *          expansion is prefix-stable across L iterations.
     *
     *          Row permutations per experiment are applied downstream by
     *          getPermuted()/permuteInto() and are unaffected.
     *
     * @param p_new  Number of new dummy columns to append.
     *
     * @throws std::logic_error if no base dummies are stored.
     */
    void expandBaseDummies(std::size_t p_new) {

        if (!base_initialized_) {
            throw std::logic_error(
                "DummyGenerator::expandBaseDummies: base dummies not initialized");
        }

        const std::size_t old_cols =
            static_cast<std::size_t>(base_dummies_perm_.cols());

        base_dummies_perm_.conservativeResize(Eigen::NoChange,
            static_cast<Eigen::Index>(old_cols + p_new));

        expandInto(base_dummies_perm_, old_cols,
                   static_cast<std::size_t>(base_seed_perm_));
    }


    /**
     * @brief Get dummy matrix for experiment k using permutation strategy.
     *
     * @details
     *  - k == 0: returns base dummies as-is.
     *  - k > 0:  returns a deterministic row-permutation of the base dummies.
     *            Seed = mix_seed(base_seed_perm_, k) (stable across T-loop iterations).
     *
     * @param k             Experiment index [0, K).
     * @param num_dummies   Expected number of dummy columns (validated).
     *
     * @return Permuted dummy matrix (n × num_dummies).
     */
    Eigen::MatrixXd getPermuted(std::size_t k,
                                std::size_t num_dummies) const
    {

        if (!base_initialized_) {
            throw std::logic_error(
                "DummyGenerator::getPermuted: base dummies not initialized");
        }
        if (static_cast<std::size_t>(base_dummies_perm_.cols()) != num_dummies) {
            throw std::invalid_argument(
                "DummyGenerator::getPermuted: size mismatch, expected "
                + std::to_string(num_dummies) + " cols, got "
                + std::to_string(base_dummies_perm_.cols()));
        }

        // Experiment 0: use base directly
        if (k == 0) {
            return base_dummies_perm_;
        }

        // Experiments k > 0: deterministic row permutation
        // (full 64-bit seed path; see dummygen::make_column_engine)
        std::mt19937 rng = dummygen::make_column_engine(base_seed_perm_, k);
        return applyRowPermutation(base_dummies_perm_, rng);
    }


    /** @brief Whether base dummies are initialized for permutation. */
    bool hasBaseDummies() const noexcept { return base_initialized_; }


    /** @brief Get the base permutation seed. */
    unsigned int baseSeedPerm() const noexcept { return base_seed_perm_; }


    // ==========================================================================
    // Reset
    // ==========================================================================

    /**
     * @brief Clear all stored state (base dummies + stored K dummies).
     *
     * @details Called when L-loop changes the dummy configuration.
     */
    void reset() {
        stored_dummies_.clear();
        base_dummies_perm_.resize(0, 0);
        base_initialized_ = false;
    }


private:

    // ==========================================================================
    // Configuration
    // ==========================================================================

    /** @brief Number of rows in the dummy matrix. */
    std::size_t n_;

    /** @brief Distribution type for dummy generation. */
    dummygen::Distribution distribution_;

    /** @brief Flag to print progress. */
    bool verbose_;

    /** @brief Column scaling convention applied to generated dummies. */
    data_normalizer::ScalingMode scaling_mode_;

    /** @brief 64-bit base seed resolved once at construction.
     *
     *  Equals `seed_` when a deterministic seed (>= 0) was requested, and two
     *  `std::random_device` draws packed into 64 bits otherwise. Resolving the
     *  base once (instead of per call) guarantees that repeated generation
     *  requests for the same (experiment, l_tag) pair — e.g. the DIRECT
     *  strategy re-deriving D_k at every T-loop step — reproduce the identical
     *  dummy matrix within one selector run, while separate runs still obtain
     *  fresh entropy. 64-bit width keeps the whole seeding pipeline out of the
     *  collision-prone 32-bit space (see dummygen::mix_seed64). */
    std::uint64_t resolved_base_seed_;

    // ==========================================================================
    // State — STANDARD/HCONCAT/SKIP
    // ==========================================================================

    /** @brief K pre-generated dummy matrices. */
    std::vector<Eigen::MatrixXd> stored_dummies_;

    // ==========================================================================
    // State — PERMUTATION
    // ==========================================================================

    /** @brief Base dummy matrix (n × num_dummies). */
    Eigen::MatrixXd base_dummies_perm_;

    /** @brief Seed used to generate base dummies (id for prefix-stable expansion). */
    std::uint64_t base_seed_perm_{0};

    /** @brief Whether base dummies are initialized. */
    bool base_initialized_{false};


    // ==========================================================================
    // Internal helpers
    // ==========================================================================

    /**
     * @brief Derive the 64-bit block base seed for (experiment k, L-tag).
     *
     * @details Packs the experiment id and the L-loop tag into DISJOINT bit
     *          fields of a single 64-bit index (`l_tag << 32 | k`) and mixes it
     *          with `mix_seed64`.  Because mix_seed64 is injective in the index
     *          for a fixed base, two distinct (k, l_tag) pairs can NEVER yield
     *          the same block base — whole-block duplication between experiments
     *          or L-iterations is impossible by construction (the former 32-bit
     *          `mix_seed(k, L)` scheme could collide).
     *
     *          Per-column seeds are then `mix_seed64(block_base, col)` inside
     *          generate_dummies(): injective in the column index for a fixed
     *          block base, so bit-identical columns within one block (or one
     *          incrementally grown HCONCAT design, which keeps l_tag = 0 and
     *          advances the column offset) are likewise impossible.
     *
     *          Tag conventions:
     *            - l_tag = 0:      prefix-stable strategies (HCONCAT, DIRECT,
     *                              PERMUTATION base). The same (k, col) always
     *                              reproduces the same column, so re-derivation
     *                              at any T-/L-step and incremental expansion
     *                              agree bit-exactly.
     *            - l_tag = L_iter: fresh-per-iteration strategies (STANDARD;
     *                              SKIPL uses L_max for equivalence with
     *                              STANDARD's final iteration).
     *
     *          The base is `resolved_base_seed_`, fixed at construction: the
     *          same request always reproduces the same dummies within one
     *          selector run. For Monte Carlo FDR sweeps the caller should pass
     *          `seed = -1` so that every selector run draws a fresh 64-bit
     *          hardware-entropy base; a fixed integer seed is suitable for
     *          exact reproducibility of a single run.
     *
     * @param experiment_id  Experiment index k (or other unique identifier).
     * @param l_tag          L-loop tag (see conventions above).
     *
     * @return 64-bit block base seed.
     */
    std::uint64_t deriveBlockSeed64(std::size_t experiment_id,
                                    std::size_t l_tag = 0) const {
        const std::uint64_t packed =
            (static_cast<std::uint64_t>(l_tag) << 32)
            | (static_cast<std::uint64_t>(experiment_id) & 0xFFFFFFFFULL);
        return dummygen::mix_seed64(resolved_base_seed_, packed);
    }


    /**
     * @brief Apply a deterministic row permutation to a dummy matrix.
     *
     * @details Preserves L2 norm of each column (sum of squared elements is
     *          invariant under row reordering).
     *          Uses transpose → permute columns → transpose back for
     *          cache-friendly access.
     *
     * @param dummies  Input matrix (n × L).
     * @param rng      Random engine (seeded deterministically).
     *
     * @return Row-permuted matrix (n × L).
     */
    static Eigen::MatrixXd applyRowPermutation(
        const Eigen::MatrixXd& dummies,
        std::mt19937& rng) {

        Eigen::MatrixXd result(dummies.rows(), dummies.cols());
        applyRowPermutationInto(dummies, result, rng);
        return result;
    }


    /**
     * @brief Apply a deterministic row permutation, writing directly into target.
     *
     * @details Zero-copy variant for the memory-mapped path.
     *          target must be pre-sized to (n × L).
     *          Row-permutes `source` into `target` without intermediate
     *          allocation.  Operates row-by-row (source[perm[i]] → target[i]),
     *          which is efficient for column-major storage when L is large.
     *
     * @param source  Input matrix (n × L).
     * @param target  Output matrix (n × L, pre-sized).
     * @param rng     Random engine (seeded deterministically).
     */
    static void applyRowPermutationInto(
        const Eigen::MatrixXd& source,
        Eigen::Ref<Eigen::MatrixXd> target,
        std::mt19937& rng)
    {

        const auto n = source.rows();

        // Generate row permutation indices
        std::vector<Eigen::Index> row_indices(n);
        std::iota(row_indices.begin(), row_indices.end(), 0);
        std::shuffle(row_indices.begin(), row_indices.end(), rng);

        // Permute: target[i, :] = source[perm[i], :]
        #pragma omp parallel for schedule(static) if(n > 100)
        for (Eigen::Index i = 0; i < n; ++i) {
            target.row(i) = source.row(row_indices[i]);
        }
    }

// ---------------------------------------------------------------------
}; /* End of class DummyGenerator */

// ===================================================================================
} /* End of namespace trex::trex_selector_methods::utils::dummy_generator */
// ===================================================================================
#endif /* End of TREX_SELECTOR_METHODS_UTILS_TREX_DUMMY_GENERATOR_HPP */
