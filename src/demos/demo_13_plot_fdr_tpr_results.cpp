// =============================================================================
/**
 * @file demo_13_plot_fdr_tpr_results.cpp
 *
 * @details Demonstration of plotting FDR, TPR, and Average L results
 */
// =============================================================================

// stdlib includes
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


// Matplot++ includes
#include <matplot/matplot.h>

// =============================================================================


/**
 * @brief Helper to format double values to string with one decimal place, exmple: 0.1, 1.0.
 *
 * @param value Double value to format.
 *
 * @return Formatted string representation of the double value.
 */
std::string format_double(double value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << value;
    return ss.str();
}

// ------------------------------------------------------------------------------------------

/**
 * @brief Major plot function to visualize FDR, TPR, Avg L, and ROC results in PDF.
 *
 * @param snr_values Vector of SNR values.
 * @param solver_names Vector of solver names.
 * @param fdr_results 2D vector of FDR results (rows: solvers)
 * @param tpr_results 2D vector of TPR results (rows: solvers)
 * @param avg_L_results 2D vector of Average L results (rows: solvers)
 * @param target_fdr Target FDR line to plot (default: 0.1)
 * @param window_size Window size for file naming
 */
void plot_results_pdf(
    const std::vector<double>& snr_values,
    const std::vector<std::string>& solver_names,
    const std::vector<std::vector<double>>& fdr_results,
    const std::vector<std::vector<double>>& tpr_results,
    const std::vector<std::vector<double>>& avg_L_results,
    const std::vector<std::vector<double>>& avg_T_results,
    double target_fdr,
    std::size_t window_size
) {
    using namespace matplot;

    std::string dir = "./simulations/";
    std::string win = "_w_" + std::to_string(window_size);

    std::vector<std::string> markers = {"o", "s", "^", "d", "v", "x", "*", "h"};

    // Custom Colors (R, G, B)
    std::vector<std::vector<double>> custom_colors = {
        {0.00, 0.45, 0.74}, // TLARS: Blue
        {0.85, 0.33, 0.10}, // TLASSO: Orange
        {0.47, 0.67, 0.19}, // TENET: Green
        {0.49, 0.18, 0.56}, // TSTEPWISE: Purple
        {0.64, 0.08, 0.18}, // TOMP: Red
        {0.05, 0.25, 0.30}, // TGP: Cyan
        {0.80, 0.00, 0.80}  // TACGP: Magenta
    };

    // Setup Ticks for SNR x-Axis (Grid every 0.1, Label every 0.5)
    std::vector<double> x_tick_pos;
    std::vector<std::string> x_tick_labels;
    double min_val = 0.0;
    double max_val = 5.0;

    for (double v = min_val; v <= max_val + 1e-5; v += 0.1) {
        x_tick_pos.push_back(v);
        // Label only integers and halves (0, 0.5, 1.0...)
        double remainder = std::abs(v * 2.0 - std::round(v * 2.0));
        if (remainder < 0.001) x_tick_labels.push_back(format_double(v));
        else x_tick_labels.push_back("");
    }

    // Setup Ticks for TPR y-Axis (Grid every 0.1, Label every 0.2)
    std::vector<double> y_tick_pos_TPR;
    std::vector<std::string> y_tick_labels_TPR;
    for (double v = 0.0; v <= 1.0 + 1e-5; v += 0.1) {
        y_tick_pos_TPR.push_back(v);
        // Label only integers and halves (0, 0.2, 0.4...)
        double remainder = std::abs(v * 5.0 - std::round(v * 5.0));
        if (remainder < 0.001) y_tick_labels_TPR.push_back(format_double(v));
        else y_tick_labels_TPR.push_back("");
    }

    // Calculate Sparse Marker Indices (Only at 0.5 steps)
    std::vector<size_t> sparse_indices;
    for (size_t i = 0; i < snr_values.size(); ++i) {
        double val = snr_values[i];
        double remainder = std::abs(val * 2.0 - std::round(val * 2.0));
        if (remainder < 0.001) sparse_indices.push_back(i);
    }
    // Safety: ensure endpoints are included
    if (sparse_indices.empty() || sparse_indices.front() != 0) {
        sparse_indices.insert(sparse_indices.begin(), 0);
    }
    if (sparse_indices.empty() || sparse_indices.back() != snr_values.size() - 1) {
        sparse_indices.push_back(snr_values.size() - 1);
    }

    // Find maximum in nested vectors
    auto find_max_in_nested = [](const std::vector<std::vector<double>>& data) {
        double max_val = 0.0;
        for (const auto& vec : data) {
            double local_max = *std::max_element(vec.begin(), vec.end());
            if (local_max > max_val) {
                max_val = local_max;
            }
        }
        return max_val;
    };


    // =========================================================
    // FIGURE 1: FDR vs SNR
    // =========================================================
    auto f1 = figure(true);
    f1->size(400, 300);
    f1->font_size(14);
    hold(on);
    grid(on);
    colororder(custom_colors);
    //gca()->position({0.10, 0.16, 0.88, 0.78});
    gca()->position({0.11, 0.13, 0.86, 0.84});

    for (size_t i = 0; i < solver_names.size(); ++i) {
        auto p = plot(snr_values, fdr_results[i]);
        p->line_width(2);
        p->line_style("-");
        p->marker(markers[i % markers.size()]);
        p->marker_size(5);
        p->marker_face(true);
        p->display_name(solver_names[i] + std::string(4, ' '));
        p->marker_indices(sparse_indices);
    }

    std::vector<double> target_line(snr_values.size(), target_fdr);
    auto p_t = plot(snr_values, target_line);
    p_t->line_style("--");
    p_t->color("red");
    p_t->line_width(2);
    p_t->display_name("tFDR = " + format_double(target_fdr) + std::string(4, ' '));

    xlabel("SNR");
    ylabel("FDR");
    xlim({min_val, max_val});
    ylim({0.0, 0.5});
    xticks(x_tick_pos);
    xticklabels(x_tick_labels);
    legend()->location(legend::general_alignment::topright);
    save(dir + "sim_01_snr_fdr_tfdr_0.1" + win + ".pdf");


    // =========================================================
    // FIGURE 2: TPR vs SNR
    // =========================================================
    auto f2 = figure(true);
    f2->size(400, 300);
    f2->font_size(14);
    hold(on);
    grid(on);
    colororder(custom_colors);
    //gca()->position({0.10, 0.16, 0.88, 0.78});
    gca()->position({0.11, 0.13, 0.86, 0.84});

    for (size_t i = 0; i < solver_names.size(); ++i) {
        auto p = plot(snr_values, tpr_results[i]);
        p->line_width(2);
        p->line_style("-");
        p->marker(markers[i % markers.size()]);
        p->marker_size(5);
        p->marker_face(true);
        p->display_name(solver_names[i] + std::string(4, ' '));
        p->marker_indices(sparse_indices);
    }

    xlabel("SNR");
    ylabel("TPR");
    xlim({min_val, max_val});
    yticks(y_tick_pos_TPR);
    yticklabels(y_tick_labels_TPR);
    ylim({0.0, 1.05});
    xticks(x_tick_pos);
    xticklabels(x_tick_labels);
    legend()->location(legend::general_alignment::bottomright);
    save(dir + "sim_01_snr_tpr_tfdr_0.1" + win + ".pdf");


    // =========================================================
    // FIGURE 3: Average L vs SNR
    // =========================================================
    auto f3 = figure(true);
    f3->size(400, 300);
    f3->font_size(14);
    hold(on);
    grid(on);
    colororder(custom_colors);
    //gca()->position({0.10, 0.16, 0.88, 0.78});
    gca()->position({0.11, 0.13, 0.86, 0.84});

    for (size_t i = 0; i < solver_names.size(); ++i) {

        // Round up Average L
        //std::vector<double> result;
        //std::transform(avg_L_results[i].begin(),
        //       avg_L_results[i].end(),
        //       std::back_inserter(result),
        //       [](double val) { return std::ceil(val); });
        //matplot::vector_1d dummy_val = result;
        //auto p = plot(snr_values, dummy_val);

        // Average L data plotting
        auto p = plot(snr_values, avg_L_results[i]);
        p->line_width(2);
        p->line_style("-");
        p->marker(markers[i % markers.size()]);
        p->marker_size(5);
        p->marker_face(true);
        p->display_name(solver_names[i] + std::string(4, ' '));
        p->marker_indices(sparse_indices);
    }

    xlabel("SNR");
    ylabel("Average L");
    xlim({min_val, max_val});
    double max_avg_L = find_max_in_nested(avg_L_results);
    max_avg_L = std::ceil(max_avg_L * 1.10);
    ylim({0.0, max_avg_L});
    xticks(x_tick_pos);
    xticklabels(x_tick_labels);
    legend()->location(legend::general_alignment::topright);
    save(dir + "sim_01_snr_avgL_tfdr_0.1" + win + ".pdf");


    // =========================================================
    // FIGURE 4: Average T vs SNR
    // =========================================================
    if (true) {
        auto f4 = figure(true);
        f4->size(400, 300);
        f4->font_size(14);
        hold(on);
        grid(on);
        colororder(custom_colors);
        //gca()->position({0.10, 0.16, 0.88, 0.78});
        gca()->position({0.11, 0.13, 0.86, 0.84});

        for (std::size_t i = 0; i < solver_names.size(); ++i) {
            auto p = plot(snr_values, avg_T_results[i]);
            p->line_width(2);
            p->line_style("-");
            p->marker(markers[i % markers.size()]);
            p->marker_size(5);
            p->marker_face(true);
            p->display_name(solver_names[i] + std::string(4, ' '));
            p->marker_indices(sparse_indices);
        }

        xlabel("SNR");
        ylabel("Average T");
        xlim({min_val, max_val});
        double max_avg_T = find_max_in_nested(avg_T_results);
        max_avg_T = std::ceil(max_avg_T * 1.10);
        ylim({0.0, max_avg_T});
        xticks(x_tick_pos);
        xticklabels(x_tick_labels);
        legend()->location(legend::general_alignment::bottomright);
        save(dir + "sim_01_snr_avgT_tfdr_0.1" + win + ".pdf");
    }

}



// ------------------------------------------------------------------------------------------



/**
 * @brief Main function to create TPR, FDR, and average L (avg L) plots from simulation data,
 *        and save them as PDF files.
 */
void main_plot_fdr_tpr_from_file() {

    std::cout << "Generating PDF plots from TRex_SNR_Sim data...\n";

    // SNR Values
    std::vector<double> snr_values = {
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        5.0
    };

    // Solver Names
    std::vector<std::string> solver_names = {
        "TLARS", "TLASSO", "TENET", "TSTEPWISE", "TOMP", "TGP", "TACGP"
    };

    // Data: from File
    // ----------------------------------------------------------
    // FDR Data -> insert your FDR data here
std::vector<std::vector<double>> fdr_results = {
    {0.0170, 0.0543, 0.1000, 0.1527, 0.1854, 0.1941, 0.2140, 0.1953, 0.2028, 0.2014, 0.2090, 0.1971, 0.1904, 0.1943, 0.1890, 0.1875, 0.1875, 0.1781, 0.1839, 0.1832, 0.1739}, // TLARS
    {0.0147, 0.0441, 0.1210, 0.1510, 0.1738, 0.1966, 0.2142, 0.1954, 0.2035, 0.2083, 0.2023, 0.1995, 0.1953, 0.1947, 0.1893, 0.1908, 0.1870, 0.1884, 0.1887, 0.1789, 0.1787}, // TLASSO
    {0.0230, 0.0621, 0.1059, 0.1894, 0.1969, 0.2163, 0.2286, 0.2272, 0.2252, 0.2221, 0.2231, 0.2289, 0.2309, 0.2326, 0.2310, 0.2405, 0.2441, 0.2436, 0.2390, 0.2413, 0.3001}, // TENET
    {0.0180, 0.0345, 0.0769, 0.0983, 0.1212, 0.1080, 0.1162, 0.1039, 0.1155, 0.0995, 0.0821, 0.0791, 0.0693, 0.0724, 0.0631, 0.0547, 0.0514, 0.0468, 0.0476, 0.0474, 0.0263}, // TSTEPWISE
    {0.0120, 0.0540, 0.0656, 0.1074, 0.1155, 0.1255, 0.1137, 0.1167, 0.0953, 0.0976, 0.0797, 0.0813, 0.0688, 0.0673, 0.0610, 0.0621, 0.0520, 0.0503, 0.0452, 0.0489, 0.0276}, // TOMP
    {0.0200, 0.0583, 0.0755, 0.1087, 0.0954, 0.1016, 0.1041, 0.1116, 0.1027, 0.0928, 0.0878, 0.0832, 0.0685, 0.0659, 0.0636, 0.0595, 0.0494, 0.0601, 0.0513, 0.0490, 0.0317}, // TGP
    {0.0147, 0.0267, 0.0648, 0.1070, 0.1175, 0.1143, 0.1173, 0.0990, 0.1142, 0.0956, 0.0900, 0.0793, 0.0695, 0.0731, 0.0637, 0.0593, 0.0488, 0.0532, 0.0492, 0.0445, 0.0303}  // TACGP
};


    // TPR Data -> insert your TPR data here
std::vector<std::vector<double>> tpr_results = {
    {0.0036, 0.0182, 0.0760, 0.1588, 0.2698, 0.3890, 0.5024, 0.6080, 0.6744, 0.7548, 0.8066, 0.8572, 0.8846, 0.9158, 0.9336, 0.9462, 0.9606, 0.9698, 0.9810, 0.9806, 1.0000}, // TLARS
    {0.0034, 0.0202, 0.0766, 0.1600, 0.2600, 0.3806, 0.5094, 0.6120, 0.6890, 0.7618, 0.8036, 0.8424, 0.8806, 0.9026, 0.9238, 0.9486, 0.9664, 0.9682, 0.9800, 0.9818, 1.0000}, // TLASSO
    {0.0036, 0.0232, 0.0734, 0.1710, 0.2818, 0.3992, 0.5030, 0.6094, 0.6932, 0.7500, 0.8132, 0.8426, 0.8778, 0.9138, 0.9354, 0.9452, 0.9626, 0.9614, 0.9738, 0.9754, 1.0000}, // TENET
    {0.0054, 0.0186, 0.0480, 0.1036, 0.1796, 0.3074, 0.4112, 0.5544, 0.6370, 0.7620, 0.8338, 0.8626, 0.9038, 0.9158, 0.9352, 0.9518, 0.9582, 0.9678, 0.9664, 0.9690, 0.9988}, // TSTEPWISE
    {0.0032, 0.0132, 0.0470, 0.1040, 0.1688, 0.2876, 0.4020, 0.5406, 0.6526, 0.7488, 0.8290, 0.8648, 0.9020, 0.9274, 0.9374, 0.9456, 0.9556, 0.9614, 0.9684, 0.9656, 0.9990}, // TOMP
    {0.0010, 0.0148, 0.0472, 0.1078, 0.1778, 0.2796, 0.4184, 0.5206, 0.6506, 0.7420, 0.8268, 0.8658, 0.9028, 0.9212, 0.9320, 0.9506, 0.9582, 0.9534, 0.9620, 0.9718, 0.9962}, // TGP
    {0.0032, 0.0164, 0.0488, 0.1036, 0.1792, 0.3024, 0.4096, 0.5292, 0.6244, 0.7576, 0.8092, 0.8732, 0.9018, 0.9178, 0.9342, 0.9464, 0.9606, 0.9594, 0.9646, 0.9670, 0.9982}  // TACGP
};


    // Avg L Data -> insert your Avg L data here
std::vector<std::vector<double>> avg_L_results = {
    {3.4720, 5.0600, 6.7100, 7.3580, 6.8340, 5.9260, 5.1440, 4.6040, 4.3020, 3.5820, 3.2660, 2.8600, 2.6980, 2.4200, 2.2560, 2.2880, 2.0020, 1.9060, 1.8340, 1.6980, 1.1560}, // TLARS
    {3.3460, 5.0760, 6.8800, 7.4140, 7.0180, 5.8980, 5.5840, 4.6940, 4.1280, 3.8620, 3.3160, 2.9240, 2.8100, 2.4820, 2.3820, 2.0940, 2.0480, 1.8960, 1.7460, 1.7260, 1.1580}, // TLASSO
    {3.4000, 5.4120, 6.6100, 7.2700, 6.9800, 6.2260, 5.1720, 4.3960, 4.0460, 3.6340, 3.4400, 2.9100, 2.7520, 2.5820, 2.4260, 2.2180, 1.9440, 1.8600, 1.8060, 1.7280, 1.1460}, // TENET
    {3.3180, 5.3860, 6.9080, 7.3740, 6.8720, 6.2200, 5.1520, 4.3040, 3.8640, 2.9180, 2.6560, 2.3080, 1.8880, 1.7020, 1.5980, 1.3720, 1.3460, 1.2620, 1.2660, 1.2200, 1.1100}, // TSTEPWISE
    {3.3120, 5.2620, 6.7720, 7.3600, 7.0700, 6.0160, 5.3500, 4.3500, 3.8660, 3.1840, 2.5620, 2.3280, 2.0040, 1.7880, 1.5060, 1.4120, 1.3400, 1.3140, 1.2840, 1.2380, 1.1040}, // TOMP
    {3.3920, 5.6220, 7.0380, 7.3220, 7.0360, 6.2240, 5.0560, 4.4880, 3.9600, 3.2040, 2.7100, 2.2440, 1.9360, 1.7120, 1.5280, 1.3820, 1.3360, 1.2920, 1.2800, 1.2520, 1.1500}, // TGP
    {3.3160, 5.4180, 7.0260, 7.0620, 6.8680, 5.9600, 5.2860, 4.3440, 3.7360, 3.1180, 2.6160, 2.3040, 1.9340, 1.7400, 1.5760, 1.3600, 1.3540, 1.3280, 1.2500, 1.2100, 1.1280}  // TACGP
};

    // Avg T Data -> insert your Avg T data here
std::vector<std::vector<double>> avg_T_results = {
    {3.6100, 3.5040, 3.8060, 4.4580, 5.0500, 5.6280, 5.9720, 6.1160, 6.2520, 6.0680, 5.8880, 5.7840, 5.5920, 5.3380, 5.2520, 5.2200, 4.9340, 4.8820, 4.8180, 4.7260, 4.1300}, // TLARS
    {3.6300, 3.4760, 3.8380, 4.4120, 5.0400, 5.5520, 6.3440, 6.1940, 6.2340, 6.3720, 5.8940, 5.6360, 5.7100, 5.4400, 5.2460, 5.0860, 5.0540, 4.9780, 4.7340, 4.7300, 4.1360}, // TLASSO
    {3.6720, 3.5200, 3.8460, 4.6400, 5.3080, 5.9020, 6.0820, 6.1560, 6.3180, 6.2120, 6.1460, 5.9480, 5.8360, 5.6940, 5.6080, 5.4740, 5.2800, 5.1300, 5.0500, 4.9760, 4.4920}, // TENET
    {3.8880, 3.8060, 3.8060, 3.8440, 4.0980, 4.6480, 4.6640, 4.8020, 4.7280, 4.5900, 4.5980, 4.3540, 4.2320, 4.1600, 4.1060, 4.0380, 4.0440, 4.0460, 4.0540, 4.0360, 4.0080}, // TSTEPWISE
    {3.9080, 3.7480, 3.7680, 3.9280, 4.1080, 4.4380, 4.7120, 4.8120, 4.8260, 4.7220, 4.4200, 4.3580, 4.2660, 4.1540, 4.0640, 4.0920, 4.0540, 4.0080, 4.0680, 4.0100, 3.9980}, // TOMP
    {3.9180, 3.7280, 3.6440, 3.9300, 4.1660, 4.4560, 4.6760, 4.7740, 4.8380, 4.6380, 4.5480, 4.4000, 4.2120, 4.1340, 4.0960, 4.0860, 4.0380, 4.0540, 4.0480, 4.0020, 4.0340}, // TGP
    {3.9180, 3.7840, 3.6760, 3.9120, 4.1040, 4.4820, 4.5800, 4.7880, 4.7040, 4.6200, 4.4260, 4.4040, 4.2320, 4.2020, 4.1160, 4.0980, 4.0640, 4.0380, 4.0140, 4.0000, 4.0020}  // TACGP
};


    // Set stagnation window size
    std::size_t window_size = 3;


    // Create figures
    // ----------------------------------------------------------
    plot_results_pdf(
        snr_values,
        solver_names,
        fdr_results,
        tpr_results,
        avg_L_results,
        avg_T_results,
        /*tFDR=*/0.1,
        /*stagnation_window_size=*/window_size
    );
}


// ------------------------------------------------------------------------------------------

int main() {

    main_plot_fdr_tpr_from_file();

    return 0;
}
