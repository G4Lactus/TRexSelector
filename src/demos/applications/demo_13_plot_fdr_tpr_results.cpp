// =============================================================================
/**
 * @file demo_13_plot_fdr_tpr_results.cpp
 *
 * @details Demonstration of plotting FDR, TPR, and Average L results
 */
// =============================================================================

// stdlib includes
#include <array>
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
 * @brief Helper to format double values to string with one decimal place, example: 0.1, 1.0.
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

    std::vector<std::string> markers = {
        "o", "s", "^", "d", "v", "x", "*", "h"};

    // Custom Colors (R, G, B)
    std::vector<std::vector<double>> custom_colors = {
        {0.00, 0.45, 0.74}, // TLARS: Blue
        {0.85, 0.33, 0.10}, // TLASSO: Orange
        {0.25, 0.40, 0.08}, // TENET: Green ; old green: {0.47, 0.67, 0.19}
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

    std::pair<int, int> fig_size(800, 600);
    constexpr std::array<float, 4>  default_pos = {0.11f, 0.13f, 0.86f, 0.84f};
    int font_size = 56;
    double line_width = 2;
    double marker_size = 5;
    std::string font_name = "Times New Roman";

    // =========================================================
    // FIGURE 1: FDR vs SNR
    // =========================================================
    /*
    auto f1 = figure(true);
    f1->size(fig_size.first, fig_size.second);
    f1->font_size(font_size);
    f1->font(font_name);

    hold(on);
    grid(on);
    colororder(custom_colors);
    gca()->position(default_pos);

    for (size_t i = 0; i < solver_names.size(); ++i) {
        auto p = plot(snr_values, fdr_results[i]);
        p->line_width(line_width);
        p->line_style("-");
        p->marker(markers[i % markers.size()]);
        p->marker_size(marker_size);
        p->marker_face(true);
        p->display_name(solver_names[i] + std::string(4, ' '));
        p->marker_indices(sparse_indices);
    }

    std::vector<double> target_line(snr_values.size(), target_fdr);
    auto p_t = plot(snr_values, target_line);
    p_t->line_style("--");
    p_t->color("red");
    p_t->line_width(line_width);
    p_t->display_name("tFDR = " + format_double(target_fdr) + std::string(6, ' '));

    xlabel("SNR");
    ylabel("FDR");
    xlim({min_val, max_val});
    ylim({0.0, 0.25});
    xticks(x_tick_pos);
    xticklabels(x_tick_labels);
    legend()->location(legend::general_alignment::topright);
    save(dir + "sim_snr_fdr_tfdr_01" + win + ".pdf");
    */


// ========================================
// CONFIGURE PDF TERMINAL FIRST - This gives you full control!
// ========================================
auto f1 = figure(true);
// Use pdfcairo for better TrueType font embedding (not Type 3)
f1->backend()->run_command("set terminal pdfcairo enhanced color font 'Times-Roman,15' size 5.5in,3.5in");

// Set individual font sizes for each element
f1->backend()->run_command("set xlabel font 'Times-Roman,17'");    // X-axis label
f1->backend()->run_command("set ylabel font 'Times-Roman,17'");    // Y-axis label
f1->backend()->run_command("set xtics font 'Times-Roman,15'");     // X-axis tick labels
f1->backend()->run_command("set ytics font 'Times-Roman,15'");     // Y-axis tick labels
f1->backend()->run_command("set key font 'Times-Roman,17'");       // Legend font

// Optional: These matplot++ settings may or may not be needed now
// f1->size(fig_size.first, fig_size.second);  // Size set in terminal
// f1->font_size(font_size);                   // Font set in terminal
// f1->font(font_name);                        // Font set in terminal

hold(on);
grid(on);
colororder(custom_colors);
gca()->position(default_pos);

for (size_t i = 0; i < solver_names.size(); ++i) {
    auto p = plot(snr_values, fdr_results[i]);
    p->line_width(line_width);
    p->line_style("-");
    p->marker(markers[i % markers.size()]);
    p->marker_size(marker_size);
    p->marker_face(true);
    p->display_name(solver_names[i] + std::string(4, ' '));
    p->marker_indices(sparse_indices);
}

std::vector<double> target_line(snr_values.size(), target_fdr);
auto p_t = plot(snr_values, target_line);
p_t->line_style("--");
p_t->color("red");
p_t->line_width(line_width);
p_t->display_name("tFDR = " + format_double(target_fdr) + std::string(6, ' '));

xlabel("SNR");
ylabel("FDR");
xlim({min_val, max_val});
ylim({0.0, 0.5});
xticks(x_tick_pos);
xticklabels(x_tick_labels);
legend()->location(legend::general_alignment::topright);
save(dir + "sim_snr_fdr_tfdr_01" + win + ".pdf");


    // =========================================================
    // FIGURE 2: TPR vs SNR
    // =========================================================
    auto f2 = figure(true);

    // Use pdfcairo for better TrueType font embedding (not Type 3)
    f2->backend()->run_command("set terminal pdfcairo enhanced color font 'Times-Roman,15' size 5.5in,3.5in");

    // Set individual font sizes for each element
    f2->backend()->run_command("set xlabel font 'Times-Roman,17'");    // X-axis label
    f2->backend()->run_command("set ylabel font 'Times-Roman,17'");    // Y-axis label
    f2->backend()->run_command("set xtics font 'Times-Roman,15'");     // X-axis tick labels
    f2->backend()->run_command("set ytics font 'Times-Roman,15'");     // Y-axis tick labels
    f2->backend()->run_command("set key font 'Times-Roman,17'");       // Legend font
//    f2->size(fig_size.first, fig_size.second);
//    f2->font_size(font_size);
//    f2->font(font_name);
    hold(on);
    grid(on);
    colororder(custom_colors);
    gca()->position(default_pos);

    for (size_t i = 0; i < solver_names.size(); ++i) {
        auto p = plot(snr_values, tpr_results[i]);
        p->line_width(line_width);
        p->line_style("-");
        p->marker(markers[i % markers.size()]);
        p->marker_size(marker_size);
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
    save(dir + "sim_snr_tpr_tfdr_01" + win + ".pdf");


    // =========================================================
    // FIGURE 3: Average L vs SNR
    // =========================================================
    auto f3 = figure(true);
//    f3->size(fig_size.first, fig_size.second);
//    f3->font_size(font_size);
//    f3->font(font_name);

    // Use pdfcairo for better TrueType font embedding (not Type 3)
    f3->backend()->run_command("set terminal pdfcairo enhanced color font 'Times-Roman,15' size 5.5in,3.5in");

    // Set individual font sizes for each element
    f3->backend()->run_command("set xlabel font 'Times-Roman,17'");    // X-axis label
    f3->backend()->run_command("set ylabel font 'Times-Roman,17'");    // Y-axis label
    f3->backend()->run_command("set xtics font 'Times-Roman,15'");     // X-axis tick labels
    f3->backend()->run_command("set ytics font 'Times-Roman,15'");     // Y-axis tick labels
    f3->backend()->run_command("set key font 'Times-Roman,17'");       // Legend font

    hold(on);
    grid(on);
    colororder(custom_colors);
    gca()->position(default_pos);

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
        p->line_width(line_width);
        p->line_style("-");
        p->marker(markers[i % markers.size()]);
        p->marker_size(marker_size);
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
    save(dir + "sim_snr_avgL_tfdr_01" + win + ".pdf");


    // =========================================================
    // FIGURE 4: Average T vs SNR
    // =========================================================

    // Only for no stagnation detection
    //std::vector<std::string> avgT_labels(11, "");
    //avgT_labels[2] = "20";
    //avgT_labels[4] = "40";
    //avgT_labels[6] = "60";
    //avgT_labels[8] = "80";
    //avgT_labels[10] = "100";

    if (true) {
        auto f4 = figure(true);
//        f4->size(fig_size.first, fig_size.second);
//        f4->font_size(font_size);
//        f4->font(font_name);

        // Use pdfcairo for better TrueType font embedding (not Type 3)
        f4->backend()->run_command("set terminal pdfcairo enhanced color font 'Times-Roman,15' size 5.5in,3.5in");

        // Set individual font sizes for each element
        f4->backend()->run_command("set xlabel font 'Times-Roman,17'");    // X-axis label
        f4->backend()->run_command("set ylabel font 'Times-Roman,17'");    // Y-axis label
        f4->backend()->run_command("set xtics font 'Times-Roman,15'");     // X-axis tick labels
        f4->backend()->run_command("set ytics font 'Times-Roman,15'");     // Y-axis tick labels
        f4->backend()->run_command("set key font 'Times-Roman,17'");       // Legend font

        hold(on);
        grid(on);
        colororder(custom_colors);
        gca()->position(default_pos);

        for (std::size_t i = 0; i < solver_names.size(); ++i) {
            auto p = plot(snr_values, avg_T_results[i]);
            p->line_width(line_width);
            p->line_style("-");
            p->marker(markers[i % markers.size()]);
            p->marker_size(marker_size);
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
        // Only for no stagnation detection
        // yticks(matplot::iota(0, 10, 100));
        // yticklabels(avgT_labels);
        legend()->location(legend::general_alignment::bottomright);
        save(dir + "sim_snr_avgT_tfdr_01" + win + ".pdf");
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
        "TLARS", "TLASSO", "TENET",
        "TSTEPWISE", "TOMP",
        "TGP", "TACGP"
    };

    // Data: from File
    // ----------------------------------------------------------
    // FDR Data -> insert your FDR data here
std::vector<std::vector<double>> fdr_results = {
    {0.0153, 0.0278, 0.0538, 0.0841, 0.1044, 0.1268, 0.1327, 0.1250, 0.1368, 0.1320, 0.1388, 0.1274, 0.1258, 0.1275, 0.1313, 0.1233, 0.1208, 0.1188, 0.1225, 0.1183, 0.1047}, // TLARS
    {0.0240, 0.0362, 0.0749, 0.0883, 0.1067, 0.1354, 0.1372, 0.1254, 0.1370, 0.1298, 0.1350, 0.1327, 0.1293, 0.1242, 0.1261, 0.1302, 0.1152, 0.1178, 0.1218, 0.1166, 0.1094}, // TLASSO
    {0.0200, 0.0368, 0.0638, 0.1105, 0.1175, 0.1295, 0.1506, 0.1425, 0.1485, 0.1412, 0.1535, 0.1480, 0.1471, 0.1430, 0.1429, 0.1521, 0.1510, 0.1514, 0.1457, 0.1499, 0.1757}, // TENET
    {0.0180, 0.0148, 0.0429, 0.0618, 0.0734, 0.0574, 0.0589, 0.0590, 0.0661, 0.0555, 0.0448, 0.0498, 0.0385, 0.0463, 0.0392, 0.0364, 0.0341, 0.0314, 0.0333, 0.0318, 0.0225}, // TSTEPWISE
    {0.0110, 0.0237, 0.0308, 0.0574, 0.0590, 0.0702, 0.0579, 0.0595, 0.0490, 0.0553, 0.0460, 0.0523, 0.0414, 0.0391, 0.0375, 0.0333, 0.0369, 0.0350, 0.0311, 0.0306, 0.0229}, // TOMP
    {0.0080, 0.0240, 0.0448, 0.0582, 0.0587, 0.0507, 0.0572, 0.0686, 0.0544, 0.0531, 0.0504, 0.0467, 0.0399, 0.0394, 0.0396, 0.0361, 0.0321, 0.0339, 0.0355, 0.0319, 0.0251}, // TGP
    {0.0060, 0.0248, 0.0258, 0.0552, 0.0536, 0.0669, 0.0639, 0.0581, 0.0624, 0.0571, 0.0535, 0.0480, 0.0425, 0.0410, 0.0351, 0.0380, 0.0342, 0.0342, 0.0311, 0.0306, 0.0240}  // TACGP
};


    // TPR Data -> insert your TPR data here
std::vector<std::vector<double>> tpr_results = {
    {0.0040, 0.0148, 0.0688, 0.1386, 0.2652, 0.3734, 0.4986, 0.6152, 0.6834, 0.7544, 0.8072, 0.8636, 0.8892, 0.9130, 0.9362, 0.9490, 0.9650, 0.9710, 0.9810, 0.9850, 1.0000}, // TLARS
    {0.0032, 0.0142, 0.0678, 0.1472, 0.2576, 0.3786, 0.5018, 0.6056, 0.6906, 0.7488, 0.8098, 0.8504, 0.8756, 0.9034, 0.9276, 0.9490, 0.9696, 0.9692, 0.9798, 0.9848, 1.0000}, // TLASSO
    {0.0040, 0.0210, 0.0676, 0.1600, 0.2712, 0.3806, 0.4884, 0.6014, 0.6898, 0.7476, 0.8144, 0.8368, 0.8726, 0.9130, 0.9250, 0.9404, 0.9546, 0.9588, 0.9734, 0.9766, 1.0000}, // TENET
    {0.0056, 0.0170, 0.0430, 0.1050, 0.1756, 0.3242, 0.4388, 0.6074, 0.6882, 0.8162, 0.8854, 0.9108, 0.9520, 0.9594, 0.9720, 0.9838, 0.9852, 0.9870, 0.9906, 0.9922, 1.0000}, // TSTEPWISE
    {0.0022, 0.0118, 0.0404, 0.1000, 0.1744, 0.2958, 0.4418, 0.5734, 0.7048, 0.8008, 0.8806, 0.9098, 0.9482, 0.9660, 0.9718, 0.9814, 0.9840, 0.9868, 0.9912, 0.9926, 0.9998}, // TOMP
    {0.0024, 0.0138, 0.0460, 0.1002, 0.1886, 0.3036, 0.4428, 0.5614, 0.6928, 0.7904, 0.8782, 0.9226, 0.9478, 0.9548, 0.9710, 0.9784, 0.9882, 0.9850, 0.9894, 0.9936, 1.0000}, // TGP
    {0.0022, 0.0132, 0.0472, 0.1008, 0.1860, 0.3094, 0.4440, 0.5674, 0.6862, 0.8138, 0.8638, 0.9216, 0.9436, 0.9616, 0.9778, 0.9796, 0.9858, 0.9862, 0.9920, 0.9906, 1.0000}  // TACGP
};


    // Avg L Data -> insert your Avg L data here
std::vector<std::vector<double>> avg_L_results = {
    {3.2800, 4.7680, 6.5540, 7.4100, 7.1640, 6.2900, 5.4580, 5.0840, 4.4140, 3.9660, 3.5360, 3.1860, 2.9280, 2.5560, 2.5760, 2.3520, 2.1840, 2.0800, 1.9540, 1.7960, 1.2180}, // TLARS
    {3.1020, 4.6340, 6.7160, 7.3520, 7.2420, 6.3620, 5.7420, 5.0480, 4.2220, 3.9560, 3.6340, 3.2440, 2.9540, 2.6260, 2.4720, 2.2320, 2.1280, 2.0720, 1.8300, 1.8620, 1.2120}, // TLASSO
    {3.1720, 5.1760, 6.5920, 7.3640, 7.0500, 6.3340, 5.7640, 4.9460, 4.4240, 3.8460, 3.6680, 3.1200, 2.9560, 2.6120, 2.6060, 2.3520, 2.1760, 2.0020, 1.9680, 1.8360, 1.2340}, // TENET
    {3.2480, 5.2100, 6.5820, 7.4520, 6.9960, 6.0100, 5.3240, 4.3420, 3.8960, 2.9340, 2.5580, 2.2580, 1.7540, 1.6180, 1.4740, 1.3860, 1.3120, 1.2480, 1.1900, 1.1840, 1.1460}, // TSTEPWISE
    {3.1400, 4.8040, 6.7200, 7.4640, 7.0360, 6.2760, 5.5340, 4.5000, 3.8360, 3.1740, 2.6120, 2.1840, 1.8780, 1.6060, 1.4120, 1.3460, 1.2860, 1.2860, 1.2500, 1.2020, 1.1060}, // TOMP
    {3.1380, 5.3820, 6.7960, 7.4880, 7.0060, 6.2640, 5.4580, 4.6040, 3.8300, 3.0380, 2.5160, 2.1980, 1.8620, 1.7340, 1.4780, 1.4140, 1.3100, 1.2640, 1.2220, 1.1860, 1.1040}, // TGP
    {3.1720, 5.1620, 6.8500, 7.2040, 7.0520, 6.0680, 5.2100, 4.5660, 3.8500, 3.0900, 2.5560, 2.1280, 1.8080, 1.7580, 1.4960, 1.3460, 1.2740, 1.2700, 1.2300, 1.2260, 1.1300}  // TACGP
};


    // Avg T Data -> insert your Avg T data here
std::vector<std::vector<double>> avg_T_results = {
    {3.9960, 3.7280, 4.2180, 4.6880, 5.8300, 6.2980, 6.7680, 7.3240, 7.0760, 6.9460, 6.8260, 6.4040, 6.3200, 5.9380, 6.0940, 5.8400, 5.7420, 5.4840, 5.3260, 5.3060, 4.5440}, // TLARS
    {4.1000, 3.7680, 4.2140, 4.8360, 5.7540, 6.3020, 6.8120, 7.1260, 7.0740, 6.8140, 7.0620, 6.6920, 6.4160, 6.0780, 5.8680, 5.6820, 5.5680, 5.5100, 5.2020, 5.3140, 4.5400}, // TLASSO
    {3.9800, 3.9120, 4.2620, 5.0920, 5.7680, 6.2940, 6.9620, 7.1020, 7.4020, 6.8300, 7.1520, 6.5020, 6.2620, 6.0580, 6.1100, 5.7520, 5.7140, 5.4720, 5.4680, 5.2400, 4.4680}, // TENET
    {6.5500, 5.8120, 5.6040, 5.7380, 6.0880, 6.8460, 7.1940, 7.3420, 7.2740, 7.0020, 7.0520, 6.8480, 6.9700, 6.8620, 6.9800, 6.9800, 7.1380, 7.0020, 6.7900, 6.9120, 7.0720}, // TSTEPWISE
    {6.3940, 5.7960, 5.5860, 5.9080, 6.1460, 6.6400, 7.2580, 7.1340, 7.2120, 7.3220, 6.8500, 7.0040, 7.1000, 6.9360, 6.9280, 6.9220, 6.9200, 6.9460, 6.9580, 6.8160, 7.0220}, // TOMP
    {6.2100, 5.5880, 5.6340, 5.7540, 6.0760, 6.7880, 7.1880, 7.1780, 7.2060, 6.9580, 6.9240, 6.8340, 6.9140, 6.8100, 6.8620, 6.9100, 6.9020, 7.0160, 6.6980, 6.8920, 6.9400}, // TGP
    {6.0960, 5.8540, 5.5620, 5.6700, 6.1880, 6.5900, 6.9520, 7.2340, 7.1240, 7.0280, 6.8720, 6.8080, 6.8760, 6.7740, 6.8640, 6.8540, 6.8360, 6.8860, 6.8240, 6.8100, 7.0100}  // TACGP
};


    // Set stagnation window size
    std::size_t window_size = 7;


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
