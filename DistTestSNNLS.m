%_________________________________________________________________________
%   Original Author and programmer: Mohd Mawardi Saari, 
%                               Universiti Malaysia Pahang
%                               mmawardi@umpsa.edu.my
%_________________________________________________________________________


function DistTestSNNLS(freq_min, freq_max, DH_min, DH_max)
    rng(123); % Set random seed for reproducibility

    % Constants
    [kB, T, visco] = physical_constants();
    
    % Define spline order and number of knots
    spline_order = 3;    
    n_knots = 10;     

    % Fixed parameters
    F_SamplingDensity = 8;
    noiseSTD = 0.001;  

    % Define two cases, case 1: closely mixed distribution, case 2
    % separated and has close distribution peak height
    cases = struct(...
        'distributionRatio', {0.5, 0.1}, ...
        'mu_lognorm1', {log(40e-9), log(20e-9)}, ...
        'sigma_lognorm1', {0.2, 0.2}, ...
        'mu_lognorm2', {log(70e-9), log(70e-9)}, ...
        'sigma_lognorm2', {0.2, 0.2} ...
    );

    % In both cases, the susceptibility at extreme high frequency is reduced to 0
    % (X_infinity = 0)
    kaiInfinite =0;

    % Generate frequency data
    F_NoDecade = log10(freq_max / freq_min);
    Freqs = logspace(log10(freq_min), log10(freq_max), F_SamplingDensity * F_NoDecade);
    

    % Generate core size distribution
    DH_size_interval = 1e-9; 
    true_DH_diam = (DH_min:DH_size_interval:DH_max);
    true_DH_diamDelta = diff(true_DH_diam);

    methodNames = {'SNNLS','NNLS', 'Tikhonov-NNLS', 'LogNormalFit'};

    % Loop through cases
    for caseIdx = 1:2
        % Extract parameters
        rng(123); % Set random seed for reproducibility
        distributionRatio = cases(caseIdx).distributionRatio;
        mu_lognorm1 = cases(caseIdx).mu_lognorm1;
        sigma_lognorm1 = cases(caseIdx).sigma_lognorm1;
        mu_lognorm2 = cases(caseIdx).mu_lognorm2;
        sigma_lognorm2 = cases(caseIdx).sigma_lognorm2;

        % Compute lognormal weights
        DH_pdf = distributionRatio * lognpdf(true_DH_diam, mu_lognorm1, sigma_lognorm1) + ...
                            (1 - distributionRatio) * lognpdf(true_DH_diam, mu_lognorm2, sigma_lognorm2);

                    
        M_real = zeros(size(Freqs));
        M_imag = zeros(size(Freqs));
    
        for j=1:length(Freqs)
        
            for i = 1:length(true_DH_diam)-1
                
                [chi_real, chi_imag] = compute_BrownianRelaxation(true_DH_diam(i), Freqs(j), visco, kB, T, kaiInfinite);
                
                M_real(j) = M_real(j) + DH_pdf(i) *true_DH_diamDelta(i)* chi_real;
                M_imag(j) = M_imag(j) + DH_pdf(i) *true_DH_diamDelta(i)* chi_imag;
            end
    
        end
    
        %noise_std = 0.00000001;  % Standard deviation for Gaussian noise
        
        % Apply noise to M_real and M_imag
        M_real = M_real + noiseSTD * randn(size(M_real));
        M_imag = M_imag + noiseSTD * randn(size(M_imag));
              
        % Fit using different methods
        %SNNLS
        [d_grid_SNNLS, diam_dist_SNNLS, M_fitReal_SNNLS, M_fitImag_SNNLS, Residual_SNNLS] = ...
                SplineNNLS(Freqs, M_real, M_imag, DH_min, DH_max, n_knots, spline_order);
        
        %NNLS
        [d_grid_NNLS, diam_dist_NNLS, M_fitReal_NNLS, M_fitImag_NNLS, Residual_NNLS] = NNLS(Freqs, M_real, M_imag, DH_min, DH_max);

        %Tikhonov-LFlat-NNLS
        [d_grid_TikNNLS, diam_dist_TikNNLS, M_fitReal_TikNNLS, M_fitImag_TikNNLS, Residual_TikNNLS] = Tikhonov_LFlat(Freqs, M_real, M_imag, DH_min, DH_max);

        %Log Normal Fit
        [d_grid_LogNormFit, diam_dist_LogNormFit, M_fitReal_LogNormFit, M_fitImag_LogNormFit, Residual_LogNormFit] = lognormal_fit(Freqs, M_real, M_imag, DH_min, DH_max);

        % Compute JS divergence
        JS_div_SNNLS = js_divergence(d_grid_SNNLS, diam_dist_SNNLS, ...
                          mu_lognorm1, sigma_lognorm1, mu_lognorm2, sigma_lognorm2, distributionRatio);

        JS_div_NNLS = js_divergence(d_grid_NNLS, diam_dist_NNLS, ...
                          mu_lognorm1, sigma_lognorm1, mu_lognorm2, sigma_lognorm2, distributionRatio);

        JS_div_TikNNLS = js_divergence(d_grid_TikNNLS, diam_dist_TikNNLS, ...
                          mu_lognorm1, sigma_lognorm1, mu_lognorm2, sigma_lognorm2, distributionRatio);

        JS_div_LogNormFit = js_divergence(d_grid_LogNormFit, diam_dist_LogNormFit, ...
                          mu_lognorm1, sigma_lognorm1, mu_lognorm2, sigma_lognorm2, distributionRatio);

        % Store residuals and JS divergence
        residuals = [norm(Residual_SNNLS), norm(Residual_NNLS), norm(Residual_TikNNLS), norm(Residual_LogNormFit)];
        js_divergences = [JS_div_SNNLS, JS_div_NNLS, JS_div_TikNNLS, JS_div_LogNormFit];

        disp(['Residual SNNLS: ', num2str(norm(Residual_SNNLS)), ', NNLS: ', num2str(norm(Residual_NNLS)),', Tikhonov-NNLS: ', num2str(norm(Residual_TikNNLS)),', LogNormalFit: ', num2str(norm(Residual_LogNormFit))]);
        disp(['JS Div. SNNLS: ', num2str(JS_div_SNNLS), ', NNLS: ', num2str(JS_div_NNLS),', Tikhonov-NNLS: ', num2str(JS_div_TikNNLS),', LogNormalFit: ', num2str(JS_div_LogNormFit)]);

        % % Plot: M_fit vs. Freqs with Residual Bar Plot
        figure('DefaultAxesFontSize', 14)
        hold on;
                    
        h1 = plot(Freqs, M_fitReal_SNNLS, '--', 'LineWidth', 1.5, 'DisplayName', 'SNNLS');
        h2 = plot(Freqs, M_fitReal_NNLS, '--', 'LineWidth', 1.5, 'DisplayName', 'NNLS');
        h3 = plot(Freqs, M_fitReal_TikNNLS, '--', 'LineWidth', 1.5, 'DisplayName', 'Tikhonov-NNLS');
        h4 = plot(Freqs, M_fitReal_LogNormFit, '--', 'LineWidth', 1.5, 'DisplayName', 'LogNormalFit');

        % Extract Colors from Line Plots
        color_SNNLS = get(h1, 'Color');
        color_NNLS = get(h2, 'Color');
        color_TikNNLS = get(h3, 'Color');
        color_LogNormFit = get(h4, 'Color');
        
        plot(Freqs, M_fitImag_SNNLS, '--', 'LineWidth', 1.5, 'Color', color_SNNLS, 'HandleVisibility', 'off');
        plot(Freqs, M_fitImag_NNLS, '--', 'LineWidth', 1.5, 'Color', color_NNLS, 'HandleVisibility', 'off');
        plot(Freqs, M_fitImag_TikNNLS, '--', 'LineWidth', 1.5, 'Color', color_TikNNLS, 'HandleVisibility', 'off');
        plot(Freqs, M_fitImag_LogNormFit, '--', 'LineWidth', 1.5, 'Color', color_LogNormFit, 'HandleVisibility', 'off');

        % % Simulated Real data
        plot(Freqs, M_real, 'ok', 'LineWidth', .5, 'MarkerSize', 4, 'DisplayName', 'Simulated $\mathrm{Re}[\chi]/\chi_{0}$');
        % % Simulated Imag data
        plot(Freqs, M_imag, '^k', 'LineWidth', .5, 'MarkerSize', 4, 'DisplayName', 'Simulated $\mathrm{Im}[\chi]/\chi_{0}$');

        % Set LaTeX interpreter for all legend entries
        legend('Interpreter', 'latex', 'Location', 'best');
        
               
        set(gca, 'YScale', 'linear','XScale', 'log');
        ylim([0, 1.2]); % Adjust limits 
        xlabel('Frequency (Hz)', 'Interpreter', 'latex');
        ylabel('Reconstructed $\chi/\chi_{0}$', 'Interpreter', 'latex');
        title(sprintf('M_{fit} vs. freq for Case %d', caseIdx));
        legend('Location', 'northwest');
        grid on;
        

        
        % Inset Residual Bar Plot
        inset_ax = axes('Position', [0.65, 0.6, 0.25, 0.25]);
        methodNames = categorical(methodNames, methodNames, 'Ordinal', true);
        % Compute ratios relative to SNNLS (assuming residuals(1) is SNNLS)
        ratios = residuals / residuals(1);
        b1 = bar(inset_ax, categorical(methodNames), residuals);
        % Check if 'b1' is an array and apply colors
        b1.FaceColor = 'flat'; % Enables individual bar coloring
        b1.CData = [color_SNNLS; color_NNLS;color_TikNNLS;color_LogNormFit]; % Assign extracted colors
        % Add ratio text annotations
        % Add ratio text annotations inside the bars
        for i = 2:length(residuals) % Skip SNNLS (first bar)
            text(i, residuals(i) * 0.5, sprintf('%.1fx', ratios(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
                'Color', 'k'); % White text for contrast
        end
        ylabel(inset_ax, 'Residual Mag.'); 
        set(inset_ax, 'YMinorGrid', 'on'); 
        
        hold off;


        % Plot: diam_dist vs. d_grid with JS Divergence Bar Plot
        figure('DefaultAxesFontSize', 14)
        hold on;
        
        h5 = plot(d_grid_SNNLS, diam_dist_SNNLS, '-', 'LineWidth', 2, 'DisplayName', 'SNNLS');
        h6 = plot(d_grid_NNLS, diam_dist_NNLS, '-', 'LineWidth', 2, 'DisplayName', 'NNLS');
        h7 = plot(d_grid_TikNNLS, diam_dist_TikNNLS, '-', 'LineWidth', 2, 'DisplayName', 'Tikhonov-NNLS');
        h8 = plot(d_grid_LogNormFit, diam_dist_LogNormFit, '-', 'LineWidth', 2, 'DisplayName', 'LogNormalFit');

        plot(true_DH_diam, DH_pdf, '--k', 'LineWidth', 1, 'DisplayName', 'Simulated Distribution');

        % Extract Colors from Line Plots
        color2_SNNLS = get(h5, 'Color');
        color2_NNLS = get(h6, 'Color');
        color2_TikNNLS = get(h7, 'Color');
        color2_LogNormFit = get(h8, 'Color');

        xlabel('Hydrodynamic Size (m)');
        ylabel('Size Distribution (a.u.)');
        title(sprintf('DH Distribution for Case %d', caseIdx));
        legend('Location', 'northeast');
        grid on;

        % Inset JS Divergence Bar Plot
        inset_ax2 = axes('Position', [0.7, 0.65, 0.2, 0.2]); 
        methodNames = categorical(methodNames, methodNames, 'Ordinal', true);
        ratios2 = js_divergences / js_divergences(1);
        
        b2 = bar(inset_ax2, categorical(methodNames), js_divergences, 'FaceColor', 'b');
        b2.FaceColor = 'flat'; % Enables individual bar coloring
        b2.CData = [color_SNNLS; color_NNLS;color_TikNNLS;color_LogNormFit]; % Assign extracted colors
        %ylim([0, 1.1 * max(js_divergences)]); % Adjust limits slightly above max value
        %yticks(linspace(0, max(js_divergences), 5)); % Force tick marks at 5 intervals
        set(inset_ax2, 'YMinorGrid', 'on', 'Box', 'on'); % Add box around inset
        % Increase the number of ticks
        %set(inset_ax2, 'YTickMode', 'auto'); 
        for i = 2:length(js_divergences) % Skip SNNLS (first bar)
            text(i, js_divergences(i)*0.5 + 0.1*max(js_divergences), sprintf('%.1fx', ratios2(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
                'Color', 'k'); % black text for contrast
        end
        set(inset_ax2, 'YMinorGrid', 'on'); 
        ylabel(inset_ax2, '$D_{JS}$', 'Interpreter', 'latex');
        %title(inset_ax2, 'JS Divergence Comparison');
        hold off;
    end
end
