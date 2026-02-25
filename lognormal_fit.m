function [DH_grid, DH_pdf, M_RealFit, M_ImagFit, Residuals] = lognormal_fit(freqs, M_RealExp, M_ImagExp, DH_min, DH_max)

digits(100);
disp('Fitting lognormal mixture model using experimental susceptibility data...');
tic

%_________________________________________________________________________
%   Original Author and programmer: Mohd Mawardi Saari, 
%                               Universiti Malaysia Pahang
%                               mmawardi@umpsa.edu.my
%_________________________________________________________________________

% Call the function to retrieve constants
[kB, T, visco] = physical_constants();

%set infinite susceptibility
kaiInfinite=0.9*M_RealExp(end);

% Define function to compute susceptibility from a lognormal mixture model
lognormal_model = @(p, freqs) compute_susceptibility_from_lognormal(freqs, p, DH_min, DH_max, visco, kB, T, kaiInfinite);

% Initial parameter guesses [mu1, sigma1, mu2, sigma2, w1]
init_guess = [log(30e-9), 0.2, log(70e-9), 0.1, 0.5];

% Parameter bounds
lb = [log(DH_min), 0.01, log(1*(DH_min+DH_max)/2), 0.01, 0];  % Lower bounds
ub = [log(1.0*(DH_min+DH_max)/2), 1, log(DH_max), 1, 1];  % Upper bounds

% Combine real and imaginary parts for fitting
M_Exp = [M_RealExp'; M_ImagExp'];

% Fit using nonlinear least squares
opts = optimoptions('lsqcurvefit', 'Display', 'iter', 'SpecifyObjectiveGradient', false);
[fit_params, resnorm, residuals, ~, ~, ~, jacobian] = lsqcurvefit(@(p, freqs) lognormal_model(p, freqs), init_guess, freqs, M_Exp, lb, ub, opts);

% Compute confidence intervals (95%) using Jacobian
ci = nlparci(fit_params, residuals, 'jacobian', jacobian);

% Construct hydrodynamic size distribution using fitted parameters
DH_grid = linspace(DH_min, DH_max, 100); % Discretize core sizes

mu1_fit = fit_params(1);
sigma1_fit = fit_params(2);
mu2_fit = fit_params(3);
sigma2_fit = fit_params(4);
w1_fit = fit_params(5);

DH_pdf = w1_fit * lognpdf(DH_grid, mu1_fit, sigma1_fit) + (1 - w1_fit) * lognpdf(DH_grid, mu2_fit, sigma2_fit);
DH_pdf = DH_pdf / trapz(DH_grid, DH_pdf);  % Normalize PDF

% Compute fitted susceptibility using fitted parameters
M_Fit = lognormal_model(fit_params, freqs);
M_RealFit = M_Fit(1:length(freqs));
M_ImagFit = M_Fit(length(freqs) + 1:end);

% Compute residuals
% Residuals_Real = M_RealExp - M_RealFit;
% Residuals_Imag = M_ImagExp - M_ImagFit;

% Store results in a structured output
% fit_results.params = fit_params;
% fit_results.confidence_intervals = ci;
% fit_results.DH_grid = DH_grid;
% fit_results.DH_pdf = DH_pdf;
% fit_results.M_RealFit = M_RealFit;
% fit_results.M_ImagFit = M_ImagFit;
% fit_results.Residuals_Real = Residuals_Real;
% fit_results.Residuals_Imag = Residuals_Imag;
% fit_results.resnorm = resnorm;
% DH_grid=DH_grid';
% DH_pdf = DH_pdf';
Residuals= M_Exp - M_Fit;

disp('Fitting completed.');
toc

end

function M_Fit = compute_susceptibility_from_lognormal(freqs, p, DH_min, DH_max, visco, kB, T, kaiInfinite)
    % Extract parameters from p
    mu1 = p(1);
    sigma1 = p(2);
    mu2 = p(3);
    sigma2 = p(4);
    w1 = p(5);  % Weight of the first distribution

    % Define the hydrodynamic size grid
    DH_grid = linspace(DH_min, DH_max, 100);  % Discretized core sizes
    DH_grid_Diff = diff(DH_grid);
    
    % Compute probability density function for the mixture of lognormals
    DH_pdf = w1 * lognpdf(DH_grid, mu1, sigma1) + (1 - w1) * lognpdf(DH_grid, mu2, sigma2);
    DH_pdf = DH_pdf / trapz(DH_grid, DH_pdf);  % Normalize the PDF

    % Initialize susceptibility components
    M_RealFit = zeros(length(freqs), 1);
    M_ImagFit = zeros(length(freqs), 1);

    % Loop over all frequency points
    for j = 1:length(freqs)
        for i = 1:length(DH_grid) - 1
            % Compute real and imaginary susceptibility using Brownian relaxation
            [chi_real, chi_imag] = compute_BrownianRelaxation(DH_grid(i), freqs(j), visco, kB, T, kaiInfinite);

            % Integrate using the PDF
            M_RealFit(j) = M_RealFit(j) + DH_pdf(i) * DH_grid_Diff(i) * chi_real;
            M_ImagFit(j) = M_ImagFit(j) + DH_pdf(i) * DH_grid_Diff(i) * chi_imag;
        end
    end

    % Concatenate real and imaginary parts for fitting
    M_Fit = [M_RealFit; M_ImagFit];

end
