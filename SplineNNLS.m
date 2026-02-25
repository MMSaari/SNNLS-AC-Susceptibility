%_________________________________________________________________________
%   Original Author and programmer: Mohd Mawardi Saari, 
%                               Universiti Malaysia Pahang
%                               mmawardi@umpsa.edu.my
%_________________________________________________________________________

function [DH_grid, DH_dist, M_fitReal, M_fitImag, Residual_SPNNLS] = SplineNNLS(freqs, M_RealExp, M_ImagExp, DH_min, DH_max, knots_per_decade, spline_order)
% Fits the brownian relaxation using SNNLS   

    
    % Step 0: Combine M_real and M_imag in a data array
    M_exp=[M_RealExp';M_ImagExp'];

    %set infinite susceptibility
    kaiInfinite=0.9*M_RealExp(end);

    % Step 1: Generate core size grid and spline basis

        % Call the function to retrieve constants
    [kB, T, visco] = physical_constants();

    DH_int= 1e-9; %unit in nm
    
    DH_grid = (DH_min:DH_int:DH_max);
    num_decades = log10(DH_max / DH_min);% Compute the number of decades
    n_knots = round(num_decades * knots_per_decade);% Compute total number of knots
    knots = logspace(log10(DH_min), log10(DH_max), n_knots);  % Spline knots
    %spline_order = 3;  % Quadratic (3) spline

    % Step 2: Evaluate spline basis functions on DH_grid
    B = bspline_basis(knots, spline_order, DH_grid);


    % Step 3: Construct design matrix A for Langevin curve
    A_Real = zeros(length(freqs), size(B, 2)); % Initialize design matrix
    A_Imag = zeros(length(freqs), size(B, 2)); % Initialize design matrix

    for i = 1:size(B, 2)
        temp_w = B(:, i); % Weight (d distribution)
        [yReal_weighted, yImag_weighted]=computeRelaxationWeighted(temp_w, DH_grid, visco, freqs, kB, T, kaiInfinite);
        A_Real(:, i) = yReal_weighted;
        A_Imag(:, i) = yImag_weighted;
    end
    A = [A_Real; A_Imag]; % Concatenate A_real on top of A_imag


    % Step 4: Solve for non-negative spline coefficients using NNLS
    spline_coeffs = lsqnonneg(A, M_exp);

    % Step 5: Compute the fitted Langevin curve and fitted distribution
    M_fit = A * spline_coeffs;
    M_fitReal= M_fit(1:length(freqs));
    M_fitImag= M_fit(length(freqs)+1 : 2*length(freqs));
    DH_dist=B * spline_coeffs;
    DH_grid=DH_grid.';
    Residual_SPNNLS= M_exp - M_fit;

    
end

% Function to compute weighted Relaxation
function [yReal_weighted, yImag_weighted]= computeRelaxationWeighted(weights, DH_grid, visco, freqs, kB, T, kaiInfinite)
    DH_gridDiff = diff(DH_grid);
    yReal_weighted = zeros(size(freqs));
    yImag_weighted = zeros(size(freqs));
    for i = 1:length(DH_grid)-1
        [chi_real, chi_imag] = compute_BrownianRelaxation(DH_grid(i), freqs, visco, kB, T, kaiInfinite);
        yReal_weighted = yReal_weighted + weights(i) * DH_gridDiff(i)*chi_real;
        yImag_weighted = yImag_weighted + weights(i) * DH_gridDiff(i)*chi_imag;
    end
end


% Generate B-spline basis matrix
function B = bspline_basis(knots, order, DH_grid)
    B = spcol(knots, order, DH_grid); % Compute spline basis functions
    B = B(1:length(DH_grid), :); % Adjust size to match d grid
end