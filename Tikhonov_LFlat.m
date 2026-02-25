%_________________________________________________________________________
%   Original Author and programmer: Mohd Mawardi Saari, 
%                               Universiti Malaysia Pahang
%                               mmawardi@umpsa.edu.my
%_________________________________________________________________________


function [DH_grid, DH_dist, M_RealFit, M_ImagFit, Residuals] = Tikhonov_LFlat(freqs, M_RealExp, M_ImagExp, DH_min, DH_max)

%digits(100);

disp('NNLS with Tikhonov Regularization using L-Flat Method...')

% Call the function to retrieve constants
[kB, T, visco] = physical_constants();

%set infinite susceptibility
kaiInfinite=0.9*M_RealExp(end);

% Number of magnetic core bins
DH_No = 75;

% Linearly spaced core size values
DH_grid = linspace(DH_min, DH_max, DH_No);

core_start = 1e-9;
NewDHList = DH_grid(DH_grid >= core_start);
%NewDHList_Diff = diff(NewDHList);

NewDHNo = length(NewDHList);

Kd = 74; % Number of sections
Sd = 1;  % Number of subdomains in each section

NewDHList_Diff = diff(NewDHList(1:Kd*Sd+1));

% Allocate memory
NewDH = zeros(Kd,1);
Av_DH_dist = [];
M_RealFit = zeros(length(freqs),1);
M_ImagFit = zeros(length(freqs),1);

% L-Flat Analysis: Define lambda range
lambdas = logspace(-10, -8, 100);
residual_norms = zeros(size(lambdas));
solution_norms = zeros(size(lambdas));

for kk = 1:Sd
    for ll = 1:Kd
        DH_element = kk + (ll-1) * Sd;
        NewDH(ll) = NewDHList(DH_element);
        NewDHDiff(ll) = NewDHList_Diff(DH_element);
    end

    % Construct design matrix A for Langevin curve
    A_Real = zeros(length(freqs), length(NewDH));
    A_Imag = zeros(length(freqs), length(NewDH));
    
    for ii = 1:length(freqs)
        for i = 1:length(NewDH)
            [chi_real, chi_imag] = compute_BrownianRelaxation(NewDH(i), freqs(ii), visco, kB, T, kaiInfinite);
            A_Real(ii, i) = A_Real(ii, i) + NewDHDiff(i) * chi_real;
            A_Imag(ii, i) = A_Imag(ii, i) + NewDHDiff(i) * chi_imag;
        end
    end

    % Regularization matrix (First-order difference)
    L = diag(ones(length(NewDH)-1,1), 1) - eye(length(NewDH));
    L = L(1:end-1, :);

    % Combine regularization into system
    M_Exp = [M_RealExp'; M_ImagExp'];
    A = [A_Real; A_Imag];
    b = [M_Exp; zeros(size(L,1), 1)];

    % Compute residual and solution norms
    for idx = 1:length(lambdas)
        lambda = lambdas(idx);
        DHEstimate = lsqnonneg([A; lambda * L], b) / Sd;
        
        residual_norms(idx) = norm(A * DHEstimate - M_Exp);
        solution_norms(idx) = norm(L * DHEstimate);
    end
    
    % Compute log-log slope and find where curve flattens
    slopes = diff(log10(solution_norms)) ./ diff(log10(residual_norms));
    [~, optimal_idx] = min(abs(slopes)); % Select the minimum slope point (flattest region)
    optimal_lambda = lambdas(optimal_idx);
    
    % Final solution with optimal lambda
    DHEstimate = lsqnonneg([A; optimal_lambda * L], b) / Sd;

    % Store solution
    solution(:,:,kk) = cat(2, NewDH, DHEstimate);
    Av_DH_dist = sortrows([Av_DH_dist; solution(:,:,kk)], 1);
end

% Extract final distributions
DH_grid = Av_DH_dist(:,1);
DH_dist = Av_DH_dist(:,2);

% Fit magnetization response
for i = 1:length(freqs)
    for ii = 1:length(DH_grid)
        [chi_real, chi_imag] = compute_BrownianRelaxation(DH_grid(ii), freqs(i), visco, kB, T, kaiInfinite);
        M_RealFit(i) = M_RealFit(i) + DH_dist(ii) * NewDHList_Diff(ii) * chi_real;
        M_ImagFit(i) = M_ImagFit(i) + DH_dist(ii) * NewDHList_Diff(ii) * chi_imag;
    end
end

% Compute normalized residual
M_Fit=[M_RealFit;M_ImagFit];
Residuals = M_Exp- M_Fit;

% Plot the L-Flat Curve
% figure;
% loglog(residual_norms, solution_norms, '-o', 'LineWidth', 1.5);
% hold on;
% plot(residual_norms(optimal_idx), solution_norms(optimal_idx), 'ro', 'MarkerFaceColor', 'r');
% xlabel('Residual Norm ||A x - b||');
% ylabel('Solution Norm ||Lx||');
% title('L-Flat Method for Optimal Lambda Selection');
% legend('L-Curve', 'Optimal \lambda');
% grid on;

disp(['Optimal Lambda Selected using L-Flat Method: ', num2str(optimal_lambda)]);

end
