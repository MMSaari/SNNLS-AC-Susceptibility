%_________________________________________________________________________
%   Original Author and programmer: Mohd Mawardi Saari, 
%                               Universiti Malaysia Pahang
%                               mmawardi@umpsa.edu.my
%_________________________________________________________________________

function [DH_sizes, DH_pdf, freqs, M_real, M_imag] = simulate_AC_susceptibility(DH_min, DH_max, DH_int, freq_min, freq_max, freq_no, noise_std)
    rng(123);% Set random seed for reproducibility
    DH_sizes=(DH_min:DH_int:DH_max)'; % Hydrodynamic core sizes (nm)
    DH_sizes_Delta=diff(DH_sizes);
    
    % % Lognormal distribution 1 (smaller peak)
    % mu1 = log(30e-9);  % Log-mean (~30 nm)
    % sigma1 = 0.2;  % Log-standard deviation
    % Ratioweight = 0.5;  % Ratio weight
    % 
    % % Lognormal distribution 2 (larger peak)
    % mu2 = log(70e-9);  % Log-mean (~70 nm)
    % sigma2 = 0.1;   % Log-standard deviation
    
    %call parameters of DH distribution
    [mu1, sigma1, Ratioweight, mu2, sigma2] = DHdist_parameters();

    %set infinite susceptibility
    kaiInfinite=0;
    
    
    % Compute combined distribution
     DH_pdf = Ratioweight*lognpdf(DH_sizes, mu1, sigma1)+(1-Ratioweight)*lognpdf(DH_sizes, mu2, sigma2);
  
    % Constants
    [kB, T, visco] = physical_constants();

    % Frequency range (Hz)
    freqs = logspace(log10(freq_min), log10(freq_max), freq_no);
    
    M_real = zeros(size(freqs));
    M_imag = zeros(size(freqs));

    for j=1:length(freqs)
    
        for i = 1:length(DH_sizes)-1
            
            [chi_real, chi_imag] = compute_BrownianRelaxation(DH_sizes(i), freqs(j), visco, kB, T, kaiInfinite);
            
            M_real(j) = M_real(j) + DH_pdf(i) *DH_sizes_Delta(i)* chi_real;
            M_imag(j) = M_imag(j) + DH_pdf(i) *DH_sizes_Delta(i)* chi_imag;
        end

    end
    
    %noise_std = 0.00000001;  % Standard deviation for Gaussian noise
    
    % Apply noise to M_real and M_imag
    M_real = M_real + noise_std * randn(size(M_real));
    M_imag = M_imag + noise_std * randn(size(M_imag));
end