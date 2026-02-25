%_________________________________________________________________________
%   Original Author and programmer: Mohd Mawardi Saari, 
%                               Universiti Malaysia Pahang
%                               mmawardi@umpsa.edu.my
%_________________________________________________________________________

function [chi_real, chi_imag] = compute_BrownianRelaxation(DH, freq, visco, kB, T, kaiInfinite)
    
    VH = (4/3)*pi*(DH/2)^3;
    tauB = 3*visco*VH/(kB*T); % Brownian relaxation time
    omega = 2 * pi * freq;
    
    chi_real = (1./(1 + (omega*tauB).^2))*(1-kaiInfinite)+kaiInfinite;
    chi_imag = ((omega*tauB)./(1 + (omega*tauB).^2))*(1-kaiInfinite);

end

