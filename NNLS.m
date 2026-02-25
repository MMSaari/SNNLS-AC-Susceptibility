%_________________________________________________________________________
%   Original Author and programmer: Mohd Mawardi Saari, 
%                               Universiti Malaysia Pahang
%                               mmawardi@umpsa.edu.my
%_________________________________________________________________________
%nnls inversion algorithm

function [DH_grid, DH_dist, M_RealFit, M_ImagFit, Residuals] = NNLS(freqs, M_RealExp, M_ImagExp, DH_min, DH_max)

digits(100);

disp('NNLS method is now estimating the global optimum of the problem....')
tic
%to reduce the calculation time, divide the calculation into several region
%define Kd (domain),Sd (sub domain)

% Call the function to retrieve constants
[kB, T, visco] = physical_constants();

%set infinite susceptibility
kaiInfinite=0.9*M_RealExp(end);

%preparing initial magnetic moment data that are equally spaced in log scale.
DH_No=41;%number of magnetic core

%listing all core solution

DH_grid = logspace(log10(DH_min), log10(DH_max), DH_No);
%DH_grid = linspace(DH_min, DH_max, DH_No);

core_start=1e-9;
NewDHList=DH_grid(DH_grid >= core_start);
NewDHList_Diff=diff(NewDHList);

NewDHNo=length(NewDHList);%makesure Kd*Sd<NewDHList

Kd=8;%we set the domain to have Kd sections %optimal kd=10, sd=8
Sd=5;%we set the a domain to have Sd subdomains in it,make sure Kd*Sd<momentNo 8 12 or 6 16

NewDHList_Diff=diff(NewDHList(1:Kd*Sd+1));

%we evaluate 1 subdomain value in each domain, then average it by number of subdomain. 

NewDH=zeros(Kd,1);
Av_DH_dist=[];
M_RealFit=zeros(length(freqs),1);
M_ImagFit=zeros(length(freqs),1);
for kk=1:Sd

    for ll=1:Kd
    % Step 1: Generate core size grid  
    DH_element=kk+(ll-1)*Sd;%picking element in DH List
    NewDH(ll)=NewDHList(DH_element);
    NewDHDiff(ll)=NewDHList_Diff(DH_element);
    end
    

    % Step 3: Construct design matrix A for Langevin curve
    A_Real = zeros(length(freqs), length(NewDH)); % Initialize design matrix
    A_Imag = zeros(length(freqs), length(NewDH)); % Initialize design matrix
    for ii=1:length(freqs)
        for i = 1:length(NewDH)

            [chi_real, chi_imag] = compute_BrownianRelaxation(NewDH(i), freqs(ii), visco, kB, T, kaiInfinite);
            A_Real(ii, i) = A_Real(ii, i) +  NewDHDiff(i).*chi_real;
            A_Imag(ii, i) = A_Imag(ii, i) +  NewDHDiff(i).*chi_imag;
            
        end
    end

    M_Exp = [M_RealExp'; M_ImagExp'];
    
    A = [A_Real; A_Imag];
        
    DHEstimate=lsqnonneg(A,M_Exp)/Sd;         %since the calculation is divided into Sd subdomains, the probability is divided by Sd.
    solution(:,:,kk)=cat(2,NewDH,DHEstimate);
    Av_DH_dist=sortrows([Av_DH_dist;solution(:,:,kk)],1);%combining and sorting the solutions
    
end
DH_grid=Av_DH_dist(:,1);
DH_dist=Av_DH_dist(:,2); %original dist
size(Av_DH_dist);
for i=1:length(freqs)
    for ii=1:length(DH_grid)
           [chi_real, chi_imag] = compute_BrownianRelaxation(DH_grid(ii), freqs(i), visco, kB, T, kaiInfinite);
           M_RealFit(i)= M_RealFit(i)+DH_dist(ii).*NewDHList_Diff(ii)*chi_real;
           M_ImagFit(i)= M_ImagFit(i)+DH_dist(ii).*NewDHList_Diff(ii)*chi_imag;
    end
end

% Compute normalized residual
M_Fit=[M_RealFit;M_ImagFit];
Residuals = M_Exp- M_Fit;

end


    