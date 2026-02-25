%_________________________________________________________________________
%   Original Author and programmer: Mohd Mawardi Saari, 
%                               Universiti Malaysia Pahang
%                               mmawardi@umpsa.edu.my
%_________________________________________________________________________
clc; clear; close all;

%Define the range of the hydrodynamic distribution and measurement
%frequency range and sampling points.

DH_min=10e-9;
DH_max=200e-9;
DH_int=1e-9;
freq_min=10;
freq_max=1e6;
freq_no=50;

% Function for testing two distrbitution cases, 
% case 1: closely mixed distribution, 
% case 2 separated and has close distribution peak height

DistTestSNNLS(freq_min, freq_max, DH_min, DH_max);




