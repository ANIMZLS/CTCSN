function [LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH] = afb3D_vectorized(x, af1, af2, af3)

% Modified by Mike Wakin from Ivan Selesnick's original function

if nargin < 3
   af2 = af1;
   af3 = af1;
end

% filter along dimension 1
[L, H] = afb3D_A(x, af1, 1);

% filter along dimension 2
[LL LH] = afb3D_A(L, af2, 2);
[HL HH] = afb3D_A(H, af2, 2);

% filter along dimension 3
[LLL LLH] = afb3D_A(LL, af3, 3);
[LHL LHH] = afb3D_A(LH, af3, 3);
[HLL HLH] = afb3D_A(HL, af3, 3);
[HHL HHH] = afb3D_A(HH, af3, 3);

% LOCAL FUNCTION

function [lo, hi] = afb3D_A(x, af, d)

% 3D Analysis Filter Bank
% (along one dimension only)
%
% [lo, hi] = afb3D_A(x, af, d);
% INPUT:
%    x - N1xN2xN2 matrix, where min(N1,N2,N3) > 2*length(filter)
%           (Ni are even)
%    af - analysis filter for the columns
%    af(:, 1) - lowpass filter
%    af(:, 2) - highpass filter
%    d - dimension of filtering (d = 1, 2 or 3)
% OUTPUT:
%     lo, hi - lowpass, highpass subbands
%
% % Example
% x = rand(32,64,16);
% [af, sf] = farras;
% d = 2;
% [lo, hi] = afb3D_A(x, af, d);
% y = sfb3D_A(lo, hi, sf, d);
% err = x - y;
% max(max(max(abs(err))))

lpf = af(:, 1);     % lowpass filter
hpf = af(:, 2);     % highpass filter

% permute dimensions of x so that dimension d is first.
p = mod(d-1+[0:2], 3) + 1;
x = permute(x, p);

% filter along dimension 1
[N1, N2, N3] = size(x);
L = size(af, 1)/2;
x = cshift3D(x, -L, 1);
lo = zeros(L+N1/2, N2, N3);
hi = zeros(L+N1/2, N2, N3);

for k = 1:N3
   lo(:, :, k) = upfirdn(x(:, :, k), lpf, 1, 2);
end
lo(1:L, :, :) = lo(1:L, :, :) + lo([1:L]+N1/2, :, :);
lo = lo(1:N1/2, :, :);

for k = 1:N3
   hi(:, :, k) = upfirdn(x(:, :, k), hpf, 1, 2);
end
hi(1:L, :, :) = hi(1:L, :, :) + hi([1:L]+N1/2, :, :);
hi = hi(1:N1/2, :, :);

% permute dimensions of x (inverse permutation)
lo = ipermute(lo, p);
hi = ipermute(hi, p);

