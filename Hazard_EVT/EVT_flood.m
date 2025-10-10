%% === INPUT: exceedance magnitudes over thresholds ===

clc
clear all
load("Data_Moderate.mat")

% Replace with your vectors (same length N)
H = Data_Moderate(:,1);   % flood HEIGHT exceedances over threshold u_H  (N×1, positive)
D = Data_Moderate(:,2);   % flood DURATION exceedances over threshold u_D (N×1, positive)

Tr=1.405;   % Treshold for peak over treshold selection
assert(all(H>0) && all(D>0), 'H and D must be positive exceedances.');
assert(numel(H)==numel(D), 'H and D must have the same length.');

figure;
scatter(D, H+Tr, 'o', 'filled'); 
xlabel('Flood Duration (hours)');
ylabel('Flood Height (m)'); 
grid on; axis tight;

%% === 1) Fit GPD marginals by MLE (independence across events) ===
% gpfit returns [k (shape), sigma (scale)] and 95% CIs

[paramH,ciH] = gpfit(H);  kH = paramH(1); sH = paramH(2);
[paramD,ciD] = gpfit(D);  kD = paramD(1); sD = paramD(2);

% Log-likelihood, AIC, BIC
nH = numel(H); nD = numel(D);
negLL_H = gplike([kH sH], H);            % total negative log-likelihood
negLL_D = gplike([kD sD], D);
k_params = 2;                            % k (shape) + sigma (scale)
AIC_H = 2*k_params + 2*negLL_H;
AIC_D = 2*k_params + 2*negLL_D;
BIC_H = k_params*log(nH) + 2*negLL_H;
BIC_D = k_params*log(nD) + 2*negLL_D;

% --- KS goodness-of-fit (use a distribution object) ---
pdH = makedist('GeneralizedPareto','k',kH,'sigma',sH,'theta',0);
pdD = makedist('GeneralizedPareto','k',kD,'sigma',sD,'theta',0);

% (Optional but recommended) Support check when k<0:
if kH < 0
    x_max_H = -sH/kH; 
    assert(all(H < x_max_H), 'Some H exceed fitted GPD support (k<0).');
end
if kD < 0
    x_max_D = -sD/kD; 
    assert(all(D < x_max_D), 'Some D exceed fitted GPD support (k<0).');
end

[h_H,p_H,ksstat_H] = kstest(H, 'CDF', pdH);
[h_D,p_D,ksstat_D] = kstest(D, 'CDF', pdD);


%% === 2) Diagnostics: QQ plots against fitted GPD ===
figure;
subplot(1,2,1);
H_sorted = sort(H);
pH = ((1:nH)' - 0.5)/nH;                  % plotting positions
qH = gpinv(pH, kH, sH);                   % fitted quantiles
plot(qH, H_sorted, 'o'); hold on; plot(qH,qH,'-');
xlabel('Theoretical GPD quantiles'); ylabel('Empirical H');
title('Height exceedances: QQ plot'); grid on; axis tight;

subplot(1,2,2);
D_sorted = sort(D);
pD = ((1:nD)' - 0.5)/nD;
qD = gpinv(pD, kD, sD);
plot(qD, D_sorted, 'o'); hold on; plot(qD,qD,'-');
xlabel('Theoretical GPD quantiles'); ylabel('Empirical D');
title('Duration exceedances: QQ plot'); grid on; axis tight;

%% === 3) Dependence: Kendall tau & Spearman rho with bootstrap CIs ===
% Paired bootstrap to preserve H–D dependence
nboot = 5000;   % increase if you want tighter CIs (more compute)
X = [H D];

% Define scalar-stat functions for bootci
bootfun_tau  = @(Z) corr(Z(:,1), Z(:,2), 'Type','Kendall','Rows','complete');
bootfun_rho  = @(Z) corr(Z(:,1), Z(:,2), 'Type','Spearman','Rows','complete');

% BCa confidence intervals (robust)
[CI_tau,  boot_tau] = bootci(nboot, {bootfun_tau, X}, 'type','bca');
[CI_rho,  boot_rho] = bootci(nboot, {bootfun_rho, X}, 'type','bca');

tau_hat = bootfun_tau(X);
rho_hat = bootfun_rho(X);

% Optional: visualize bootstrap sampling distributions
figure;
histogram(boot_tau, 'Normalization','pdf'); xlabel('Kendall \tau'); ylabel('PDF');
title('Bootstrap dist. of Kendall \tau'); grid on;
figure;
histogram(boot_rho, 'Normalization','pdf'); xlabel('Spearman \rho'); ylabel('PDF');
title('Bootstrap dist. of Spearman \rho'); grid on;

%% === 4) Summarize results ===

%decision = ["Fail to reject H0","Reject H0"];          % if h is 0 or 1
fprintf('\n=== GPD MLE (Height exceedances) ===\n');
fprintf('k (shape)  = %.4f  [95%% CI: %.4f, %.4f]\n', kH, ciH(1,1), ciH(2,1));
fprintf('sigma      = %.4f  [95%% CI: %.4f, %.4f]\n', sH, ciH(1,2), ciH(2,2));
fprintf('AIC = %.2f, BIC = %.2f, KS stat = %.4f\n, p-value=%.4f\n, h=%.0f', AIC_H, BIC_H, ksstat_H, p_H, h_H);

fprintf('\n=== GPD MLE (Duration exceedances) ===\n');
fprintf('k (shape)  = %.4f  [95%% CI: %.4f, %.4f]\n', kD, ciD(1,1), ciD(2,1));
fprintf('sigma      = %.4f  [95%% CI: %.4f, %.4f]\n', sD, ciD(1,2), ciD(2,2));
fprintf('AIC = %.2f, BIC = %.2f, KS stat = %.4f\n, p-value=%.4f\n, h=%.0f', AIC_D, BIC_D, ksstat_D, p_D, h_D);

fprintf('\n=== Dependence (paired bootstrap %d) ===\n', nboot);
fprintf('Kendall tau  = %.4f  [95%% CI: %.4f, %.4f]\n', tau_hat, CI_tau(1), CI_tau(2));
fprintf('Spearman rho = %.4f  [95%% CI: %.4f, %.4f]\n', rho_hat, CI_rho(1), CI_rho(2));
%% KS test results 

names   = ["H","D"];
AICs    = [AIC_H, AIC_D];
BICs    = [BIC_H, BIC_D];
ksstats = [ksstat_H, ksstat_D];
pvals   = [p_H, p_D];
hs      = [h_H, h_D];

decision = ["Fail to reject H0","Reject H0"];

for k = 1:numel(names)
    fprintf('%s: AIC = %.2f, BIC = %.2f, KS stat = %.4f, p = %.4f, h=%d (%s)\n', ...
        names(k), AICs(k), BICs(k), ksstats(k), pvals(k), hs(k), decision(hs(k)+1));
end

%% === 6) Monte Carlo simulation from fitted GPDs + histograms ===
N = 1000;                 % number of Monte Carlo samples
rng(42,'twister');       % reproducibility

% Draw samples (theta=0). Either method works:
% simH = gprnd(kH, sH, 0, N, 1);
% simD = gprnd(kD, sD, 0, N, 1);
% or via distribution objects (if you created pdH/pdD above):
pdH = makedist('GeneralizedPareto','k',kH,'sigma',sH,'theta',0);
pdD = makedist('GeneralizedPareto','k',kD,'sigma',sD,'theta',0);
simH = random(pdH, N, 1);
simD = random(pdD, N, 1);

% Support upper bound if k<0 (finite tail) — trim plotting range accordingly
xmaxH = (kH < 0) * (-sH/kH) + (kH >= 0) * prctile(simH, 99.5);
xmaxD = (kD < 0) * (-sD/kD) + (kD >= 0) * prctile(simD, 99.5);

% Plot histogram (PDF-normalized) + fitted PDF overlay
figure;
tiledlayout(1,2, 'Padding','compact', 'TileSpacing','compact');

% --- Height ---
nexttile;
histogram(simH, 'Normalization','pdf', 'EdgeColor','none'); hold on;
xH = linspace(0, xmaxH, 600);
plot(xH, gppdf(xH, kH, sH), 'LineWidth', 1.8);
xlabel('Flood height exceeding tr (m)'); ylabel('PDF');
title(sprintf('Monte Carlo (N = %d): H', N)); grid on; box on;
legend('MC histogram','Fitted GPD PDF', 'Location','best');

% --- Duration ---
nexttile;
histogram(simD, 'Normalization','pdf', 'EdgeColor','none'); hold on;
xD = linspace(0, xmaxD, 600);
plot(xD, gppdf(xD, kD, sD), 'LineWidth', 1.8);
xlabel('Flood duration (h)'); ylabel('PDF');
title(sprintf('Monte Carlo (N = %d): D', N)); grid on; box on;
legend('MC histogram','Fitted GPD PDF', 'Location','best');

% === Summary stats: sample vs. theoretical (when defined) ===
mH_th = NaN; vH_th = NaN;
mD_th = NaN; vD_th = NaN;
if kH < 1,  mH_th = sH/(1 - kH);   end            % theoretical mean exists if k < 1
if kH < 1/2, vH_th = sH^2 / ((1 - kH)^2 * (1 - 2*kH)); end   % variance exists if k < 1/2
if kD < 1,  mD_th = sD/(1 - kD);   end
if kD < 1/2, vD_th = sD^2 / ((1 - kD)^2 * (1 - 2*kD)); end

fprintf('\n=== Monte Carlo summary (N = %d) ===\n', N);
fprintf('H: mean(sample)=%.4f,  var(sample)=%.4f,  mean(th)=%.4f, var(th)=%.4f\n', ...
        mean(simH), var(simH), mH_th, vH_th);
fprintf('D: mean(sample)=%.4f,  var(sample)=%.4f,  mean(th)=%.4f, var(th)=%.4f\n', ...
        mean(simD), var(simD), mD_th, vD_th);

%% === 5) (Optional) Return levels for POT ===
% If you know the exceedance rate lambda (events per year), you can estimate
% T-year return levels z_T via the GPD quantile for p = 1 - 1/(lambda*T).
% Example (set your lambda):
% lambda = 3;  % e.g., 3 exceedances/year
% T = [2 5 10 20 50 100]';
% p  = 1 - 1./(lambda*T);
% zT_H = gpinv(p, kH, sH);   % return levels for H-exceedances over u_H
% zT_D = gpinv(p, kD, sD);   % return levels for D-exceedances over u_D
% table(T, zT_H, zT_D)
