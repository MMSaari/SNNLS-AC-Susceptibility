function JS_div = js_divergence(d_grid_SPNNLS, diam_dist_SPNNLS, mu_lognorm1, sigma_lognorm1, mu_lognorm2, sigma_lognorm2, distributionRatio)
    % Compute the theoretical reference distribution (lognormal mixture)
    f_ref = distributionRatio * lognpdf(d_grid_SPNNLS, mu_lognorm1, sigma_lognorm1) + ...
            (1 - distributionRatio) * lognpdf(d_grid_SPNNLS, mu_lognorm2, sigma_lognorm2);

    % Normalize both distributions to sum to 1
    P = diam_dist_SPNNLS / sum(diam_dist_SPNNLS);
    Q = f_ref / sum(f_ref);

    % Compute the midpoint distribution M
    M = 0.5 * (P + Q);

    % Compute JS divergence, avoiding log(0) issues
    JS_div = 0.5 * sum(P .* log(P ./ M), 'omitnan') + ...
             0.5 * sum(Q .* log(Q ./ M), 'omitnan');
end