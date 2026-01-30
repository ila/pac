-- Test numeric accuracy of approximate pac_sum vs exact SUM
-- pac_sum with hash(1) and mi=0, divide by 2 since hash(1) has all bits set

-- Create test table with various distributions
CREATE OR REPLACE TABLE accuracy_test AS
WITH uniform_tiny   AS (SELECT 'uniform_tinyint' AS dist, (i % 127)::TINYINT AS val FROM range(1000000) t(i)),
     uniform_small  AS (SELECT 'uniform_smallint' AS dist, (i % 32767)::SMALLINT AS val FROM range(1000000) t(i)),
     uniform_int    AS (SELECT 'uniform_int' AS dist, (i % 2147483647)::INTEGER AS val FROM range(1000000) t(i)),
     uniform_bigint AS (SELECT 'uniform_bigint' AS dist, i::BIGINT AS val FROM range(1000000) t(i)),
     zipf_like      AS (SELECT 'zipf_like' AS dist, (1000000.0 / (i + 1))::BIGINT AS val FROM range(1000000) t(i)),
     exponential    AS (SELECT 'exponential' AS dist, (EXP(random() * 10))::BIGINT AS val FROM range(1000000) t(i)),
     bimodal        AS (SELECT 'bimodal' AS dist, CASE WHEN i % 2 = 0 THEN (i % 100)::BIGINT ELSE (1000000 + i % 100)::BIGINT END AS val FROM range(1000000) t(i)),
     sparse_large   AS (SELECT 'sparse_large' AS dist, CASE WHEN i % 1000 = 0 THEN 1000000::BIGINT ELSE 0::BIGINT END AS val FROM range(1000000) t(i)),
     negative_mixed AS (SELECT 'negative_mixed' AS dist, CASE WHEN i % 2 = 0 THEN i::BIGINT ELSE -i::BIGINT END AS val FROM range(1000000) t(i)),
     all_same       AS (SELECT 'all_same' AS dist, 42::BIGINT AS val FROM range(1000000) t(i))
SELECT * FROM uniform_tiny
UNION ALL SELECT * FROM uniform_small
UNION ALL SELECT * FROM uniform_int
UNION ALL SELECT * FROM uniform_bigint
UNION ALL SELECT * FROM zipf_like
UNION ALL SELECT * FROM exponential
UNION ALL SELECT * FROM bimodal
UNION ALL SELECT * FROM sparse_large
UNION ALL SELECT * FROM negative_mixed
UNION ALL SELECT * FROM all_same;

-- It is tricky to characterize the precision of an approximate if the expected mean is 0. Because if you divide ApproxResult / ExpectedMean then you divide by 0 and nothing makes sense (this is roughly the case for the negative_mixed distribution which is 0, -1, 2, -3, ... ,-N, N+1) 
-- So people use the RMSE (Root Mean Squared Error) / variance, if you can run an experiment many times. Well, we can run it 64 times, of course. RMSE is ((exact_sum[i]/pac_sum_counters[i])^2/64). You can further divide by variance: z_square = RMSE / variance(pac_sum_counters[]). 

-- test query:
WITH bit_positions AS (SELECT unnest(range(64)) as bit_pos),
    exact_sums AS (SELECT dist, bit_pos, sum(val * ((hash(rowid) >> bit_pos) & 1))::DOUBLE as exact_sum
                   FROM accuracy_test, bit_positions GROUP BY dist, bit_pos),
    exact_lists AS (SELECT dist, list(exact_sum ORDER BY bit_pos) as exact_counters
                    FROM exact_sums GROUP BY dist),
    approx_sums AS (SELECT dist, pac_sum_counters(hash(rowid), val, 0.0) as approx_counters
                    FROM accuracy_test GROUP BY dist),
    per_counter_stats AS (
        SELECT e.dist, e.exact_counters, a.approx_counters,
               [POWER(e.exact_counters[x+1] - a.approx_counters[x+1], 2) FOR x IN range(64)] as sq_diffs,
               [CASE WHEN e.exact_counters[x+1] = 0 THEN NULL
                     ELSE 100.0 * (a.approx_counters[x+1] - e.exact_counters[x+1]) / e.exact_counters[x+1]
                END FOR x IN range(64)] as pct_diffs
        FROM exact_lists e JOIN approx_sums a ON e.dist = a.dist)
    SELECT
        dist, cast(sqrt(list_avg(sq_diffs)) as bigint) as rmse,
        ROUND(list_avg(list_transform(pct_diffs, x -> abs(x))), 4) as pct_err,
        ROUND(POWER(rmse, 2) / NULLIF(list_var_samp(approx_counters), 0), 4) as z_squared,
        cast(list_var_samp(exact_counters) as float) as var_exact,
        cast(list_var_samp(approx_counters) as float) as var_approx,
        cast(var_exact / var_approx as float) as var_ratio
    FROM per_counter_stats ORDER BY dist; 
-- which computes the percentual difference, but also this RMSE/variance, which is known as "z_squared". 

-- reminder: this query takes 64 random 50% samples of the distribution, and computes the exact sum of that, as well as the approximated sum (using pac_sum_couters, so we take the 64 non-noised counter values). So handy, that function! 

-- using: binaries/duckdb_signedsum
-- ┌──────────────────┬───────────┬──────────┬───────────┬───────────────────┬───────────────────┬────────────┐
-- │       dist       │   rmse    │ pct_err  │ z_squared │     var_exact     │    var_approx     │ var_ratio  │
-- │     varchar      │   int64   │  double  │  double   │       float       │       float       │   float    │
-- ├──────────────────┼───────────┼──────────┼───────────┼───────────────────┼───────────────────┼────────────┤
-- │ all_same         │     34739 │   0.1648 │    2.5222 │       480663500.0 │       478479400.0 │  1.0045646 │
-- │ bimodal          │ 506582657 │   0.2024 │    2.1763 │      1.146621e+17 │     1.1791784e+17 │ 0.97238976 │
-- │ exponential      │   3330793 │   0.2993 │    1.9499 │   5233842300000.0 │   5689536000000.0 │  0.9199067 │
-- │ negative_mixed   │ 253518956 │  113.2954 │  13.3062 │      6.323951e+16 │ 301312700000000.0 │     209.88 │
-- │ sparse_large     │    495405 │    0.098 │    0.0007 │ 350350950000000.0 │ 350088250000000.0 │  1.0007504 │
-- │ uniform_bigint   │ 531127114 │   0.2122 │    3.7432 │      7.294264e+16 │      7.536269e+16 │ 0.96788794 │
-- │ uniform_int      │ 535868708 │   0.2142 │    3.5841 │     8.0645055e+16 │     8.0119565e+16 │  1.0065588 │
-- │ uniform_smallint │  19266000 │   0.2369 │    5.3011 │  67736760000000.0 │  70019694000000.0 │ 0.96739584 │
-- │ uniform_tinyint  │     45045 │   0.1426 │    1.9842 │       942674600.0 │      1022611260.0 │  0.9218309 │
-- │ zipf_like        │      9477 │   0.1296 │    0.0002 │    385481740000.0 │    385250720000.0 │  1.0005996 │
-- ├──────────────────┴───────────┴──────────┴───────────┴───────────────────┴───────────────────┴────────────┤
-- │ 10 rows                                                                                        7 columns │
-- └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘ 

-- using: binaries/duckdb_default
-- ┌──────────────────┬───────────┬─────────┬───────────┬───────────────────┬───────────────────┬────────────┐
-- │       dist       │   rmse    │ pct_err │ z_squared │     var_exact     │    var_approx     │ var_ratio  │
-- │     varchar      │   int64   │ double  │  double   │       float       │       float       │   float    │
-- ├──────────────────┼───────────┼─────────┼───────────┼───────────────────┼───────────────────┼────────────┤
-- │ all_same         │     15656 │  0.0739 │    0.5099 │       480663500.0 │       496122080.0 │ 0.96884114 │
-- │ bimodal          │ 265348395 │   0.106 │    0.6141 │      1.146621e+17 │     1.1553983e+17 │ 0.99240327 │
-- │ exponential      │   1572928 │  0.1422 │    0.4727 │   5233842300000.0 │   5262080500000.0 │  0.9946337 │
-- │ negative_mixed   │  15924302 │ 22.3675 │     0.004 │      6.323951e+16 │      6.390592e+16 │ 0.98957205 │
-- │ sparse_large     │    265290 │  0.0524 │    0.0002 │ 350350950000000.0 │ 350576270000000.0 │  0.9993573 │
-- │ uniform_bigint   │ 298083839 │   0.119 │    1.2181 │      7.294264e+16 │     7.1160045e+16 │  1.0250505 │
-- │ uniform_int      │ 300003011 │  0.1199 │     1.116 │     8.0645055e+16 │      8.085194e+16 │  0.9974412 │
-- │ uniform_smallint │  11108176 │  0.1365 │    1.8216 │  67736760000000.0 │  67592694000000.0 │  1.0021313 │
-- │ uniform_tinyint  │     22192 │  0.0701 │    0.5224 │       942674600.0 │       928901000.0 │  1.0148278 │
-- │ zipf_like        │      3673 │  0.0507 │       0.0 │    385481740000.0 │    385352560000.0 │  1.0003352 │
-- ├──────────────────┴───────────┴─────────┴───────────┴───────────────────┴───────────────────┴────────────┤
-- │ 10 rows                                                                                       7 columns │
-- └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
-- so for negative_mixed the z_squared improves from 13.3 to  0.004! 

-- We also see that that for negative_mixed the var_approx was in fact much smaller than var_exact (var_ratio). This smaller variance sounds good but is bad. 

-- the problem appears to be that due to the approximation the totals were collapsing to 0. Yes, they exact sums for negative_mixed are also centered around 0, but they are not zero, they vary around it (quite wildly). If the approx always returns sums close to zero, the variance is small. But the variance of the exact sums was not small. 
-- The big variances in the exact totals are good, in that they make the z_mean become small (because dividing by that big variance makes the big RMSEs small). But the collapse of the approximation causes unnatural small variance, which results in this very bad z_squared of 209 

-- The percentual errors also improve by a factor 2. This is because the two counters for pos and neg are unsigned counters of positive values (because the neg counters contain -values so they are also positive). And unsigned counters have one extra bit of precision for positive values and that extra bit is a factor 2 in precision. 
